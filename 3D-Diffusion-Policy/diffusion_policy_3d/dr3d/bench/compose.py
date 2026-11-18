#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""dr3d bench scene composer + SHARED lazy cache (V0 type generalization).

Repo keeps only source assets under distractors/. Full <base>+distractor scenes
are NEVER committed: they are composed on first use and cached under
    dr3d/tmp/scenes/dr3d_<base>_scene_<type>_<asset_hash>_<pos_hash>.xml (+.sha256)
(git-ignored). <base> is part of the cache name, so different tasks never
overwrite each other's scene for the same type/pos.

The asset hash makes any edit of distractor/<type>/... invalidate the cache
automatically; re-reading a scene = (seed, run, ep) -> type -> same cache file,
so order is reproducible from seed+rng with a shared per-type cache.

Composed scenes are self-contained XML (includes expanded, meshdir/texturedir made
absolute -- see test_load.selfcontain_base_text), therefore:
  * mujoco validation runs in this cache dir (third_party assets stay read-only);
  * the scene can be written to any private path and loaded by scene_runtime.

CLI only for manual checks; eval-time entry is bench/run_eval.py -> ensure_scene.
"""
import argparse
import hashlib
import os
import pathlib
import sys

DR3D = pathlib.Path(__file__).resolve().parents[1]
if str(DR3D) not in sys.path:
    sys.path.insert(0, str(DR3D))
import test_load as TL

TYPES = ["cube", "can", "bottle", "cereal", "mug"]     # catgen type order (do not change)
TYPE_XML = {
    "cube": DR3D / "distractors" / "cube" / "type_cube.xml",
    "can": DR3D / "distractors" / "can" / "model.xml",
    "bottle": DR3D / "distractors" / "bottle" / "model.xml",
    "cereal": DR3D / "distractors" / "cereal" / "model.xml",
    "mug": DR3D / "distractors" / "mug" / "model.xml",
    # not a catgen asset: sphere used by the dynamic distractors
    # (injected as mocap/kinematic, see bench/dynamic.py)
    "sphere": DR3D / "distractors" / "sphere" / "type_sphere.xml",
    # inspire left hand for the dynamic distractors
    # (12 finger hinges; zero contact), see bench/dynamic_hand.py
    "inspire": DR3D / "distractors" / "inspire" / "model.xml",
}
# cube inertia (mass 0.15, box half 0.03)
MASS = {"cube": 0.15}
DIAG = {"cube": 5e-4}
DEFAULT_CACHE = DR3D / "tmp" / "scenes"
# Cache key must cover *composer code* changes, not just the source asset: `_asset_tag`
# hashes the distractor asset file only, and `ensure_scene`'s integrity check compares a
# cached file against its own sidecar -- so a scene text written by an older composer
# would look like a permanent hit and never be recomputed (dr3d/tmp/scenes does contain such
# Oct-2 texts, e.g. an injected body at the FIRST </worldbody> instead of the last).
# Bump this whenever compose/test_load change the produced text.
_COMPOSER_VER = "c2"
POS_DEFAULT = "-0.22 -0.15"      # P0 (shared by hammer/door; screen-right = world -x,
                                 # top-camera euler z=pi verified numerically);
                                 # posgen passes random positions
# P0 for the metaworld assembly table (table spans x in [-0.7, 0.7],
# y in [0.2, 1.0], top at z=0; peg at (0, 0.6), hole |x|<=0.1, y in [0.75, 0.85]).
# Calibrated against the corner2 point cloud via single-env toggle diff
# (/hy-tmp/3DP/calib_toggle.py): corner2 only sees the LEFT half of the table,
# so P0 is on the front-left corner (-0.45, 0.35): 48 cube points survive the
# policy crop, cube rests on the table (z=0.03), occludes nothing (occ=0), and
# is well outside the arm working zone (|x|<=0.2, y in [0.5, 0.9]).
POS_ASSEMBLY = "-0.45 0.35"     # P0(metaworld assembly; calibrated)
# stick-push and assembly share basic_scene.xml (same table) and xyz_base.xml (same
# corner2 + the wrapper's cam_pos override) => identical crop/frustum, P0 is reused.
# Real-render toggle diff (seed=0): cube@P0 adds 50 points (22 rows of FPS512) and the
# cluster centroid matches assembly@P0's cube exactly (0.6131,-0.8345,-0.7538).
POS_STICK_PUSH = POS_ASSEMBLY
# disassemble-v2 (SawyerNutDisassembleEnvV2) resolves model_name to the SAME
# sawyer_xyz/sawyer_assembly_peg.xml as assembly-v2 -> identical table, corner2
# camera and crop bounds, so POS_ASSEMBLY and its scene cache sha are reused.
POS_DISASSEMBLE = POS_ASSEMBLY
# metaworld task id -> (catgen base scene key, P0). The injection must land on THIS
# task's XML: stick-push keeps its manipulated object in qpos[9:12] and the stick in
# qpos[16:18], nothing like assembly's RoundNut/peg -- a wrong base corrupts the
# sawyer_xyz_env reset. disassemble IS the same MJCF as assembly (POS_DISASSEMBLE),
# so the two share one base key.
MW_SCENES = {
    "assembly": ("assembly", POS_ASSEMBLY),
    "stick-push": ("stick_push", POS_STICK_PUSH),
    "disassemble": ("assembly", POS_DISASSEMBLE),
}


def mw_scene(task):
    """metaworld task id -> (base, P0); unregistered tasks raise (no assembly fallback)."""
    if task not in MW_SCENES:
        raise ValueError("metaworld catgen base: unsupported task %r "
                         "(supported: %s)" % (task, sorted(MW_SCENES)))
    return MW_SCENES[task]


BASE = TL.BASE_XML                 # {'hammer': DAPG_hammer.xml, 'door': DAPG_door.xml}


def _norm_pos(pos):
    """Normalize a position: (x, y) tuple/list/ndarray/'x y'/'x,y' -> ('x y', 8-char hash).

    Single entry point, so an upstream tuple does not break the parse
    (str((0.26, -0.3)) == "(0.26, -0.3)").
    """
    if isinstance(pos, (tuple, list)) or hasattr(pos, "shape"):
        vals = [float(v) for v in pos]
    else:
        s = str(pos).replace(",", " ").replace("(", " ").replace(")", " ")
        vals = [float(v) for v in s.split()]
    if len(vals) != 2:
        raise ValueError("pos must be (x, y), got %r" % (pos,))
    p = "%.4f %.4f" % (vals[0], vals[1])
    return p, hashlib.sha256(p.encode("utf-8")).hexdigest()[:8]


def _atomic_write(path, text):
    """Concurrency-safe write: temp file in the same dir, then os.replace (atomic)."""
    path = pathlib.Path(path)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(path))


def _pos_tag(pos, mocap=False, scale=1.0, free=True):
    """Position (+ mocap flag, + non-1.0 scale) -> (normalized position, 8-char hash).

    The hash is part of the cache name: different positions / the same position
    injected as free body vs mocap / a non-1.0 mesh scale must be cached
    separately, otherwise they overwrite each other. (scale=1.0 stays out of
    the key, so historical cache names are unchanged.)
    """
    p, _ = _norm_pos(pos)
    key = p + ("|mocap" if mocap else ("" if free else "|static"))
    if abs(float(scale) - 1.0) > 1e-9:
        key += "|s%g" % float(scale)
    return p, hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]


def _asset_tag(type_name):
    """8-char hash of the source asset content (type_*.xml / model.xml).

    Must be part of the cache name: otherwise an edited asset (e.g. ball
    contype/conaffinity) keeps reusing the old scene and the change looks
    like a no-op.
    """
    return hashlib.sha256(TYPE_XML[type_name].read_bytes()).hexdigest()[:8]


def _cache_paths(cache_root, type_name, pos, mocap=False, scale=1.0, base="hammer",
                 free=True):
    """Paths of the cached scene and its sidecar
    (name = base scene + type + asset hash + pos/mocap/scale hash)."""
    _p, tag = _pos_tag(pos, mocap, scale, free)
    name = (f"dr3d_{_COMPOSER_VER}_{base}_scene_{type_name}_"
            f"{_asset_tag(type_name)}_{tag}.xml")
    return cache_root / name, cache_root / (name + ".sha256")


def compose_scene(type_name, pos, out_dir=None, check=True, mocap=False, scale=1.0,
                  base="hammer", free=True):
    """Compose scene text deterministically; write to out_dir (default cache).

    mocap=True injects the distractor as a kinematic body (no joint, driven by
    mocap_pos) for the dynamic suites (dynsphere/dynhand); it is part of the cache
    name, so it never mixes with the free-body variant.
    scale != 1.0 scales every mesh of that type (also part of the cache name).
    base: 'hammer' | 'door' -> base scene XML (test_load.BASE_XML[base]).
    """
    # --- injection mode (details in test_load.compose_hammer_object) ------------
    # free=False = STATIC (no joint, no mocap): the only workable injection on the
    # metaworld side.
    #   * sawyer_xyz_env hardcodes the object at qpos[9:12] -> no <freejoint>;
    #   * set_xyz_action reads all of data.mocap_pos   -> exactly one mocap body.
    #   The body is appended to the LAST </worldbody> (assembly expands to 3 of
    #   them); an asset that carries joints (e.g. dynhand's inspire hand) gets
    #   compiler boundmass/boundinertia, avoiding 0-mass/0-inertia errors caused by
    #   the base scene's inertiagrouprange excluding that geom group; joint-free
    #   assets (catgen/posgen/dynsphere) get no patch -> the composed text stays
    #   byte-identical.
    # base supports 'hammer' | 'door' | 'pen' | 'assembly' | 'stick_push' (the
    # metaworld table follows the task, see mw_scene); injected bodies that clash
    # with base body names are renamed (test_load._rename_conflicting_bodies).
    assert type_name in TYPE_XML, (type_name, list(TYPE_XML))
    assert base in BASE, (base, list(BASE))
    if out_dir is None:
        out_dir = DEFAULT_CACHE
    out_dir = pathlib.Path(out_dir)
    pos, _tag = _norm_pos(pos)          # tuple/list/str -> "x y"
    xml_path = TYPE_XML[type_name]
    tmp = TL.compose_hammer_object(
        type_name, xml_path, pos, base_xml=BASE[base], scale=float(scale),
        mass=MASS.get(type_name, 0.1), diag=DIAG.get(type_name, 1e-4),
        mocap=mocap, free=free)
    text = pathlib.Path(tmp).read_text(encoding="utf-8", errors="ignore")
    os.remove(tmp)
    # validation: the scene is self-contained, so loading it in the cache dir is
    # enough (no need to stage it under third_party assets)
    if check:
        from mujoco_py import load_model_from_path
        out_dir.mkdir(parents=True, exist_ok=True)
        chk = out_dir / f".dr3d_chk_{base}_{type_name}_{os.getpid()}.xml"
        chk.write_text(text, encoding="utf-8")
        try:
            load_model_from_path(str(chk))
        finally:
            chk.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    out, sha_f = _cache_paths(out_dir, type_name, pos, mocap, scale, base, free)
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    _atomic_write(out, text)
    _atomic_write(sha_f, sha)
    return out, sha


def ensure_scene(type_name, pos=POS_DEFAULT, cache_root=None, mocap=False, scale=1.0,
                 base="hammer", free=True):
    """Shared lazy cache: compose once, reuse while the sha matches
    (keyed by position + mocap + scale + base scene)."""
    cache_root = pathlib.Path(cache_root) if cache_root else DEFAULT_CACHE
    cache_root.mkdir(parents=True, exist_ok=True)
    out, sha_f = _cache_paths(cache_root, type_name, pos, mocap, scale, base, free)
    if out.exists() and sha_f.exists():
        old_sha = sha_f.read_text().strip()
        if hashlib.sha256(out.read_bytes()).hexdigest() == old_sha:
            return out, old_sha, False          # cache hit, no recompose
    out, sha = compose_scene(type_name, pos, cache_root, mocap=mocap, scale=scale,
                             base=base, free=free)
    return out, sha, True                       # composed/cached now


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default=",".join(TYPES))
    ap.add_argument("--pos", default="-0.22,-0.15")
    ap.add_argument("--base", default="hammer", choices=sorted(BASE),
                    help="base scene: hammer | door")
    ap.add_argument("--out", default=str(DEFAULT_CACHE))
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args()
    pos = a.pos.replace(",", " ")
    for t in [s.strip() for s in a.types.split(",") if s.strip()]:
        out, sha, fresh = ensure_scene(t, pos, pathlib.Path(a.out), base=a.base)
        print(f"[scene] {a.base}/{t} -> {out}  sha256={sha[:16]}  pos={a.pos}"
              f"  {'cached-now' if fresh else 'cache-hit'}")


if __name__ == "__main__":
    main()
