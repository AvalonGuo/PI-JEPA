"""Pre-FPS distractor cluster injection (standalone, default off).

Pipeline
--------
``env/metaworld/metaworld_wrapper.py::get_point_cloud`` renders the full cloud
(transform -> crop) and then FPS-samples it to ``num_points``.  This module is
hooked in *between* those two steps and does exactly one thing::

    cloud_out = fps_fn( concat([cloud_fullpc, cluster_rows]) )

No existing numeric logic is touched: the hook is a ``None``-defaulted kwarg, so
with the injector disabled the code path is the historical one, line by line.

Semantics
---------
* ``points`` (N) is the number of *candidate* rows per cluster, **not** a quota.
  The original FPS decides how many survive; nothing is counted, nothing is
  repaired and no real row is overwritten.  The rows are *appended*, so the FPS
  seed row (``random_start_point=False`` picks the first input row) and the
  relative order of the real rows are unchanged.
* radius follows an *area* law, because a depth camera samples visible surfaces::

      r(N) = s_ref * sqrt(N / (2*pi))        # equal-area upper hemisphere

  i.e. the in-cluster point spacing equals ``s_ref``.  ``s_ref`` defaults to the
  task's own measured FPS point spacing (``delta_fps``, 0.86 cm for
  metaworld_assembly), so N reads as "expected number of FPS slots".
  assembly, s_ref = 0.86 cm: N = 20 / 50 / 100 -> r = 1.55 / 2.45 / 3.46 cm
  (diameter 3.1 / 4.9 / 6.9 cm; the legacy posgen cube is 6 cm).
  Because the injected shell is *sparser* than the raw cloud (delta_raw =
  0.59 cm), the FPS prefers those rows: measured on the assembly zarr
  (test/verify_cluster_distractor.py) 98-100 % of the injected rows survive the
  1024-row FPS, so N is (to first order) also the number of *occupied* slots.
* the crop box of ``MetaWorldEnv`` (``TASK_BOUDNS['default']``) is tighter than
  the table: measured clearance of the green cells is 2.94 / 3.39 / 3.69 cm
  (p10/p50/p90).  The cluster centre is therefore *clamped* into
  ``[crop_lo + r, crop_hi - r]`` so every injected row stays inside the
  observation region while ``r`` stays fixed (no per-episode dose drift).
* deterministic: ``RandomState(seed + 10429 + episode)``; the calibration
  artefact is frozen, so the geometry is reproducible across runs and machines.

Families
--------
* ``metaworld_*``: the crop box of ``MetaWorldEnv`` plus the corner2 camera
  (``green_cells_assembly``).
* ``adroit_*`` (Option A, reuses the suite's ``green_cells``): the world xy cells
  of ``green_cells.load_green_cells(<task>)`` are mapped through the *eval* chain
  of ``mjpc_diffusion_wrapper.py``.  Folding ``scale``/``offset`` into the affine
  keeps the injector byte-for-byte identical::

      M   = pc_transform.T @ diag(scale)
      cam = cam_pos - cam_body_pos - offset @ inv(M)
      cloud = (world - cam) @ M          # == cam_geom.to_cloud  (|diff| < 1e-15 m)

  Adroit's crop box is only bounded in z (x/y are +-10), so the clamp is a no-op
  there, and every cell is additionally filtered to those whose *full* AABB stays
  inside the crop box **and** the camera frustum (same criterion as
  ``cam_geom.aabb_visible``, which is what the posgen suite uses to place cubes),
  so a partially clipped cluster is never sampled.

Usage
-----
Calibration (once; reads the task zarr, needs mujoco only for the camera matrix)::

    python -m diffusion_policy_3d.cluster_distractor --task metaworld_assembly --calibrate
    python -m diffusion_policy_3d.cluster_distractor --task adroit_hammer --calibrate

Offline self-check (no mujoco / zarr needed once calibrated)::

    python -m diffusion_policy_3d.cluster_distractor --task adroit_hammer --check
"""
import math
import os
import pathlib
import sys

import numpy as np

__all__ = ["ClusterInjector", "radius_for", "calibrate", "load_spec",
           "DEFAULTS", "GOLDEN_ANGLE", "OBSTACLE_RGB"]

GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))

# same constant as voxel/voxelizer.py::OBSTACLE_RGB (xyz-only consumers ignore it)
OBSTACLE_RGB = (210.0, 160.0, 59.0)

# fallbacks, measured on metaworld_assembly_expert.zarr (see --calibrate);
# values are cm, i.e. 100 * the median nearest-neighbour distance
DELTA_FPS_CM = 0.86     # FPS'd cloud (`data/point_cloud`, 1024 rows)
DELTA_RAW_CM = 0.60     # full cloud (`data/raw_point_cloud`, 1564-2737 rows)

DEFAULTS = dict(points=0, clusters=1, radius_mode="fps", radius_cm=None,
                s_ref_cm=None, rgb=OBSTACLE_RGB, min_gap_cm=3.0, seed=0)

# --- adroit (Option A) -------------------------------------------------------
ADROIT_FAMILY = "adroit"
ADROIT_Z_PLACE = 0.0      # table surface; the 1033 hammer cells are in-crop for z=0
ADROIT_GS = 0.02          # green_cells grid pitch (m)

# Fraction of green cells whose full AABB stays in crop+frustum, measured on
# adroit_hammer (1033 cells, z_place=0, aabb_visible).  Used by --check as a
# drift alarm, not as a tuning target: a camera/config change would move these.
EXPECT_VISIBLE = {"adroit_hammer": {20: 1.00, 50: 0.94, 100: 0.85, 200: 0.72}}
VISIBLE_FLOOR = 0.70


def radius_for(n_points, s_ref_cm=DELTA_FPS_CM):
    """Area law: equal-area upper hemisphere whose NN spacing is ``s_ref_cm``."""
    n = float(n_points)
    if n <= 0.0:
        return 0.0
    return float(s_ref_cm) * math.sqrt(n / (2.0 * math.pi))


def _bench():
    """``dr3d/bench/green_cells_assembly``: crop box, camera, cached green cells."""
    bench = pathlib.Path(__file__).resolve().parent / "dr3d" / "bench"
    if str(bench) not in sys.path:
        sys.path.insert(0, str(bench))
    import green_cells_assembly
    return green_cells_assembly


def _spec_default_path(task):
    return _bench().DR3D / "tmp" / f"cluster_cells_{task}.npz"


def _measure_delta(zarr_path, frames=8):
    """(delta_fps_cm, delta_raw_cm) from a collected fullpc zarr.

    ``delta`` is the median nearest-neighbour distance, i.e. the natural scale
    for "how far apart two rows of this cloud are".  The FPS'd cloud sets the
    finest spacing that still survives the sampler, hence it is the default
    ``s_ref``.
    """
    import zarr
    from scipy.spatial import cKDTree

    root = zarr.open(str(zarr_path), "r")
    d = root["data"] if "data" in root else root
    total = int(d["point_cloud"].shape[0])
    idx = np.unique(np.linspace(0, total - 1, min(int(frames), total)).astype(int))
    fps_nn, raw_nn = [], []
    for f in idx:
        pc = np.asarray(d["point_cloud"][int(f)], np.float64)
        pc = pc[np.any(pc != 0.0, axis=1)]
        if len(pc) > 8:
            fps_nn.append(np.median(cKDTree(pc[:, :3]).query(pc[:, :3], k=2)[0][:, 1]))
        if "raw_point_cloud" in d:
            n = int(np.asarray(d["raw_point_cloud_len"][int(f)]))
            raw = np.asarray(d["raw_point_cloud"][int(f)], np.float64).reshape(-1, 6)[:n]
            if len(raw) > 8:
                raw_nn.append(np.median(cKDTree(raw[:, :3]).query(raw[:, :3], k=2)[0][:, 1]))
    df = 100.0 * float(np.median(fps_nn)) if fps_nn else DELTA_FPS_CM
    dr = 100.0 * float(np.median(raw_nn)) if raw_nn else DELTA_RAW_CM
    return df, dr


def _is_adroit(task):
    return str(task).startswith("adroit_")


def _affine(pc_transform, scale):
    """``M`` of ``cloud = (world - cam) @ M`` (row-vector form of to_cloud)."""
    M = np.eye(3) if pc_transform is None else \
        np.asarray(pc_transform, np.float64).T
    if scale is not None:
        M = M @ np.diag(np.asarray(scale, np.float64))
    return M


def _adroit_geometry(task, xml_path=None, z_place=ADROIT_Z_PLACE, verbose=True):
    """Option A: ``green_cells`` world xy + the eval camera/wrapper affine.

    Reproduces ``cam_geom.to_cloud`` exactly (verified to 2e-16 m against
    ``MujocoPointcloudWrapperAdroit.get_point_cloud``: transform -> scale ->
    offset -> crop), so ``(world - cam) @ M`` is the frame the FPS actually sees.
    The camera intrinsics/extrinsics are kept in the spec only for the
    crop+frustum cell filter (``ClusterInjector._visible_mask``).
    """
    _bench()                                     # sys.path side effect only
    import cam_geom
    import green_cells
    import test_load
    from mujoco_py import load_model_from_path
    from diffusion_policy_3d.gym_util.mjpc_diffusion_wrapper import \
        ENV_POINT_CLOUD_CONFIG

    sub = str(task).split("_", 1)[-1]            # adroit_hammer -> hammer
    if sub not in test_load.BASE_XML:
        raise ValueError(f"cluster_distractor: no base XML for {task!r} "
                         f"(available: {sorted(test_load.BASE_XML)})")
    cfg = ENV_POINT_CLOUD_CONFIG[str(task)]
    model = load_model_from_path(str(xml_path or test_load.BASE_XML[sub]))
    g = cam_geom.find_geometry(model, cfg["cam_names"], img=cam_geom.IMG_DEFAULT)
    M = _affine(cfg.get("transform"), cfg.get("scale"))
    offset = cfg.get("offset")
    cam = np.asarray(g["pos"], np.float64) - np.asarray(g["body_pos"], np.float64)
    if offset is not None:
        cam = cam - np.asarray(offset, np.float64) @ np.linalg.inv(M)
    cells = np.asarray(green_cells.load_green_cells(task=sub, verbose=verbose),
                       np.float64)
    if verbose:
        print(f"[cluster] {task}: cells={len(cells)} cam={np.round(g['pos'], 3)} "
              f"-> cam_eff={np.round(cam, 4)} img={g['img']} fovy={g['fovy']:.1f}")
    return dict(cells=cells, M=M, cam=cam,
                crop_lo=np.asarray(cfg["min_bound"], np.float64),
                crop_hi=np.asarray(cfg["max_bound"], np.float64),
                z_place=float(z_place), gs=ADROIT_GS,
                offset=(np.zeros(3) if offset is None
                        else np.asarray(offset, np.float64)),
                cam_pos=np.asarray(g["pos"], np.float64),
                cam_R=np.asarray(g["R"], np.float64),
                cam_body_pos=np.asarray(g["body_pos"], np.float64),
                cam_f=float(g["f"]), cam_loc=np.asarray(g["loc"], np.float64),
                cam_img=float(g["img"]))


def calibrate(task="metaworld_assembly", zarr_path=None, out=None, xml_path=None,
              force=False, verbose=True, z_place=None):
    """Freeze {green cells, world->cloud frame, crop box, spacings} into an npz.

    The camera matrix needs mujoco (``mujoco_py.load_model_from_path``); the
    green cells come from the existing ``green_cells_assembly`` cache and the
    spacings from the task zarr.  Everything is written once and read back as
    plain numpy afterwards, so the eval process never re-derives geometry.
    """
    out = pathlib.Path(out) if out else _spec_default_path(task)
    if out.exists() and not force:
        if verbose:
            print(f"[cluster] spec exists (use --force to rebuild): {out}")
        return load_spec(task, spec_path=out, verbose=verbose)

    if _is_adroit(task):
        geo = _adroit_geometry(task, xml_path=xml_path,
                               z_place=(ADROIT_Z_PLACE if z_place is None
                                        else float(z_place)),
                               verbose=verbose)
        cells, M, cam = geo["cells"], geo["M"], geo["cam"]
        crop_lo, crop_hi, z_pl, gs = (geo["crop_lo"], geo["crop_hi"],
                                      geo["z_place"], geo["gs"])
        extra = dict(family=np.array(ADROIT_FAMILY), offset=geo["offset"],
                     cam_pos=geo["cam_pos"], cam_R=geo["cam_R"],
                     cam_body_pos=geo["cam_body_pos"], cam_f=geo["cam_f"],
                     cam_loc=geo["cam_loc"], cam_img=geo["cam_img"],
                     filter_visible=np.int8(1))
    else:
        ga = _bench()
        cells = np.asarray(ga.load_green_cells(verbose=False), np.float64)
        Rcam, cam = ga._corner2_cam(xml_path)
        M = Rcam @ ga._pc_transform().T             # cloud = (world - cam) @ M
        crop_lo, crop_hi = ga.CROP_MIN, ga.CROP_MAX
        z_pl, gs = ga._Z_PLACE, ga.PARAMS["gs"]
        extra = dict(family=np.array("metaworld"), offset=np.zeros(3),
                     filter_visible=np.int8(0))

    if zarr_path is None:
        cand = pathlib.Path("/hy-tmp/3DP/data") / f"{task}_expert.zarr"
        zarr_path = cand if cand.exists() else None
    if zarr_path is not None and pathlib.Path(zarr_path).exists():
        delta_fps, delta_raw = _measure_delta(zarr_path)
    else:
        delta_fps, delta_raw = DELTA_FPS_CM, DELTA_RAW_CM
        if verbose:
            print(f"[cluster] no zarr found -> delta fallback "
                  f"{DELTA_FPS_CM}/{DELTA_RAW_CM} cm")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        np.savez_compressed(
            f, cells=cells.astype(np.float32), M=M, cam=np.asarray(cam, np.float64),
            crop_lo=np.asarray(crop_lo, np.float64),
            crop_hi=np.asarray(crop_hi, np.float64),
            z_place=np.float64(z_pl), gs=np.float64(gs),
            delta_fps=np.float64(delta_fps), delta_raw=np.float64(delta_raw),
            task=np.array(str(task)), **extra)
    os.replace(str(tmp), str(out))
    if verbose:
        print(f"[cluster] calibrated {len(cells)} cells, delta_fps={delta_fps:.3f} cm, "
              f"delta_raw={delta_raw:.3f} cm -> {out}")
    return load_spec(task, spec_path=out, verbose=False)


def load_spec(task="metaworld_assembly", spec_path=None, verbose=False):
    """Frozen spec dict; auto-calibrates when the npz is missing."""
    sp = pathlib.Path(spec_path) if spec_path else _spec_default_path(task)
    if not sp.exists():
        return calibrate(task, out=sp, verbose=verbose)
    with np.load(sp) as z:
        spec = {k: z[k] for k in z.files}
    for k in ("task", "family"):
        if k in spec:
            v = spec[k]
            spec[k] = str(task) if getattr(v, "ndim", 1) != 0 else str(v)
    spec.setdefault("family", "metaworld")
    for k in ("M", "cam", "crop_lo", "crop_hi", "cells"):
        spec[k] = np.asarray(spec[k], np.float64)
    for k in ("offset", "cam_pos", "cam_R", "cam_body_pos", "cam_loc"):
        if k in spec:
            spec[k] = np.asarray(spec[k], np.float64)
    for k in ("z_place", "gs", "delta_fps", "delta_raw"):
        spec[k] = float(spec[k])
    for k in ("cam_f", "cam_img"):
        if k in spec:
            spec[k] = float(spec[k])
    spec["filter_visible"] = bool(spec.get("filter_visible", 0))
    spec["Minv"] = np.linalg.inv(spec["M"])
    spec["spec_path"] = str(sp)
    return spec


def _hemisphere(rng, r, n):
    """Equal-area upper hemisphere about the origin (z up), NN spacing ~ r*sqrt(2pi/n)."""
    n = int(n)
    i = np.arange(n, dtype=np.float64) + 0.5
    cos_t = 1.0 - i / n                                  # uniform in (0, 1] -> equal area
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t * cos_t))
    phi = i * GOLDEN_ANGLE + rng.uniform(0.0, 2.0 * math.pi)
    return r * np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=1)


class ClusterInjector:
    """Append ``clusters * points`` distractor rows before the original FPS.

    ``injector(cloud, fps_fn)`` -> ``fps_fn(concat([cloud, rows]))``; with
    ``points <= 0`` it is ``fps_fn(cloud)``, i.e. the historical path.  The rows
    of one episode are drawn in :meth:`begin` (deterministic in ``seed`` and the
    episode index) and reused for every observation of that episode.
    """

    def __init__(self, task="metaworld_assembly", seed=0, points=0, clusters=1,
                 radius_mode="fps", radius_cm=None, s_ref_cm=None,
                 rgb=OBSTACLE_RGB, min_gap_cm=3.0, cells=None, spec_path=None,
                 num_points=None, verbose=True):
        self.task = str(task)
        self.seed = int(seed or 0)
        self.points = int(points)
        self.clusters = max(1, int(clusters))
        self.radius_mode = str(radius_mode)
        self.rgb = np.asarray(rgb, np.float64)
        self.min_gap = float(min_gap_cm) / 100.0
        self.spec = None
        self.rows = None
        self.episode = None
        self.history = []
        self.s_ref_cm = 0.0
        self.radius = 0.0

        if self.points <= 0:
            return

        self.spec = load_spec(self.task, spec_path=spec_path, verbose=verbose)
        if cells is not None:                       # explicit cell override (M,2) world xy
            self.spec["cells"] = np.asarray(cells, np.float64).reshape(-1, 2)
        self.s_ref_cm = 100.0 * self._pick_s_ref(s_ref_cm)
        self.radius = (float(radius_cm) / 100.0 if radius_cm
                       else radius_for(self.points, self.s_ref_cm) / 100.0)   # cm -> m
        if num_points is not None and self.clusters * self.points > int(num_points):
            raise ValueError(
                f"cluster rows ({self.clusters}x{self.points}) exceed num_points="
                f"{num_points}: the cluster must stay a minority of the FPS budget")

        # keep only cells whose full AABB (x+-r, y+-r, z_place+-r) survives both
        # the crop box and the camera frustum: a clipped cluster would break the
        # dose<->area relation (adroit's crop box only bounds z, so the frustum
        # is what actually limits the reachable cells)
        self.cells_total = int(len(self.spec["cells"]))
        self.cells_visible = self.cells_total
        if self.spec.get("filter_visible") and self.radius > 0.0:
            ok = self._visible_mask(self.radius)
            self.cells_visible = int(ok.sum())
            self.spec["cells"] = self.spec["cells"][ok]
            if self.cells_visible <= 0:
                raise ValueError(
                    f"cluster_distractor: no green cell is fully visible at "
                    f"r={self.radius * 100:.2f} cm; lower the dose or the radius")

        if verbose:
            print(f"[cluster] {self.task}: N={self.points} clusters={self.clusters} "
                  f"r={self.radius * 100:.2f} cm ({self.radius_mode}, "
                  f"s_ref={self.s_ref_cm:.2f} cm) "
                  f"cells={self.cells_visible}/{self.cells_total}")

    # -- config ---------------------------------------------------------------
    def _pick_s_ref(self, s_ref_cm):
        """Spacing used by the area law, in metres."""
        if s_ref_cm is not None:
            return float(s_ref_cm) / 100.0
        if self.radius_mode == "raw":
            return float(self.spec["delta_raw"]) / 100.0
        return float(self.spec["delta_fps"]) / 100.0

    # -- frame helpers --------------------------------------------------------
    def _w2c(self, w):
        return (np.asarray(w, np.float64) - self.spec["cam"]) @ self.spec["M"]

    def _c2w(self, c):
        return np.asarray(c, np.float64) @ self.spec["Minv"] + self.spec["cam"]

    def _visible_mask(self, r):
        """Cells whose full AABB stays in the crop box *and* the camera frustum.

        Same criterion as ``cam_geom.aabb_visible`` (the judge the posgen suite
        uses to place cubes): the 8 corners of ``(x+-r, y+-r, z_place+-r)`` are
        checked against the wrapper's strict crop inequalities and against the
        projected image bounds, so a sampled cluster is never partially clipped.
        """
        cells = np.asarray(self.spec["cells"], np.float64)
        n = len(cells)
        s = np.array([[-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
                      [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]],
                     np.float64) * float(r)
        w = np.stack([cells[:, None, 0] + s[None, :, 0],
                      cells[:, None, 1] + s[None, :, 1],
                      np.full((n, 8), self.spec["z_place"]) + s[None, :, 2]],
                     axis=2)
        cloud = self._w2c(w.reshape(-1, 3)).reshape(n, 8, 3)
        crop_ok = (np.all(cloud > self.spec["crop_lo"], axis=2) &
                   np.all(cloud < self.spec["crop_hi"], axis=2))
        p = (w - self.spec["cam_pos"]) @ self.spec["cam_R"]     # z<0 = in front
        z = p[..., 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            inv = np.where(z < -1e-9, -1.0 / z, np.nan)
            u = self.spec["cam_loc"][0] + self.spec["cam_f"] * p[..., 0] * inv
            v = self.spec["cam_loc"][1] - self.spec["cam_f"] * p[..., 1] * inv
        m = 1.0
        hi_px = self.spec["cam_img"] - 1.0 - m
        frus_ok = (np.isfinite(u) & np.isfinite(v) & (z < 0.0) &
                   (u >= m) & (u <= hi_px) & (v >= m) & (v <= hi_px))
        return (crop_ok & frus_ok).all(axis=1)

    # -- per-episode plan -----------------------------------------------------
    def begin(self, episode_idx):
        """Draw the cluster geometry for ``episode_idx``."""
        self.rows = None
        self.episode = int(episode_idx)
        if self.points <= 0:
            return None
        rng = np.random.RandomState(self.seed + 10429 + self.episode)
        rows, info = [], []
        for c in self._pick_centers(rng):
            r_rows, meta = self._cluster_rows(rng, c, self.radius)
            rows.append(r_rows)
            info.append(meta)
        self.rows = np.concatenate(rows, axis=0) if rows else None
        self.history.append(dict(episode=self.episode,
                                 centers=[m["center"] for m in info],
                                 center_shift_cm=[m["shift_cm"] for m in info],
                                 center_dz_cm=[m["dz_cm"] for m in info],
                                 radius_cm=100.0 * self.radius))
        return self.rows

    def _pick_centers(self, rng):
        """``clusters`` distinct green cells, separated by ``2r + min_gap`` if possible."""
        cells = self.spec["cells"]
        n = min(self.clusters, len(cells))
        if n == len(cells):
            return cells[rng.permutation(len(cells))]
        order = rng.permutation(len(cells))
        gap = 2.0 * self.radius + self.min_gap
        chosen = []
        for i in order:
            if all(np.hypot(*(cells[i] - cells[j])) >= gap for j in chosen):
                chosen.append(int(i))
                if len(chosen) == n:
                    break
        if len(chosen) < n:                          # dense case: relax to distinct cells
            chosen += [int(i) for i in order if int(i) not in chosen][:n - len(chosen)]
        return cells[chosen]

    def _cluster_rows(self, rng, center_xy, r):
        """One hemisphere -> (N, 6) cloud-frame rows + geometry metadata."""
        c_world = np.array([center_xy[0], center_xy[1], self.spec["z_place"]], np.float64)
        c_cloud = self._w2c(c_world)
        # keep every injected row inside the observation box; r stays fixed, so the
        # dose does not drift across episodes (measured shift ~0.2 cm at r=3.5 cm)
        c_clamped = np.clip(c_cloud, self.spec["crop_lo"] + r, self.spec["crop_hi"] - r)
        shift = float(np.linalg.norm(c_clamped - c_cloud))
        c_new = self._c2w(c_clamped)
        pts = self._w2c(c_new[None, :] + _hemisphere(rng, r, self.points))
        rows = np.concatenate([pts, np.broadcast_to(self.rgb, (len(pts), 3))], axis=1)
        return rows, dict(center=[float(center_xy[0]), float(center_xy[1])],
                          shift_cm=100.0 * shift,
                          dz_cm=100.0 * float(c_new[2] - c_world[2]))

    # -- the hook -------------------------------------------------------------
    def __call__(self, point_cloud, fps_fn=None):
        """``fps_fn(concat([point_cloud, rows]))``; passthrough when disabled."""
        rows = self.rows
        if rows is None or len(rows) == 0:
            return point_cloud if fps_fn is None else fps_fn(point_cloud)
        rows = np.asarray(rows, dtype=point_cloud.dtype)
        if point_cloud.shape[1] > rows.shape[1]:      # xyz-only cloud
            rows = np.concatenate(
                [rows, np.zeros((len(rows), point_cloud.shape[1] - rows.shape[1]),
                                dtype=point_cloud.dtype)], axis=1)
        merged = np.concatenate([point_cloud, rows[:, :point_cloud.shape[1]]], axis=0)
        return merged if fps_fn is None else fps_fn(merged)

    def __repr__(self):
        return (f"ClusterInjector(task={self.task!r}, points={self.points}, "
                f"clusters={self.clusters}, r_cm={self.radius * 100:.2f})")

    # -- reporting ------------------------------------------------------------
    def manifest(self):
        """JSON-able provenance (written next to dr3d_suite_manifest.json)."""
        if self.points <= 0:
            return {}
        return dict(task=self.task, points=int(self.points),
                    clusters=int(self.clusters),
                    radius_cm=100.0 * self.radius,
                    radius_mode=self.radius_mode,
                    s_ref_cm=self.s_ref_cm,
                    family=str(self.spec.get("family", "metaworld")),
                    cells_total=int(getattr(self, "cells_total", 0)),
                    cells_visible=int(getattr(self, "cells_visible", 0)),
                    delta_fps_cm=float(self.spec["delta_fps"]),
                    delta_raw_cm=float(self.spec["delta_raw"]),
                    spec_path=self.spec.get("spec_path"),
                    seed=int(self.seed), episodes=self.history)


# -- CLI ----------------------------------------------------------------------
def _check(injs, sample=60, seed=0):
    """R1-R4: spacing, in-crop containment, radius table, determinism."""
    from scipy.spatial import cKDTree

    failures = []
    for inj in injs:
        if inj is None:
            continue
        r = inj.radius
        rows = inj.begin(0)
        assert rows is not None and len(rows) == inj.clusters * inj.points
        lo, hi = inj.spec["crop_lo"], inj.spec["crop_hi"]
        bad = np.any((rows[:, :3] <= lo) | (rows[:, :3] >= hi), axis=1)
        if bad.any():
            failures.append(f"R2 N={inj.points}: {bad.sum()} rows outside the crop box")

        # R1: in-cluster spacing must match the nominal s_ref (spiral is equal-area)
        nn = []
        for cell in inj.spec["cells"][::max(1, len(inj.spec["cells"]) // sample)]:
            pts, _ = inj._cluster_rows(np.random.RandomState(seed), cell, r)
            d = cKDTree(pts[:, :3]).query(pts[:, :3], k=2)[0][:, 1]
            nn.append(np.median(d))
        ratio = float(np.median(nn)) * 100.0 / inj.s_ref_cm
        frac = (inj.cells_visible / inj.cells_total) if inj.cells_total else 1.0
        print(f"[check] N={inj.points:3d} r={r * 100:5.2f} cm  "
              f"NN={np.median(nn) * 100:5.2f} cm (x{ratio:.2f} of s_ref)  "
              f"rows={len(rows)}  shift<={max(inj.history[-1]['center_shift_cm']):.2f} cm"
              f"  cells={inj.cells_visible}/{inj.cells_total}")
        if not 0.8 <= ratio <= 1.25:
            failures.append(f"R1 N={inj.points}: NN/s_ref={ratio:.2f} outside [0.8, 1.25]")

        # R5: the sampled cells must be fully inside crop+frustum (adroit only;
        # metaworld cells are in-crop by construction in green_cells_assembly)
        if inj.spec.get("filter_visible"):
            if frac < VISIBLE_FLOOR:
                failures.append(f"R5 N={inj.points}: only {frac * 100:.0f}% of the "
                                f"cells are fully visible (< {VISIBLE_FLOOR * 100:.0f}%)")
            exp = EXPECT_VISIBLE.get(inj.task, {}).get(inj.points)
            if exp is not None and abs(frac - exp) > 0.05:
                failures.append(f"R5 N={inj.points}: visible fraction {frac:.2f} "
                                f"drifted from the measured {exp:.2f}")

    # R4: determinism (same seed+episode -> identical rows; different episode -> not)
    ref = injs[len(injs) // 2]
    a = ClusterInjector(task=ref.task, seed=7, points=ref.points,
                        spec_path=ref.spec["spec_path"], verbose=False)
    b = ClusterInjector(task=ref.task, seed=7, points=ref.points,
                        spec_path=ref.spec["spec_path"], verbose=False)
    if not np.array_equal(a.begin(3), b.begin(3)):
        failures.append("R4: same (seed, episode) produced different rows")
    if np.array_equal(a.begin(3), a.begin(4)):
        failures.append("R4: different episodes produced identical rows")

    # disabled injector must be a pure passthrough
    off = ClusterInjector(task=ref.task, seed=7, points=0, verbose=False)
    cloud = np.random.RandomState(0).rand(1500, 6).astype(np.float32)
    if not np.allclose(off(cloud, None), cloud):
        failures.append("R-off: points=0 changed the cloud")
    return failures


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Pre-FPS distractor cluster injection.")
    ap.add_argument("--task", default="metaworld_assembly")
    ap.add_argument("--points", type=int, default=0, help="candidate rows per cluster")
    ap.add_argument("--clusters", type=int, default=1)
    ap.add_argument("--radius-mode", default="fps", choices=["fps", "raw"])
    ap.add_argument("--radius-cm", type=float, default=None)
    ap.add_argument("--spec", default=None, help="override the frozen npz path")
    ap.add_argument("--zarr", default=None)
    ap.add_argument("--xml", default=None)
    ap.add_argument("--calibrate", action="store_true", help="(re)build the frozen npz")
    ap.add_argument("--check", action="store_true", help="offline self-check (R1-R5)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--z-place", type=float, default=None,
                    help="adroit only: cluster centre height in world z (default 0.0)")
    ap.add_argument("--num-points", type=int, default=None,
                    help="FPS budget of the task (512 adroit, 1024 metaworld)")
    a = ap.parse_args(argv)

    if a.calibrate:
        calibrate(a.task, zarr_path=a.zarr, out=a.spec, xml_path=a.xml,
                  force=a.force, verbose=True, z_place=a.z_place)

    if a.points > 0:
        inj = ClusterInjector(a.task, seed=0, points=a.points, clusters=a.clusters,
                              radius_mode=a.radius_mode, radius_cm=a.radius_cm,
                              spec_path=a.spec, num_points=a.num_points, verbose=True)
        inj.begin(0)
        print(f"[cluster] rows={inj.rows.shape} r={inj.radius * 100:.2f} cm "
              f"history={inj.history[-1]}")

    if a.check:
        spec = load_spec(a.task, spec_path=a.spec)
        print(f"[check] spec={spec['spec_path']} family={spec['family']} "
              f"cells={len(spec['cells'])} delta_fps={spec['delta_fps']:.3f} cm "
              f"delta_raw={spec['delta_raw']:.3f} cm")
        print("[check] r(N) = s_ref*sqrt(N/2pi):",
              " ".join(f"N={n}->{radius_for(n, spec['delta_fps']):.2f}cm"
                       for n in sorted({0, 20, 50, 100, 200,
                                        *( [a.points] if a.points > 0 else [] )})))
        ns = sorted({20, 50, 100, 200, *([a.points] if a.points > 0 else [])})
        injs = [ClusterInjector(a.task, seed=0, points=n, spec_path=a.spec,
                                num_points=a.num_points, verbose=False)
                for n in ns]
        failures = _check(injs)
        if failures:
            print("[check] FAILED\n  " + "\n  ".join(failures))
            return 1
        print("[check] OK (R1 spacing, R2 in-crop, R3 radius law, R4 determinism, "
              "R5 visibility)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
