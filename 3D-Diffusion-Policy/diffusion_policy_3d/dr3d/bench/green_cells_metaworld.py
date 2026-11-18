#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""dr3d bench: green-cell candidates for metaworld (posgen places a cube) -- data-driven.

Same idea as adroit's `green_cells.py::compute_green_cells` (desktop cells minus the data
footprint), but metaworld's observation chain differs, so there is one extra criterion, in-crop:

    green = desktop cells (gs/range constants)
          ∩ in-crop                  constants + XML camera matrix (corner2): keeps the cube in the policy cloud
          ∩ ¬ dilate(footprint)      data: points back-projected above the table from the zarr depth = hand/stick/thermos
          ∩ visible@reset            data: not occluded, inside the frame, in every episode's first frame

footprint / visible both come ENTIRELY from `<root>/metaworld_<task>_expert.zarr`'s
`data/depth` (128x128, the same corner2 camera/resolution as the policy); no hand-tuned workspace constants:

  * footprint : back-project each frame's depth to world, keep z > arm_height -> 0.03 desktop
                cells -> union over the dataset (same meaning as adroit's dataset_arm_footprint,
                only the camera is oblique); visible@reset : in each episode's first frame
                (= reset pose; metaworld's reset is deterministic, measured 10/10 identical),
                at least vis_need of the 5 sampled cube top-face points are inside the frame
                and have depth >= expected axial depth - vis_tol

Back-projection convention (verified on real depth frames: desktop pixels |z|<2cm give std=2.7mm):
    p_cam = ((col-cx)*d/f, (row-cy)*d/f, -d)          # d = metric axial depth after depthimg2Meters
    w     = p_cam @ cam_mat0.T + cam_pos_override
(note the v sign is flipped w.r.t. `cam_geom.project`: corner2's cam_mat0 already is the
 camera->world rotation that cam_geom's to_cloud/project lack -- it does not apply here.)

Cache: dr3d/tmp/green_cells_mw_<task>.npz + .json
       fingerprint = zarr path + frame count + base XML + all parameters; an existing cache is
       trusted once the dataset has moved away (same as green_cells.py::_meta_compatible);
       missing data with no cache -> actionable message, then exit.
Dynsphere/dynhand volumes (TASKS['dyn']=='data'; cache green_cells_mw_<task>_dyn.npz):
       3D occupancy + task-object screen footprint + per-(cell,height,radius) visibility, from
       the per-task object spec (_obj_screen_discs / _obj_coms): stick-push = one disc per stick
       end + the thermos; disassemble = RoundNut ring + wrench handle + the static peg.
Non-invasive: reads only the zarr and the third_party model XML; all output goes to dr3d/tmp.

CLI:
    python dr3d/bench/green_cells_metaworld.py --task stick-push [--force] [--show N]
    python dr3d/bench/green_cells_metaworld.py --task assembly --cells data   # data-driven rule
"""
import hashlib
import json
import os
import pathlib
import sys

import numpy as np

DR3D = pathlib.Path(__file__).resolve().parents[1]
_BENCH = pathlib.Path(__file__).resolve().parent
for _p in (str(DR3D), str(_BENCH)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dataroot as _dr          # noqa: E402
import test_load as _TL         # noqa: E402
import green_cells_assembly as _GA   # noqa: E402  single source of truth for obs-chain constants

CACHE_DIR = DR3D / "tmp"

# --- observation-chain constants (identical to metaworld_wrapper.MetaWorldEnv, from green_cells_assembly)
IMG = 128                                     # PointCloudGenerator(img_size=128)
CAM_NAME = "corner2"
CAM_POS = _GA.CAM_POS_OVERRIDE                # corner2 position after the wrapper override
CROP_MIN, CROP_MAX = _GA.CROP_MIN, _GA.CROP_MAX
# --- green-cell parameters (data-driven; names/semantics follow adroit's green_cells.PARAMS)
PARAMS = dict(
    gs=0.03,                                  # desktop cell edge (m); metaworld keeps 3cm
    x=tuple(_GA.PARAMS["x"]), y=tuple(_GA.PARAMS["y"]),   # desktop range (constants)
    z_place=0.03,                             # cube center height on a candidate cell (in-crop test)
    arm_height=0.05,                          # "above the table" threshold (m) -- footprint
    margin_cells=1,                           # footprint dilation in cells (1 cell = 3cm)
    vis_samples=5, vis_need=2,                # visibility: at least 2 of 5 sampled points visible
    vis_tol=0.03,                             # depth comparison tolerance (m)
    d_min=0.05, d_max=2.5,                    # valid depth range (m) -- drops near/far
    frame_step=2,                             # footprint frame stride (half of the 2000 frames)
)
_SAMPLE_OFFSETS = ((0.0, 0.0), (0.021, 0.021), (-0.021, 0.021),
                   (0.021, -0.021), (-0.021, -0.021))

# --- dynsphere-only parameters (data-driven: 3D occupancy + task-object screen footprint)
DYN = dict(
    voxel=0.03,                    # voxel edge (m) -- same scale as the desktop cell
    z_min=0.02,                    # only points >2cm above the table count as occupied
    dilate=1,                      # occupancy dilation (cells) -- slack for the sphere radius
    frame_step=4,                  # frame stride (500 of the 2000 frames)
    vox_lo=(-1.0, 0.0, -0.05),     # voxel grid origin (world, m): desktop +-r plus full height
    vox_shape=(60, 44, 34),        # -> x∈[-1.0,0.8], y∈[0.0,1.32], z∈[-0.05,0.97]
    sky_rgb=(127, 126, 122),       # basic_scene's skybox is a flat gradient -> drop it by RGB
    sky_tol=3,                     #   otherwise "sky" becomes fake geometry at z=0.8 (depth 1.8~2.1m)
    obj_radius=0.055,              # conservative envelope of the thermos radius (~0.05m)
    stick_half_len=0.11,           # stick half length (com to end, ~0.11m)
    stick_radius=0.045,            # stick radius (~0.02m) + margin -> one disc per x end (capsule approx)
    # disassemble (sawyer_assembly_peg): RoundNut ring + wrench handle + static peg
    nut_ring_radius=0.070,         # ring mesh outer radius 0.06615 + margin
    nut_handle_radius=0.045,       # WrenchHandle box half-diagonal 0.0425 + margin
    nut_site_offset=0.13,          # |site 'RoundNut-8' - ring center|; full_state[4:7] IS that site
    peg_radius=0.020,              # peg cylinder radius (sawyer_assembly_peg.xml)
    peg_dz=(0.02, 0.05, 0.08, 0.10),   # peg heights above the table (cylinder z in [0.005, 0.105])
    screen_margin_px=4.0,          # extra margin on the sphere's screen disc (px)
    h_step=0.02, h_max=0.80,       # height grid (same as dynamic_sphere_mw.HS)
    min_vis_frac=0.5,              # min fraction of data frames where the center pixel is unoccluded
    work_margin=0.03,              # dynhand: extra workspace (data footprint) margin (m), safe = dist >= hr + 0.03
)

# --- task table ----------------------------------------------------------------------
#   cells='data'    data-driven green cells (this module)   cells='legacy' keeps green_cells_assembly
#   dyn  ='data'    dynsphere/dynhand also data-driven (3D occupancy + screen footprint + vis_frac);
#                   'legacy' = old table_grid. The flag is shared: both metaworld distractors
#                   consume the same volumes (dyn_volumes), so 'data' enables them together.
# base is a key of test_load.BASE_XML (the same task's MJCF, matching compose.mw_scene).
TASKS = {
    "assembly":   dict(base="assembly",   cells="legacy", dyn="legacy"),
    "stick-push": dict(base="stick_push", cells="data",   dyn="data"),
    # shares assembly's MJCF (camera/crop identical), so cells come from its own zarr depth;
    # its task object (RoundNut ring + wrench handle + static peg) has its own spec below.
    "disassemble": dict(base="assembly",  cells="data",   dyn="data"),
}


def _rec(task):
    """task -> table row; an unregistered task raises (same style as compose.mw_scene, no fallback)."""
    if task not in TASKS:
        raise ValueError("green_cells_metaworld: unsupported task %r (supported: %s)"
                         % (task, sorted(TASKS)))
    return TASKS[task]


def base_xml(task):
    """task -> that task's main MJCF (single source of truth: test_load.BASE_XML, same key as mw_scene)."""
    return _TL.BASE_XML[_rec(task)["base"]]


def _cam(xml_path):
    """Read-only XML -> (Rcam, f, c): Rcam=cam_mat0 (camera->world), f=focal (px), c=principal point."""
    from mujoco_py import load_model_from_path
    m = load_model_from_path(str(xml_path))
    cid = int(m.camera_name2id(CAM_NAME))
    if cid < 0:
        raise ValueError("green_cells_metaworld: camera %r not found in %s"
                         % (CAM_NAME, xml_path))
    Rcam = np.asarray(m.cam_mat0, dtype=float).reshape(-1, 3, 3)[cid].copy()
    fovy = float(np.ravel(m.cam_fovy)[cid])
    f = IMG / (2.0 * np.tan(np.radians(fovy) / 2.0))
    return Rcam, float(f), (IMG - 1) / 2.0


def _grid_len():
    gs = float(PARAMS["gs"])
    nx = len(np.arange(PARAMS["x"][0], PARAMS["x"][1] + 1e-9, gs))
    ny = len(np.arange(PARAMS["y"][0], PARAMS["y"][1] + 1e-9, gs))
    return ny, nx


def desk_grid(gs=None):
    """Desktop (x,y) grid (constant range/step); fixed order => cache/sampling reproducible."""
    gs = float(gs or PARAMS["gs"])
    xs = np.arange(PARAMS["x"][0], PARAMS["x"][1] + 1e-9, gs)
    ys = np.arange(PARAMS["y"][0], PARAMS["y"][1] + 1e-9, gs)
    XX, YY = np.meshgrid(xs, ys)
    return np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)


def table_grid(gs=None):
    """Whole-table grid (unfiltered); for callers like dynsphere ("any desktop point + height test")."""
    return desk_grid(gs)


def crop_mask(cells, Rcam, f=None, c=None):
    """in-crop: exactly the same convention as MetaWorldEnv.get_point_cloud's crop (constants)."""
    pcT = _GA._pc_transform()
    w = np.concatenate([np.asarray(cells, float),
                        np.full((len(cells), 1), PARAMS["z_place"])], axis=1)
    cloud = ((w - CAM_POS) @ Rcam) @ pcT.T
    return (np.all(cloud >= CROP_MIN, axis=1) & np.all(cloud <= CROP_MAX, axis=1))


def _pixel_of(cells, Rcam, f, c, z):
    """World (x,y,z) -> (row, col, z_axial); metric axial depth = -p_cam.z."""
    w = np.concatenate([np.asarray(cells, float), np.full((len(cells), 1), float(z))],
                       axis=1)
    pc = (w - CAM_POS) @ Rcam
    zs = -pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        col = c + f * pc[:, 0] / zs
        row = c + f * pc[:, 1] / zs
    return row, col, zs


def footprint_mask(zarr_path, Rcam, f, c):
    """Data footprint: xy of every dataset point "above the table (arm_height)" -> 0.03 cells

    Returns (blocked, footprint):
      footprint = the union itself (cell count used for diagnostics);
      blocked   = the union dilated by margin_cells cells (= what the green cells subtract).
    Equivalent to adroit's dataset_arm_footprint / world_arm_footprint (hand/object
    footprint), except corner2 is oblique, so cam_mat0 is needed for the back-projection.
    """
    import zarr
    ny, nx = _grid_len()
    gs = float(PARAMS["gs"])
    x0, y0 = float(PARAMS["x"][0]), float(PARAMS["y"][0])
    D = zarr.open(str(zarr_path), mode="r")["data"]["depth"]
    T = int(D.shape[0])
    H, W = int(D.shape[1]), int(D.shape[2])
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    grid = np.zeros(ny * nx, dtype=bool)
    step = max(1, int(PARAMS["frame_step"]))
    for i in range(0, T, step):
        dep = np.asarray(D[i], dtype=np.float64)
        ok = (dep > PARAMS["d_min"]) & (dep < PARAMS["d_max"])
        if not ok.any():
            continue
        d = dep[ok]
        p = np.stack([(cc[ok] - c) * d / f, (rr[ok] - c) * d / f, -d], axis=1)
        p = p @ Rcam.T + CAM_POS
        p = p[p[:, 2] > PARAMS["arm_height"]]
        if len(p) == 0:
            continue
        ix = np.floor((p[:, 0] - x0) / gs + 0.5).astype(int)
        iy = np.floor((p[:, 1] - y0) / gs + 0.5).astype(int)
        k = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        grid[iy[k] * nx + ix[k]] = True
    foot = grid.reshape(ny, nx)
    blocked = foot
    if int(PARAMS["margin_cells"]):
        from scipy import ndimage
        blocked = ndimage.binary_dilation(foot, iterations=int(PARAMS["margin_cells"]))
    return blocked.ravel(), foot.ravel()


def visible_at_reset(cells, zarr_path, Rcam, f, c):
    """First-frame visibility per episode: at least vis_need of the 5 cube top-face
    samples are inside the frame and unoccluded.

    Returns (mask_all_episodes, per_episode (n_ep, M)).
    metaworld's reset pose is deterministic: measured identical over 10/10 episodes, but
    every episode's first frame is still tested so a future randomized reset cannot fail silently.
    """
    import zarr
    Z = zarr.open(str(zarr_path), mode="r")
    D = Z["data"]["depth"]
    T = int(D.shape[0])
    ends = np.asarray(Z["meta"]["episode_ends"])
    starts = [0] + [int(e) for e in ends[:-1]]
    starts = [s for s in starts if 0 <= s < T]
    cells = np.asarray(cells, dtype=float)
    samp = []
    for dx, dy in _SAMPLE_OFFSETS[:int(PARAMS["vis_samples"])]:
        row, col, zexp = _pixel_of(cells + np.array([dx, dy]), Rcam, f, c, 0.05)
        samp.append((np.round(row).astype(int), np.round(col).astype(int), zexp))
    per = []
    for s in starts:
        dep = np.asarray(D[int(s)], dtype=np.float64)
        ok = (dep > PARAMS["d_min"]) & (dep < PARAMS["d_max"])
        cnt = np.zeros(len(cells), dtype=int)
        for rin, cin, zexp in samp:
            ii = (rin >= 1) & (rin <= IMG - 2) & (cin >= 1) & (cin <= IMG - 2)
            v = np.zeros(len(cells), dtype=bool)
            v[ii] = (ok[rin[ii], cin[ii]] &
                     (dep[rin[ii], cin[ii]] >= zexp[ii] - PARAMS["vis_tol"]))
            cnt += v
        per.append(cnt >= int(PARAMS["vis_need"]))
    per = np.stack(per) if per else np.zeros((0, len(cells)), dtype=bool)
    return per.all(axis=0), per


def compute_cells(task):
    """Build the green cells for a task (data-driven or legacy); returns dict(cells, source, diag)."""
    rec = _rec(task)
    if rec["cells"] == "legacy":
        # assembly's existing rule (analytic: in-crop ∧ ¬_ARM ∧ ¬_OBJ), keeps the baseline comparable
        cells = np.asarray(_GA.load_green_cells(verbose=False), dtype=np.float32)
        return dict(cells=cells, source="legacy",
                    diag=dict(n_grid=int(len(cells)), n_in_crop=int(len(cells)),
                              n_footprint=0, n_blocked=0, n_visible=int(len(cells)),
                              n_final=int(len(cells)), n_episodes=0,
                              in_crop=None, blocked=None, visible=None))
    xml = base_xml(task)
    zarr_path = _dr.mw_expert_zarr(task)
    if not pathlib.Path(zarr_path).exists():
        raise SystemExit(_missing_msg(task, zarr_path))
    Rcam, f, c = _cam(xml)
    grid = desk_grid()
    in_crop = crop_mask(grid, Rcam, f, c)
    blocked, foot = footprint_mask(zarr_path, Rcam, f, c)
    visible, per = visible_at_reset(grid, zarr_path, Rcam, f, c)
    keep = in_crop & ~blocked & visible
    return dict(cells=grid[keep], source=f"zarr:{task}",
                diag=dict(n_grid=int(len(grid)), n_in_crop=int(in_crop.sum()),
                          n_footprint=int(foot.sum()), n_blocked=int(blocked.sum()),
                          n_visible=int(visible.sum()), n_final=int(keep.sum()),
                          n_episodes=int(per.shape[0]),
                          in_crop=in_crop, blocked=blocked, visible=visible))


def cells_for(task, kind="posgen"):
    """Candidate cells for one use case (data-driven; legacy tasks use the old rule).

      kind='posgen'   : in-crop ∧ ¬footprint ∧ visible@reset (test for the 6cm cube@z=0.03)
      kind='dynsphere': in-crop ∧ ¬footprint (the sphere may float, so no cube reset visibility)
      kind='dynhand'  : same as dynsphere (the hand has its own per-vertex crop/frustum test)
    """
    rec = _rec(task)
    if rec["cells"] == "legacy":              # assembly: keep the existing rule
        if kind in ("dynsphere", "dynhand"):
            return np.asarray(_GA.table_grid(), dtype=np.float32)
        return np.asarray(_GA.load_green_cells(verbose=False), dtype=np.float32)
    load_cells(task, verbose=False)            # make sure the npz exists (all three masks are in it)
    d = np.load(cache_paths(task)[0])
    keep = d["in_crop"] & ~d["blocked"]
    if kind == "posgen":
        keep = keep & d["visible"]
    return desk_grid()[keep]


# --- per-task task-object spec (screen footprint + task_center) -------------------------
#   metaworld's full_state layout (39 cols): [eef(3), gripper(1), obj_site(3), obj_quat(wxyz,4),
#   pad(3), ...]. obj_site/obj_quat are the task's _get_pos_objects/_get_quat_objects.
_SPEC_KEYS = {"stick-push": ("stick_half_len", "stick_radius", "obj_radius"),
              "disassemble": ("nut_ring_radius", "nut_handle_radius", "nut_site_offset",
                              "peg_radius", "peg_dz")}


def _ring_nut_center(fs):
    """disassemble full_state -> RoundNut ring center (T,3).

    full_state[4:7] is the site 'RoundNut-8' (handle tip, local (0,-0.13,0)), not the body
    origin; full_state[7:11] is its quaternion in mujoco order (w,x,y,z). Ring center (= body
    origin) = site + R(q) @ (0, +0.13, 0); matches the full_state[36:39] goal on all 2000 frames.
    """
    from scipy.spatial.transform import Rotation
    fs = np.asarray(fs, dtype=np.float64)
    q = fs[:, 7:11]                                        # mujoco: w x y z
    rot = Rotation.from_quat(np.concatenate([q[:, 1:4], q[:, :1]], axis=1))   # scipy: x y z w
    off = np.broadcast_to(np.array([0.0, float(DYN["nut_site_offset"]), 0.0]), (len(q), 3))
    return fs[:, 4:7] + rot.apply(off)


def _obj_coms(task, fs):
    """Task-object com traces -> list of (T,3); used for dynhand's task_center.

    stick-push keeps the historical order (stick, thermos) so the value is bit-identical.
    """
    fs = np.asarray(fs, dtype=np.float64)
    if task == "stick-push":
        return [fs[:, 4:7], fs[:, 11:14]]
    if task == "disassemble":
        return [_ring_nut_center(fs)]
    raise ValueError("green_cells_metaworld: no dyn object spec for task %r" % task)


def _obj_screen_discs(task, fs, ep_starts=None):
    """Task-object screen footprint -> [(points (K,3), radius m)]; one disc per entry.

    stick-push : stick = one disc per x end (capsule approx) + one thermos disc;
    disassemble: RoundNut ring + wrench handle + the static peg (one fixed disc per episode:
                 peg body_pos = that episode's reset obj_init_pos + [0,0,0.03]).
    """
    fs = np.asarray(fs, dtype=np.float64)
    if task == "stick-push":
        stick = fs[:, 4:7]
        off = np.array([float(DYN["stick_half_len"]), 0.0, 0.0])
        return [(stick + off, float(DYN["stick_radius"])),
                (stick - off, float(DYN["stick_radius"])),
                (fs[:, 11:14], float(DYN["obj_radius"]))]
    if task == "disassemble":
        ring = _ring_nut_center(fs)
        out = [(ring, float(DYN["nut_ring_radius"])),
               (fs[:, 4:7], float(DYN["nut_handle_radius"]))]
        dz = np.asarray(DYN["peg_dz"], dtype=float)
        r_peg = float(DYN["peg_radius"])
        if r_peg > 0.0 and len(dz) and ep_starts is not None and len(ep_starts):
            base = ring[np.asarray(ep_starts, dtype=int), :2]
            peg = np.stack([np.repeat(base[:, 0], len(dz)),
                            np.repeat(base[:, 1], len(dz)),
                            np.tile(dz, len(base))], axis=1)
            out.append((peg, r_peg))
        return out
    raise ValueError("green_cells_metaworld: no dyn object spec for task %r" % task)


def _obj_spec_sig(task):
    """Spec constants that feed the volumes -> part of the cache key (a change invalidates it)."""
    if task not in _SPEC_KEYS:
        raise ValueError("green_cells_metaworld: no dyn object spec for task %r" % task)
    return {k: DYN[k] for k in _SPEC_KEYS[task]}


# --- dynsphere data volumes (3D occupancy + task-object screen footprint) ----------------
def dyn_cache_paths(task):
    return (CACHE_DIR / f"green_cells_mw_{task}_dyn.npz",
            CACHE_DIR / f"green_cells_mw_{task}_dyn.json")


def _dyn_key(task):
    t = _zarr_frames(_dr.mw_expert_zarr(task))
    blob = json.dumps({"v": 1, "task": task, "zarr": _dr.mw_expert_zarr(task),
                       "T": -1 if t is None else t, "xml": str(base_xml(task)),
                       "dyn": DYN, "spec": _obj_spec_sig(task),
                       "img": IMG, "cam": CAM_POS.tolist(),
                       "crop": [CROP_MIN.tolist(), CROP_MAX.tolist()],
                       "d_min": PARAMS["d_min"], "d_max": PARAMS["d_max"]},
                      sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _compute_dyn(task):
    """zarr depth -> (dict(occ, lo, gs, vis_frac, heights, cells), screen_mask, diag).

    All three are data (no rendering):
      occ      : 3D occupancy (arm + task object, 0.03 voxel union, dilated by 1)
      screen   : task-object screen footprint (per-task spec: _obj_screen_discs)
      vis_frac : (n_cells, n_heights, n_radii) = for every (cell, height, radius), the fraction
                 of data frames where the sphere's center pixel is unoccluded (PARAMS['vis_tol'])
    """
    import zarr
    zarr_path = _dr.mw_expert_zarr(task)
    if not pathlib.Path(zarr_path).exists():
        raise SystemExit(_missing_msg(task, zarr_path))
    Rcam, f, c = _cam(base_xml(task))
    Z = zarr.open(str(zarr_path), mode="r")
    D = Z["data"]["depth"]
    T = int(D.shape[0])
    H, W = int(D.shape[1]), int(D.shape[2])
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    gs = float(DYN["voxel"])
    lo = np.asarray(DYN["vox_lo"], dtype=float)
    nx, ny, nz = (int(v) for v in DYN["vox_shape"])
    occ = np.zeros((nx, ny, nz), dtype=bool)
    sky_lo = np.asarray(DYN["sky_rgb"], dtype=np.float64) - float(DYN["sky_tol"])
    sky_hi = np.asarray(DYN["sky_rgb"], dtype=np.float64) + float(DYN["sky_tol"])

    # visibility-rate grid: sphere center pixel + expected axial depth per (cell x height)
    from dynamic import R_SET as _R_SET      # single source of truth for the radius set
    r_vals = tuple(float(v) for v in _R_SET)
    cells_dyn = np.asarray(cells_for(task, "dynsphere"), dtype=float)
    heights = np.arange(float(DYN["h_step"]), float(DYN["h_max"]) + 1e-9,
                        float(DYN["h_step"]))
    ci, hi = np.meshgrid(np.arange(len(cells_dyn)), np.arange(len(heights)),
                         indexing="ij")
    gx = cells_dyn[ci.ravel(), 0]
    gy = cells_dyn[ci.ravel(), 1]
    gz = heights[hi.ravel()]
    gpc = (np.stack([gx, gy, gz], axis=1) - CAM_POS) @ Rcam
    gzs = -gpc[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        gcol = c + f * gpc[:, 0] / gzs
        grow = c + f * gpc[:, 1] / gzs
    grin = np.round(grow).astype(int)
    gcin = np.round(gcol).astype(int)
    gin = ((grin >= 1) & (grin <= IMG - 2) & (gcin >= 1) & (gcin <= IMG - 2) &
           (gzs > 0))
    n_vis = np.zeros((len(gx), len(r_vals)), dtype=np.int32)
    vis_reset = np.zeros((len(gx), len(r_vals)), dtype=bool)
    # for dynhand: 2D footprint (points above arm_height) -> distance field (workspace margin)
    gny, gnx = _grid_len()
    gg = float(PARAMS["gs"])
    gx0, gy0 = float(PARAMS["x"][0]), float(PARAMS["y"][0])
    foot2d = np.zeros(gny * gnx, dtype=bool)

    n_clip = 0
    n_sky = 0
    n_frames = 0
    step = max(1, int(DYN["frame_step"]))
    for i in range(0, T, step):
        dep = np.asarray(D[i], dtype=np.float64)
        ok = (dep > PARAMS["d_min"]) & (dep < PARAMS["d_max"])
        if ok.any():
            # drop the skybox (flat gradient): otherwise "sky" becomes fake geometry at z=0.8
            rgb = np.asarray(Z["data"]["img"][i])
            sky = np.all((rgb >= sky_lo) & (rgb <= sky_hi), axis=-1)
            n_sky += int((ok & sky).sum())
            ok &= ~sky
            # visibility: measured depth at the center pixel >= (center axial depth - r) - tol
            # NB r must be subtracted: the pixel sees the sphere's FRONT surface, ~r closer.
            dm = dep[grin[gin], gcin[gin]]
            okp = ok[grin[gin], gcin[gin]]
            for k, rk in enumerate(r_vals):
                v = np.zeros(len(gx), dtype=bool)
                v[gin] = okp & (dm >= gzs[gin] - float(rk) - float(PARAMS["vis_tol"]))
                n_vis[:, k] += v
                if n_frames == 0:            # first frame (= reset pose) visibility
                    vis_reset[:, k] = v
            n_frames += 1
        if not ok.any():
            continue
        d = dep[ok]
        p = np.stack([(cc[ok] - c) * d / f, (rr[ok] - c) * d / f, -d], axis=1)
        p = p @ Rcam.T + CAM_POS
        # 2D footprint (same criterion as posgen's footprint: world z > arm_height)
        ph = p[p[:, 2] > float(PARAMS["arm_height"])]
        if len(ph):
            hx = np.floor((ph[:, 0] - gx0) / gg + 0.5).astype(int)
            hy = np.floor((ph[:, 1] - gy0) / gg + 0.5).astype(int)
            hk = (hx >= 0) & (hx < gnx) & (hy >= 0) & (hy < gny)
            foot2d[hy[hk] * gnx + hx[hk]] = True
        p = p[p[:, 2] > DYN["z_min"]]
        if len(p) == 0:
            continue
        ix = np.floor((p[:, 0] - lo[0]) / gs).astype(int)
        iy = np.floor((p[:, 1] - lo[1]) / gs).astype(int)
        iz = np.floor((p[:, 2] - lo[2]) / gs).astype(int)
        k = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
        n_clip += int((~k).sum())
        occ[ix[k], iy[k], iz[k]] = True
    n_raw = int(occ.sum())
    if int(DYN["dilate"]):
        from scipy import ndimage
        occ = ndimage.binary_dilation(occ, iterations=int(DYN["dilate"]))

    # project the task objects per frame -> screen footprint (per-task spec, see _obj_screen_discs;
    # stick-push: one disc per stick end + the thermos disc / disassemble: ring + handle + peg)
    mask = np.zeros((IMG, IMG), dtype=bool)
    fs = np.asarray(Z["data"]["full_state"], dtype=np.float64)
    ends = np.asarray(Z["meta"]["episode_ends"][()], dtype=int).ravel()
    ep_starts = np.concatenate([[0], ends[:-1]]).astype(int)
    groups = _obj_screen_discs(task, fs, ep_starts)
    n_obj = 0
    for objs, r_obj in groups:
        with np.errstate(invalid="ignore", divide="ignore"):
            pc = (objs - CAM_POS) @ Rcam
            zs = -pc[:, 2]
            uu = c + f * pc[:, 0] / zs
            vv = c + f * pc[:, 1] / zs
            rpx = f * r_obj / np.maximum(zs, 1e-6) + float(DYN["screen_margin_px"])
        ok = np.isfinite(uu) & np.isfinite(vv) & (zs > 0)
        n_obj += int(ok.sum())
        for cu, cv, rr_ in zip(uu[ok], vv[ok], rpx[ok]):
            iu, iv, ir = int(round(cu)), int(round(cv)), int(np.ceil(rr_))
            if iv + ir < 0 or iv - ir >= IMG or iu + ir < 0 or iu - ir >= IMG:
                continue
            y0, y1 = max(0, iv - ir), min(IMG, iv + ir + 1)
            x0, x1 = max(0, iu - ir), min(IMG, iu + ir + 1)
            yy, xx = np.meshgrid(np.arange(y0, y1), np.arange(x0, x1),
                                 indexing="ij")
            mask[y0:y1, x0:x1] |= (((yy - cv) ** 2 + (xx - cu) ** 2) <= rr_ * rr_)
    # for dynhand: workspace (data footprint) distance field + task center (mean object com)
    from scipy import ndimage as _ndi
    work_dist = (_ndi.distance_transform_edt(~foot2d.reshape(gny, gnx))
                 * float(PARAMS["gs"])).astype(np.float32)
    obj_xy = np.concatenate(_obj_coms(task, fs), axis=0)[:, :2]
    task_center = np.asarray(obj_xy.mean(axis=0), dtype=np.float32)
    diag = dict(n_vox_raw=n_raw, n_vox=int(occ.sum()), n_clip=n_clip,
                n_sky=int(n_sky), n_screen=int(mask.sum()), n_obj_px=n_obj,
                n_frames=n_frames, frame_step=step,
                n_cells=int(len(cells_dyn)), n_heights=int(len(heights)),
                n_foot2d=int(foot2d.sum()),
                work_dist_max=float(work_dist.max()),
                task_center=[float(v) for v in task_center],
                n_vis_ge_05=int((n_vis.reshape(len(cells_dyn), len(heights),
                                               len(r_vals)) >= 0.5 * n_frames).sum()))
    frac = (n_vis.reshape(len(cells_dyn), len(heights), len(r_vals))
            .astype(np.float32) / float(max(1, n_frames)))
    return (dict(occ=occ, lo=lo, gs=gs, vis_frac=frac, heights=heights,
                 radii=np.asarray(r_vals, np.float32), cells=cells_dyn,
                 vis_reset=vis_reset.reshape(len(cells_dyn), len(heights),
                                             len(r_vals)),
                 min_vis_frac=float(DYN["min_vis_frac"]),
                 work_dist=work_dist, work_margin=float(DYN["work_margin"]),
                 work_lo=np.asarray([PARAMS["x"][0], PARAMS["y"][0]], np.float32),
                 task_center=task_center),
            mask, diag)


def dyn_volumes(task, force=False, verbose=True):
    """dynsphere data volumes -> (dict(occ, lo, gs), screen_mask); legacy -> (None, None)."""
    rec = _rec(task)
    if rec.get("dyn", "legacy") == "legacy":
        return None, None
    npz, js = dyn_cache_paths(task)
    zarr_ok = pathlib.Path(_dr.mw_expert_zarr(task)).exists()
    key = _dyn_key(task)
    if not force and npz.exists() and js.exists():
        try:
            meta = json.loads(js.read_text(encoding="utf-8"))
            if meta.get("params_key") == key or (not zarr_ok and
                                                 meta.get("task") == task):
                d = np.load(npz)
                if verbose:
                    print("[green_cells_metaworld] dyn cache hit: vox=%d screen=%d "
                          "vis_frac=%s <- %s"
                          % (int(d["occ"].sum()), int(d["screen_mask"].sum()),
                             tuple(d["vis_frac"].shape), npz))
                return (dict(occ=d["occ"], lo=d["lo"], gs=float(d["gs"]),
                             vis_frac=d["vis_frac"], heights=d["heights"],
                             radii=d["radii"], cells=d["cells"],
                             vis_reset=d["vis_reset"],
                             min_vis_frac=float(d["min_vis_frac"]),
                             work_dist=d["work_dist"],
                             work_margin=float(d["work_margin"]),
                             work_lo=np.asarray([PARAMS["x"][0], PARAMS["y"][0]],
                                                np.float32),
                             task_center=d["task_center"]),
                        d["screen_mask"])
        except Exception as e:
            if verbose:
                print("[green_cells_metaworld] dyn cache unreadable (%s); recomputing" % e)
    vox, mask, diag = _compute_dyn(task)
    npz.parent.mkdir(parents=True, exist_ok=True)
    tmp = npz.with_name(".%s.tmp.%d" % (npz.name, os.getpid()))
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, occ=vox["occ"], lo=vox["lo"], gs=vox["gs"],
                            screen_mask=mask, vis_frac=vox["vis_frac"],
                            heights=vox["heights"], cells=vox["cells"],
                            radii=vox["radii"], vis_reset=vox["vis_reset"],
                            min_vis_frac=np.float32(vox["min_vis_frac"]),
                            work_dist=vox["work_dist"],
                            work_margin=np.float32(vox["work_margin"]),
                            task_center=vox["task_center"])
    os.replace(str(tmp), str(npz))
    meta = {"params_key": key, "task": task, "zarr": _dr.mw_expert_zarr(task),
            "xml": str(base_xml(task)), "dyn": DYN, "diag": diag}
    tmp_js = js.with_name(".%s.tmp.%d" % (js.name, os.getpid()))
    tmp_js.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    os.replace(str(tmp_js), str(js))
    if verbose:
        print("[green_cells_metaworld] %s dyn: vox %d(dilate %d) | screen %d | "
              "clipped %d -> %s" % (task, diag["n_vox"], int(DYN["dilate"]),
                                    diag["n_screen"], diag["n_clip"], npz))
    return vox, mask


# --- APPEND ---


# --- cache ------------------------------------------------------------------------
def cache_paths(task):
    return (CACHE_DIR / f"green_cells_mw_{task}.npz",
            CACHE_DIR / f"green_cells_mw_{task}.json")


def _zarr_frames(zarr_path):
    """Dataset frame count; None when unreadable (never invalidate the cache for that)."""
    try:
        import zarr
        return int(zarr.open(str(zarr_path), mode="r")["data"]["depth"].shape[0])
    except Exception:
        return None


def params_key(task):
    """Candidate-validity fingerprint: recipe version + task + zarr path/frames + base XML + params."""
    zarr_path = _dr.mw_expert_zarr(task)
    t = _zarr_frames(zarr_path)
    blob = json.dumps({"v": 1, "task": task, "zarr": zarr_path,
                       "T": -1 if t is None else t,
                       "xml": str(base_xml(task)), "params": PARAMS,
                       "cells": _rec(task)["cells"]}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _meta_ok(meta, task):
    """Cache acceptance when the data is absent: compares task/zarr/xml/params/cells only, not T."""
    return (str(meta.get("task")) == str(task)
            and str(meta.get("zarr")) == _dr.mw_expert_zarr(task)
            and str(meta.get("xml")) == str(base_xml(task))
            and dict(meta.get("params") or {}) == dict(PARAMS)
            and str(meta.get("cells")) == _rec(task)["cells"])


def _missing_msg(task, zarr_path):
    return ("green_cells_metaworld: task=%s needs the ID dataset (data-driven green cells): %s\n"
            "  not found, and no usable cache.\n  %s\n"
            "  (or: DR3D_DATA_ROOT=<root that holds the data>; or set this task to "
            "cells='legacy' to use the analytic rule)" % (task, zarr_path,
                                            _dr.missing_hint_mw(task)))


def load_cells(task, force=False, verbose=True):
    """Read the cache (fingerprint-checked) or compute in place and write it; returns (M,2) xy."""
    rec = _rec(task)
    if rec["cells"] == "legacy":
        return np.asarray(_GA.load_green_cells(verbose=verbose), dtype=np.float32)
    npz, js = cache_paths(task)
    zarr_ok = pathlib.Path(_dr.mw_expert_zarr(task)).exists()
    if not force and npz.exists() and js.exists():
        try:
            meta = json.loads(js.read_text(encoding="utf-8"))
            hit = (meta.get("params_key") == params_key(task)
                   or (not zarr_ok and _meta_ok(meta, task)))
            if hit:
                cells = np.load(npz)["cells"]
                if verbose:
                    print("[green_cells_metaworld] cache hit: %d cells <- %s%s"
                          % (len(cells), npz, "" if zarr_ok else " (data absent)"))
                return cells
        except Exception as e:
            if verbose:
                print("[green_cells_metaworld] cache unreadable (%s); recomputing" % e)
    d = compute_cells(task)
    cells = np.asarray(d["cells"], dtype=np.float32)
    npz.parent.mkdir(parents=True, exist_ok=True)
    tmp = npz.with_name(".%s.tmp.%d" % (npz.name, os.getpid()))
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, cells=cells,
                            in_crop=d["diag"]["in_crop"],
                            blocked=d["diag"]["blocked"],
                            visible=d["diag"]["visible"])
    os.replace(str(tmp), str(npz))
    meta = {"params_key": params_key(task), "task": task,
            "zarr": _dr.mw_expert_zarr(task), "xml": str(base_xml(task)),
            "cells": rec["cells"], "params": PARAMS,
            "n_cells": int(len(cells)), "n_episodes": d["diag"]["n_episodes"],
            "n_in_crop": d["diag"]["n_in_crop"],
            "n_footprint": d["diag"]["n_footprint"],
            "n_visible": d["diag"]["n_visible"]}
    tmp_js = js.with_name(".%s.tmp.%d" % (js.name, os.getpid()))
    tmp_js.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    os.replace(str(tmp_js), str(js))
    if verbose:
        print("[green_cells_metaworld] %s: computed %d cells "
              "(grid %d | in-crop %d | footprint %d | visible %d) -> %s"
              % (task, len(cells), d["diag"]["n_grid"], d["diag"]["n_in_crop"],
                 d["diag"]["n_footprint"], d["diag"]["n_visible"], npz))
    return cells


def cells_source(task, kind="posgen"):
    """Candidate-cell source label (goes to the manifest/logs).

      posgen   : 'legacy' (analytic) | 'zarr:<task>' (data green cells)
      dynsphere: 'table' (old table_grid) | 'zarr:<task>:dynsphere'
    """
    rec = _rec(task)
    if kind == "dynsphere":
        return ("table" if rec.get("dyn", "legacy") == "legacy"
                else "zarr:%s:dynsphere" % task)
    if kind == "dynhand":
        return ("table" if rec.get("dyn", "legacy") == "legacy"
                else "zarr:%s:dynhand" % task)
    return "legacy" if rec["cells"] == "legacy" else "zarr:%s" % task


def cells_status(task):
    """(ok, source, msg): whether the green cells are usable now (printed before eval starts)."""
    npz, js = cache_paths(task)
    if _rec(task)["cells"] == "legacy":
        return True, "legacy", "green_cells_assembly (analytic rule)"
    if npz.exists() and js.exists():
        return True, cells_source(task), str(npz)
    if pathlib.Path(_dr.mw_expert_zarr(task)).exists():
        return True, cells_source(task), "computed and cached in place on first use"
    return False, cells_source(task), _missing_msg(task, _dr.mw_expert_zarr(task))


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default="stick-push", choices=sorted(TASKS))
    ap.add_argument("--kind", default="posgen", choices=["posgen", "dynsphere"],
                    help="posgen=green cells (default) | dynsphere=cells + 3D occupancy + screen footprint")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--show", type=int, default=5)
    ap.add_argument("--cells", default=None, choices=["data", "legacy"],
                    help="temporarily override this task's rule (TASKS defaults untouched)")
    ap.add_argument("--dyn", default=None, choices=["data", "legacy"],
                    help="temporarily override the dynsphere rule (TASKS defaults untouched)")
    ap.add_argument("--margin-cells", type=int, default=None,
                    help="temporarily override the footprint dilation in cells (1 cell=3cm; default 1)")
    ap.add_argument("--min-vis-frac", type=float, default=None,
                    help="temporarily override dynsphere's min data visibility (default 0.5)")
    a = ap.parse_args()
    if a.cells:
        TASKS[a.task]["cells"] = a.cells
    if a.dyn:
        TASKS[a.task]["dyn"] = a.dyn
    if a.margin_cells is not None:
        PARAMS["margin_cells"] = int(a.margin_cells)
    if a.min_vis_frac is not None:
        DYN["min_vis_frac"] = float(a.min_vis_frac)
    if a.kind == "posgen":
        cells = load_cells(a.task, force=a.force)
    else:
        cells = cells_for(a.task, "dynsphere")
        vox, mask = dyn_volumes(a.task, force=a.force)
        if vox is not None:
            print("[green_cells_metaworld] dyn volumes: vox=%d lo=%s gs=%.3f "
                  "screen_px=%d" % (int(vox["occ"].sum()), vox["lo"].tolist(),
                                    vox["gs"], int(mask.sum())))
    print("[green_cells_metaworld] task=%s kind=%s source=%s n=%d "
          "x[%+.3f,%+.3f] y[%+.3f,%+.3f]"
          % (a.task, a.kind, cells_source(a.task, a.kind), len(cells),
             cells[:, 0].min(), cells[:, 0].max(), cells[:, 1].min(), cells[:, 1].max()))
    for x, y in cells[:max(0, a.show)]:
        print("   x=%+.3f y=%+.3f" % (x, y))


if __name__ == "__main__":
    main()
