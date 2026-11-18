"""Voxel-level polluted-cloud rendering (voxel renderer forgery).

Sample a compact blob of workspace cells, then render surface points into them
with ``voxel.voxelizer.synth_cell_points``. Shared by the PI-JEPA clean/poll
pretrain, the DP3 voxel-augmentation experiment and the offline pollution
diagnostics so that all of them use one pollution definition.

Workspace (3D bbox) is task-aware: pass ``ws=<Workspace>`` (see
``voxel.workspace.resolve_workspace``). Omitting it keeps the historical
hammer constants (bit-identical to the pre-multi-task runs).

Two anchor modes:
  * ``mode="bbox"``      uniform random cell anywhere in the box (historical)
  * ``mode="occupied"``  draw the anchor among cells occupied by real points
                         (needs ``occ_cells``); data-driven, task-adaptive
"""
import numpy as np

from diffusion_policy_3d.voxel.voxelizer import (
    synth_cell_points, grid_dims, voxelize,
    WORKSPACE_LO, WORKSPACE_HI, GRID_SIZE_DEFAULT, OBSTACLE_RGB)


def _resolve_ws(ws, grid_size, rgb):
    """Return (lo, hi, grid, rgb) for the forger; ``ws=None`` -> hammer consts."""
    if ws is None:
        lo = np.asarray(WORKSPACE_LO, dtype=np.float64)
        hi = np.asarray(WORKSPACE_HI, dtype=np.float64)
        g = GRID_SIZE_DEFAULT if grid_size is None else grid_size
        c = OBSTACLE_RGB if rgb is None else rgb
        return lo, hi, g, np.asarray(c, np.float32)
    lo = np.asarray(ws.lo, dtype=np.float64)
    hi = np.asarray(ws.hi, dtype=np.float64)
    g = float(ws.grid) if grid_size is None else grid_size
    c = np.asarray(ws.rgb if rgb is None else rgb, np.float32)
    return lo, hi, g, c


def occupied_cells_from_raws(raws, ws):
    """Union of occupied voxel cells over raw frames -> (M, 3) int64.

    Used by ``mode="occupied"``: pollution anchors are then drawn from cells that
    the task's own geometry actually occupies (physically plausible placements,
    independent of how the task's pc frame is oriented).
    """
    lo, hi, g, _ = _resolve_ws(ws, None, None)
    cells = []
    for pc in raws:
        pts = np.asarray(pc, dtype=np.float32)
        if pts.ndim != 2 or len(pts) == 0:
            continue
        fld = voxelize(pts, lo=lo, hi=hi, grid_size=g)
        if len(fld.cell_idx):
            cells.append(np.asarray(fld.cell_idx, dtype=np.int64))
    if not cells:
        dims = np.asarray(grid_dims(lo, hi, g), dtype=np.int64)
        return np.zeros((0, 3), dtype=np.int64), dims
    dims = np.asarray(grid_dims(lo, hi, g), dtype=np.int64)
    return np.unique(np.concatenate(cells, axis=0), axis=0), dims


def occupied_cells_from_zarr(zarr_path, ws):
    """``occupied_cells_from_raws`` on a fullpc zarr (streamed, no big RAM use)."""
    import zarr
    d = zarr.open(str(zarr_path), mode="r")["data"]
    n = int(d["raw_point_cloud_len"].shape[0])

    def _gen():
        for i in range(n):
            k = int(np.asarray(d["raw_point_cloud_len"][i]))
            if k <= 0:
                continue
            yield np.asarray(d["raw_point_cloud"][i], np.float32).reshape(k, 6)

    return occupied_cells_from_raws(_gen(), ws)


def forge_voxel_rows(rng, p_min=10, p_max=50, grid_size=None, rgb=None,
                     ws=None, mode="bbox", occ_cells=None):
    """Forge one compact blob of synthetic points -> (n, 6) float32 rows.

    ``mode="bbox"`` keeps the historical draw order (same rng calls in the same
    sequence), so identical (rng, p_min, p_max) and the hammer workspace
    reproduce the old outputs bit-for-bit.
    """
    lo, hi, g, c = _resolve_ws(ws, grid_size, rgb)
    dims = np.asarray(grid_dims(lo, hi, g), dtype=np.int64)
    n = int(rng.integers(int(p_min), int(p_max) + 1))
    r = int(rng.integers(0, 2))
    offs = np.array([[a, b, cc]
                     for a in range(-r, r + 1)
                     for b in range(-r, r + 1)
                     for cc in range(-r, r + 1)], dtype=np.int64)
    k = int(rng.integers(1, min(len(offs), n) + 1))
    if mode == "occupied":
        if occ_cells is None or len(occ_cells) == 0:
            raise ValueError("mode='occupied' needs a non-empty occ_cells "
                             "(see occupied_cells_from_raws)")
        occ = np.asarray(occ_cells, dtype=np.int64).reshape(-1, 3)
        anchor = occ[int(rng.integers(0, len(occ)))].copy()
        anchor = np.clip(anchor, r, np.maximum(dims - 1 - r, r))
    else:
        anchor = np.array([rng.integers(r, max(int(dims[0]) - r, r + 1)),
                           rng.integers(r, max(int(dims[1]) - r, r + 1)),
                           rng.integers(r, max(int(dims[2]) - r, r + 1))],
                          dtype=np.int64)
    sel = rng.choice(len(offs), size=k, replace=False)
    cells = anchor + offs[sel]
    base = np.full(k, n // k, dtype=np.int64)
    rem = int(n - base.sum())
    if rem > 0:
        pick = rng.choice(k, size=rem, replace=False)
        base[pick] += 1
    pts = synth_cell_points(cells, base, lo, g, rgb=c)
    return np.asarray(pts, dtype=np.float32)

