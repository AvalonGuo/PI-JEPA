"""
Deterministic fixed-workspace voxelizer: point cloud -> voxel occupancy field.

Pure numpy (no torch). The voxel field is the shared, NON-learned "board" on
which the forger G (add occupancy) / filter F (remove occupancy) adversarial
game is played.

Conventions
-----------
* Workspace: fixed lo/hi (same as test/voxelize_pointcloud.py WORKSPACE_DEFAULT).
  Cell i spans [lo + i*grid, lo + (i+1)*grid).
* Occupancy: u_c = count_c / cap, clipped to [0, 1] (cap = per-cell capacity).
* Template: every real point belongs to exactly one cell; within a cell points
  are kept in ascending (x, y, z) lexicographic order, so "renderer takes the
  first round(u*cap) points" is deterministic and lossless for count_c <= cap.
* Points outside the workspace are dropped and counted (n_dropped_outside).
* Assignments: template rows are (cell-major, in-cell lex-sorted); VoxelField
  exposes per-row point_cell / point_in_cell and (voxelize only) inside /
  sort_order so whole-cell keep decisions can be mapped back onto the original
  raster-order rows.
"""
import numpy as np
from dataclasses import dataclass

WORKSPACE_LO = np.array([-0.24570936, -0.07013131, -0.09785966], dtype=np.float64)
WORKSPACE_HI = np.array([0.44392079, 0.45474687, 0.22984123], dtype=np.float64)
GRID_SIZE_DEFAULT = 0.02
CAP_DEFAULT = 16
# observed synthetic obstacle colour (only used to fill G-created cells)
OBSTACLE_RGB = np.array([210.0, 160.0, 59.0], dtype=np.float32)


def _lexsort_xyz(xyz):
    """Ascending (x, y, z) lexicographic order."""
    return np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))


def grid_dims(lo, hi, grid_size):
    return np.ceil((hi - lo) / float(grid_size)).astype(np.int64)


def _row_assignments(lengths):
    """Per-template-row (cell_row, in_cell_rank) given per-cell lengths."""
    lengths = np.asarray(lengths, dtype=np.int64)
    M = lengths.shape[0]
    if M == 0:
        return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    cell_row = np.repeat(np.arange(M, dtype=np.int64), lengths)
    rank = np.concatenate(
        [np.arange(int(L), dtype=np.int64) for L in lengths])
    return cell_row, rank


def _stable_seed(cell):
    """Deterministic 32-bit seed from integer grid coords (hash-free)."""
    c = np.asarray(cell, dtype=np.int64).ravel()
    if c.size < 3:
        c = np.pad(c, (0, 3 - c.size), constant_values=0)
    x = (int(c[0]) * 73856093) ^ (int(c[1]) * 19349663) ^ (int(c[2]) * 83492791)
    return int(x & 0xFFFFFFFF)


def synth_cell_points(cell_idx, counts, lo, grid_size, rgb=OBSTACLE_RGB):
    """Deterministic in-cell synthetic points for empty / over-capacity cells.

    Args:
        cell_idx: (K, 3) int64 grid coordinates.
        counts:   (K,) int64 >= 0 points per cell.
        lo:       workspace origin (3,).
        grid_size: voxel size in metres.
        rgb:      colour broadcast to generated points.
    Returns:
        (sum(counts), 6) float32 xyz+rgb, cell-major order; within each cell
        points are lex-sorted so prefix truncation is deterministic.
    """
    cell_idx = np.asarray(cell_idx, dtype=np.int64).reshape(-1, 3)
    counts = np.asarray(counts, dtype=np.int64).reshape(-1)
    lo = np.asarray(lo, dtype=np.float64)
    pts = []
    for ci, n in zip(cell_idx, counts):
        n = int(n)
        if n <= 0:
            continue
        rng = np.random.default_rng(_stable_seed(ci))
        local = (rng.random((n, 3)) * float(grid_size)).astype(np.float32)
        # cell corner (lo + ci*grid) + uniform in [0, grid): points stay INSIDE
        # the declared voxel so re-voxelization maps them back to the same cell.
        base = (np.asarray(ci, dtype=np.float64) * grid_size + lo)
        xyz = base.astype(np.float32) + local
        if rgb is not None:
            cols = np.tile(np.asarray(rgb, dtype=np.float32), (n, 1))
        else:
            cols = np.zeros((n, 3), dtype=np.float32)
        p = np.concatenate([xyz, cols], axis=1)
        order = _lexsort_xyz(p[:, :3])
        pts.append(p[order])
    if not pts:
        return np.zeros((0, 6), dtype=np.float32)
    return np.concatenate(pts, axis=0)


@dataclass
class VoxelField:
    lo: np.ndarray       # workspace origin (3,)
    hi: np.ndarray       # workspace top corner (3,)
    grid_size: float
    cap: int
    dims: np.ndarray     # (3,) int64 grid dimensions
    cell_idx: np.ndarray  # (M, 3) int64, sorted lexicographically
    counts: np.ndarray   # (M,) int64
    occ: np.ndarray      # (M,) float32 in [0, 1]
    starts: np.ndarray   # (M,) int64 start index into template
    lengths: np.ndarray  # (M,) int64
    template: np.ndarray  # (P, 6) float32 xyz+rgb, lex-sorted within cell
    n_dropped_outside: int = 0
    # --- per-point assignments ---
    point_cell: np.ndarray = None     # (P,) int64: cell row for each template row
    point_in_cell: np.ndarray = None  # (P,) int64: rank of row inside its cell
    inside: np.ndarray = None         # (N_finite,) bool: input row in workspace (voxelize only)
    sort_order: np.ndarray = None     # (P,) int64: template[k] == pts_in[sort_order[k]]
    n_input: int = 0                  # number of finite input rows (voxelize only)

    @property
    def n_cells(self) -> int:
        return int(self.cell_idx.shape[0])

    @property
    def n_points(self) -> int:
        return int(self.template.shape[0]) if self.template.ndim == 2 else 0


def _assemble(cell_rows, chunk_rows, lo, hi, grid_size, cap):
    """Canonical field from parallel lists of cell coords and template chunks."""
    cell_idx = np.asarray(cell_rows, dtype=np.int64).reshape(-1, 3)
    if len(cell_idx) == 0:
        return VoxelField(
            lo=lo, hi=hi, grid_size=grid_size, cap=cap,
            dims=grid_dims(lo, hi, grid_size),
            cell_idx=np.zeros((0, 3), dtype=np.int64),
            counts=np.zeros(0, dtype=np.int64),
            occ=np.zeros(0, dtype=np.float32),
            starts=np.zeros(0, dtype=np.int64),
            lengths=np.zeros(0, dtype=np.int64),
            template=np.zeros((0, 6), dtype=np.float32),
            point_cell=np.zeros(0, dtype=np.int64),
            point_in_cell=np.zeros(0, dtype=np.int64))
    order = np.lexsort((cell_idx[:, 2], cell_idx[:, 1], cell_idx[:, 0]))
    cell_idx = cell_idx[order]
    chunks = [chunk_rows[i] for i in order]
    template = np.concatenate(chunks, axis=0)
    lengths = np.asarray([c.shape[0] for c in chunks], dtype=np.int64)
    counts = lengths.copy()
    starts = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(lengths)[:-1]], axis=0)
    occ = np.clip(counts.astype(np.float64) / cap, 0.0, 1.0).astype(np.float32)
    cell_row, in_cell = _row_assignments(lengths)
    return VoxelField(
        lo=lo, hi=hi, grid_size=grid_size, cap=cap,
        dims=grid_dims(lo, hi, grid_size),
        cell_idx=cell_idx, counts=counts, occ=occ,
        starts=starts, lengths=lengths, template=template,
        point_cell=cell_row, point_in_cell=in_cell)


def voxelize(points, lo=None, hi=None, grid_size=GRID_SIZE_DEFAULT, cap=CAP_DEFAULT):
    """Voxelize a point cloud into a canonical VoxelField.

    Args:
        points: (N, 3) or (N, 6) float32 xyz / xyz+rgb.
    Returns:
        VoxelField with one cell per occupied grid cell.
    """
    lo = np.asarray(WORKSPACE_LO, dtype=np.float64) if lo is None \
        else np.asarray(lo, dtype=np.float64)
    hi = np.asarray(WORKSPACE_HI, dtype=np.float64) if hi is None \
        else np.asarray(hi, dtype=np.float64)
    cap = int(cap)
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] not in (3, 6):
        raise ValueError(f"points must be (N,3) or (N,6), got {pts.shape}")
    if pts.shape[1] == 3:
        pts = np.concatenate([pts, np.zeros_like(pts)], axis=1)
    pts = pts[np.isfinite(pts).all(axis=1)]
    xyz = pts[:, :3]
    dims = grid_dims(lo, hi, grid_size)
    idx = np.floor((xyz - lo) / float(grid_size)).astype(np.int64)
    inside = np.all(idx >= 0, axis=1) & np.all(idx < dims, axis=1)
    n_out = int((~inside).sum())
    empty = VoxelField(
        lo=lo, hi=hi, grid_size=grid_size, cap=cap, dims=dims,
        cell_idx=np.zeros((0, 3), dtype=np.int64),
        counts=np.zeros(0, dtype=np.int64),
        occ=np.zeros(0, dtype=np.float32),
        starts=np.zeros(0, dtype=np.int64),
        lengths=np.zeros(0, dtype=np.int64),
        template=np.zeros((0, 6), dtype=np.float32),
        n_dropped_outside=n_out,
        point_cell=np.zeros(0, dtype=np.int64),
        point_in_cell=np.zeros(0, dtype=np.int64),
        inside=inside.copy(),
        sort_order=np.zeros(0, dtype=np.int64),
        n_input=int(len(pts)))
    if not bool(inside.any()):
        return empty
    pts_in = pts[inside]
    idx_in = idx[inside]
    # global (x, y, z) lex order: within a cell it is also the canonical order
    order = _lexsort_xyz(pts_in[:, :3])
    in_sorted = np.arange(len(pts_in), dtype=np.int64)[order]
    ids_sorted = idx_in[order]

    # NOTE: same cell may be NON-contiguous after xyz sorting (neighbour cells
    # with a lower y/z can interleave when x falls in the same x-cell), so group
    # explicitly by unique cell instead of assuming contiguous runs.
    uq, _, inv, cnt = np.unique(
        ids_sorted, axis=0, return_index=True, return_inverse=True,
        return_counts=True)
    group = np.argsort(inv, kind="stable")   # stable keeps in-cell xyz order
    template_idx = in_sorted[group]          # template[k] == pts_in[template_idx[k]]
    template = pts_in[template_idx]
    cell_idx = uq                            # unique cells, lex-sorted
    lengths = cnt.copy()
    counts = lengths.copy()
    starts = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(cnt)[:-1]], axis=0)
    occ = np.clip(counts.astype(np.float64) / cap, 0.0, 1.0).astype(np.float32)
    cell_row, in_cell = _row_assignments(lengths)
    assert np.array_equal(cell_row, inv[group]), "cell_row grouping mismatch"
    return VoxelField(
        lo=lo, hi=hi, grid_size=grid_size, cap=cap, dims=dims,
        cell_idx=cell_idx, counts=counts, occ=occ,
        starts=starts, lengths=lengths, template=template,
        n_dropped_outside=n_out,
        point_cell=cell_row, point_in_cell=in_cell,
        inside=inside.copy(),
        sort_order=np.asarray(template_idx, dtype=np.int64),
        n_input=int(len(pts)))


def add_synthetic_cells(field, cell_idx, counts, rgb=OBSTACLE_RGB):
    """Merge G-activated cells (no real point template) into a canonical field.

    New cells receive deterministically synthesised in-cell points; every
    existing real point is preserved. Returns a new VoxelField.
    """
    ci = np.asarray(cell_idx, dtype=np.int64).reshape(-1, 3)
    cnt = np.asarray(counts, dtype=np.int64).reshape(-1)
    assert len(ci) == len(cnt), (ci.shape, cnt.shape)
    cell_rows, chunk_rows = [], []
    for i in range(field.n_cells):
        s, L = int(field.starts[i]), int(field.lengths[i])
        cell_rows.append(field.cell_idx[i])
        chunk_rows.append(field.template[s:s + L])
    for c, n in zip(ci, cnt):
        n = int(n)
        if n <= 0:
            continue
        cell_rows.append(c)
        chunk_rows.append(synth_cell_points(
            c.reshape(1, 3), np.array([n], dtype=np.int64),
            field.lo, field.grid_size, rgb=rgb))
    if not cell_rows:
        return field
    return _assemble(cell_rows, chunk_rows, field.lo, field.hi,
                     field.grid_size, field.cap)

# ---------------------------------------------------------------------------
# synthetic block (fixed cube appended to a clean point cloud)
# ---------------------------------------------------------------------------
# Geometry / colour of the offline "pollution" diagnostics block. Measured from a
# real sim render in the same PC coordinate frame: 47 points @ RGB (210, 160, 59).
# (moved here from the deleted test/synthesize_obstacle_pc.py)
OBS_CENTER = np.array([-0.220, 0.157, -0.067], dtype=np.float32)
OBS_HALF = np.array([0.026, 0.028, 0.027], dtype=np.float32)
OBS_N_POINTS = 47


def sample_obstacle_points(n=OBS_N_POINTS, seed=0, rgb=OBSTACLE_RGB):
    """Uniformly sample points inside the synthetic obstacle cube -> (n, 6).

    Deterministic for a given (n, seed): drop-in replacement for the former
    ``test/synthesize_obstacle_pc.sample_obstacle_points`` (same rng draw order
    and same colour), so existing pollution experiments stay reproducible.
    """
    rng = np.random.RandomState(seed)
    pts = OBS_CENTER + (rng.rand(n, 3).astype(np.float32) * 2 - 1) * OBS_HALF
    color = np.tile(np.asarray(rgb, dtype=np.float32), (n, 1)) \
        + rng.randn(n, 3).astype(np.float32) * 5.0
    return np.concatenate([pts, np.clip(color, 0, 255)], axis=1)  # (n, 6)

