"""Point-cloud sampling helper (pure numpy/torch, no heavy deps).

``sample_512`` replicates the *runtime* point-cloud pipeline used by
``diffusion_policy_3d.gym_util.mujoco_point_cloud.point_cloud_sampling``:
zero-pad when ``N <= num_points``, otherwise torch3d FPS.

History: this lived in ``diffusion_policy_3d/distill/oracle.py`` together with
the frozen-DP3 action oracle (``DPActionOracle`` / ``action_gap``). The oracle
belonged to the archived G/F adversarial-distillation line and has been removed;
this sampler is still used by the PI-JEPA polluted view
(``PI-JEPA/pijepa_common.py``, ``PI-JEPA/train_pijepa.py``)
and several offline analysis scripts, so it was moved here.

Fixed-quota ("core") composition -- the *single* implementation shared by
the PI-JEPA polluted view, the downstream dataset and the eval env wrappers:

    n_core = num_points - voxel_max          # door: 512 - 50 = 462
    core   = FPS(raw, min(N, n_core))
    clean  = pad(core)                       # n_core real + (num_points-n_core) zeros
    poll   = core_add(clean, blob)           # blob goes into the *free tail slots*

``len(blob) <= voxel_max = num_points - n_core`` by construction
(``jepa/forge.py``: ``n = rng.integers(p_min, p_max + 1)``), so pollution never
competes with real rows: churn = 0 and the poll<->clean difference is *pure*
pollution. Determinism: torch3d ``sample_farthest_points`` defaults to
``random_start_point=False``, but the CPU and CUDA kernels pick slightly
different rows, so the device MUST be pinned (``device="cpu"`` by default).
"""
import numpy as np
import torch

__all__ = ["sample_512", "core_of", "core_pad", "core_add"]


def sample_512(points, num_points=512, device=None):
    """Replicate runtime point_cloud_sampling (N<=512 pad; else torch3d FPS)."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] not in (3, 6):
        raise ValueError(f"points must be (N,3)/(N,6), got {pts.shape}")
    if pts.shape[0] <= num_points:
        pad = np.zeros((num_points - pts.shape[0], pts.shape[1]), dtype=np.float32)
        return np.concatenate([pts, pad], axis=0)
    import pytorch3d.ops
    dev = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    xyz = torch.from_numpy(pts[:, :3].copy()).float().unsqueeze(0).to(dev)
    K = torch.tensor([num_points], dtype=torch.long, device=dev)
    _, inds = pytorch3d.ops.sample_farthest_points(points=xyz, K=K)
    return pts[inds[0].cpu().numpy()]


def _fps_rows(points, num_points, device="cpu"):
    """torch3d FPS on a *pinned* device (returns the selected row indices).

    ``random_start_point`` is left at its default ``False`` -> for a fixed input
    tensor, device and process the selected rows are reproducible. Note the CPU
    and CUDA kernels do NOT agree with each other, hence the explicit device.
    """
    import pytorch3d.ops
    dev = torch.device(device)
    pts = np.asarray(points, dtype=np.float32)
    xyz = torch.from_numpy(pts[:, :3].copy()).float().unsqueeze(0).to(dev)
    K = torch.tensor([int(num_points)], dtype=torch.long, device=dev)
    _, inds = pytorch3d.ops.sample_farthest_points(points=xyz, K=K)
    return inds[0].cpu().numpy().astype(np.int64)


def core_of(points, n_core, device="cpu"):
    """Real rows of the fixed-quota construction -> (min(N, n_core), D).

    ``N <= n_core`` keeps every row (same zero-pad semantics as ``sample_512``).
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] not in (3, 6):
        raise ValueError(f"points must be (N,3)/(N,6), got {pts.shape}")
    if n_core is None or int(n_core) <= 0 or pts.shape[0] <= int(n_core):
        return pts
    return pts[_fps_rows(pts, int(n_core), device=device)]


def core_pad(points, n_core=None, num_points=512, device="cpu"):
    """Fixed-quota composition: FPS the real rows to ``n_core`` + zero-pad.

    ``n_core=None``/``<=0`` reproduces ``sample_512`` exactly (historical path,
    used by hammer/pen). With ``n_core > 0`` the real rows always form a *prefix*
    of the result and the remaining ``num_points - n_core`` slots stay zero, so
    ``core_add`` can inject pollution without touching a single real row.
    """
    if n_core is None or int(n_core) <= 0:
        return sample_512(points, num_points=num_points)
    pts = core_of(points, int(n_core), device=device)
    if pts.shape[0] > int(num_points):          # cannot happen (n_core<num_points)
        pts = pts[: int(num_points)]
    pad = np.zeros((int(num_points) - pts.shape[0], pts.shape[1]), dtype=np.float32)
    return np.concatenate([pts, pad], axis=0)


def core_add(core_pc, extra, n_real=None):
    """Write ``extra`` rows into the free (all-zero) tail slots of ``core_pc``.

    ``core_pc`` is a ``core_pad`` output: its row 0 .. n_real-1 are real, the rest
    are zeros. Pollution is therefore written at ``n_real`` onward -> no
    competition, no reordering, churn = 0.  Rows that do not fit the reserved
    quota (``num_points - n_core >= voxel_max`` by construction) are dropped.
    """
    pc = np.asarray(core_pc, dtype=np.float32).copy()
    ex = np.asarray(extra, dtype=np.float32)
    if ex.ndim == 1:
        ex = ex.reshape(1, -1)
    if len(ex) == 0:
        return pc
    if n_real is None:
        free = np.flatnonzero(~np.any(pc != 0, axis=1))
        k = int(free[0]) if len(free) else int(pc.shape[0])
    else:
        k = int(n_real)
    n = int(min(len(ex), pc.shape[0] - k))
    if n <= 0:
        return pc
    pc[k:k + n] = ex[:n, : pc.shape[1]]
    return pc
