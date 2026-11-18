"""
Field -> point cloud reconstructor (non-learned; shared by G / F / eval).

Render rule (deterministic prefix truncation): for each cell
    keep = rint(occ_i * cap)  clipped to [0, cap].
Real template points (lex-sorted) are kept first; if occ exceeds the
available template points (G-created / over-capacity cells), the missing
points are synthesised deterministically inside the cell
(see voxelizer.synth_cell_points).
"""
import numpy as np

from diffusion_policy_3d.voxel.voxelizer import VoxelField, synth_cell_points


def reconstruct(field, occ=None):
    """Reconstruct a point cloud (K, 6) float32 from a VoxelField.

    Args:
        field: VoxelField.
        occ: optional (M,) float32 occupancy override in [0, 1]; default field.occ.
    Returns:
        (K, 6) float32 xyz+rgb.
    """
    occ = field.occ if occ is None else np.asarray(occ, dtype=np.float32).reshape(-1)
    assert occ.shape[0] == field.n_cells, (occ.shape, field.n_cells)
    cap = int(field.cap)
    out = []
    for i in range(field.n_cells):
        s, L = int(field.starts[i]), int(field.lengths[i])
        tmpl = field.template[s:s + L]
        keep_i = int(min(cap, max(0, int(np.rint(float(occ[i]) * cap)))))
        if keep_i <= L:
            out.append(tmpl[:keep_i])
        else:
            out.append(tmpl)
            need = keep_i - L
            if need > 0:
                out.append(synth_cell_points(
                    field.cell_idx[i:i + 1],
                    np.array([need], dtype=np.int64),
                    field.lo, field.grid_size))
    if not out:
        return np.zeros((0, 6), dtype=np.float32)
    return np.concatenate(out, axis=0)
