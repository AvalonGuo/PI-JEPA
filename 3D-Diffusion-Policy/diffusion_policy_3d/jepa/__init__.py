"""Voxel-level pollution forging shared with the PI-JEPA trainers.

``forge.py`` renders the compact synthetic blobs that define the polluted view
(``poll = sample_512(concat(raw_point_cloud[f], blob))``) plus the occupied-cell
anchors used by ``mode="occupied"``. Callers:

    PI-JEPA/pijepa_common.py             (forge_voxel_rows,
                                          occupied_cells_from_zarr)
    PI-JEPA/train_pijepa.py              (v5: occupied_cells_from_zarr)
    test/*                               (offline pollution analysis)

Keep this module free of heavy imports: it is imported by every training worker.
"""
from .forge import (
    forge_voxel_rows,
    occupied_cells_from_raws,
    occupied_cells_from_zarr,
)

__all__ = [
    "forge_voxel_rows",
    "occupied_cells_from_raws",
    "occupied_cells_from_zarr",
]
