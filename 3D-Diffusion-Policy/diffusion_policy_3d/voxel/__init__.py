"""Fixed-workspace voxel field: voxelizer (pc->field) and renderer (field->pc).

Non-learned, deterministic, shared by the forger/filter adversarial game and
by deployment.
"""
from .voxelizer import (
    VoxelField,
    voxelize,
    add_synthetic_cells,
    synth_cell_points,
    grid_dims,
    WORKSPACE_LO,
    WORKSPACE_HI,
    GRID_SIZE_DEFAULT,
    CAP_DEFAULT,
    OBSTACLE_RGB,
    OBS_CENTER,
    OBS_HALF,
    OBS_N_POINTS,
    sample_obstacle_points,
)
from .renderer import reconstruct
from .workspace import (
    Workspace,
    TASK_WORKSPACE,
    MODES,
    FORGE_MODES,
    hammer_lo_hi,
    compute_workspace_from_zarr,
    check_zarr_matches_task,
    resolve_workspace,
)

__all__ = [
    "VoxelField",
    "voxelize",
    "add_synthetic_cells",
    "synth_cell_points",
    "grid_dims",
    "reconstruct",
    "WORKSPACE_LO",
    "WORKSPACE_HI",
    "GRID_SIZE_DEFAULT",
    "CAP_DEFAULT",
    "OBSTACLE_RGB",
    "OBS_CENTER",
    "OBS_HALF",
    "OBS_N_POINTS",
    "sample_obstacle_points",
    # task-aware workspace resolution
    "Workspace",
    "TASK_WORKSPACE",
    "MODES",
    "FORGE_MODES",
    "hammer_lo_hi",
    "compute_workspace_from_zarr",
    "check_zarr_matches_task",
    "resolve_workspace",
]
