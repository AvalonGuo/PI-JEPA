#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""dr3d bench: dataset root resolution (single source of truth).

The dataset is a **replaceable external dependency** (large, moves between
machines), so it is not part of any resource registry. This module only fixes
one overridable root plus the naming convention; all other code asks here.

    DR3D_DATA_ROOT                      env var (default /hy-tmp/3DP/data)
    adroit_<task>_expert.zarr          product of the adroit collector script
    metaworld_<task>_expert.zarr       product of the metaworld collector script

Both suites share one convention (one collector -> one artifact): the zarr used
for training is also the data source for the dr3d green cells and for JEPA.

Executable hint when missing, see missing_hint(task):
    bash scripts/gen_demonstration_adroit.sh <task> 0 10 42 0 \\
         "${DR3D_DATA_ROOT:-/hy-tmp/3DP/data}"
"""
import os

DEFAULT_ROOT = "/hy-tmp/3DP/data"


def data_root():
    """Current dataset root (read from env each call, so runtime switching works)."""
    return os.environ.get("DR3D_DATA_ROOT", DEFAULT_ROOT)


def adroit_expert_zarr(task):
    """adroit task -> <root>/adroit_<task>_expert.zarr (path only, no existence check).

    Matches `dataset.zarr_path` in `config/task/adroit_<task>.yaml`: the same file
    feeds DP3 training, the green cells (`data/depth`, 84x84) and the pollution
    pipeline (`data/raw_point_cloud`).
    """
    return os.path.join(data_root(), f"adroit_{task}_expert.zarr")


def mw_expert_zarr(task):
    """metaworld task -> <root>/metaworld_<task>_expert.zarr (path only, no existence check).

    Matches `dataset.zarr_path` in `config/task/metaworld_<task>.yaml`. The green
    cells (default) only need `data/depth` (128x128, same corner2 camera as the
    policy), provided by the same single-collector artifact as on the adroit side.
    """
    return os.path.join(data_root(), f"metaworld_{task}_expert.zarr")


def missing_hint(task):
    """One-line command to copy when a task has no ID dataset (printed by eval)."""
    return (f"Run bash scripts/gen_demonstration_adroit.sh {task} 0 10 42 0 "
            f"{data_root()}")


def missing_hint_mw(task):
    """One-line command when a metaworld task has no ID dataset (posgen cells)."""
    return (f"Run bash scripts/gen_demonstration_metaworld.sh {task} 0 10 42 0 "
            f"{data_root()}")
