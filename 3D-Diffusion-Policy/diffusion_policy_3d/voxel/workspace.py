"""Task-aware workspace (3D bbox) resolution for the voxel forger / voxelizer.

Single source of truth for the fixed **3D axis-aligned bounding box** in which
the Phase-1 voxel-renderer pollution is forged. The box lives in the *point
cloud frame* of the task, i.e. the frame of ``raw_point_cloud`` in the fullpc
zarr -- after the wrapper pipeline

    camera rays -> world -> pc_transform (45 deg about x for hammer/door)
                -> scale -> offset (hammer/door: [0,0,1]) -> crop (z >= ...)

so its axes are NOT the world x/y/z and differ per task (pen uses
``transform=None, offset=[0,0,0]`` with camera ``vil_camera``).

Provenance
----------
* ``adroit_hammer``: frozen table entry, bit-identical to the historical
  ``voxelizer.WORKSPACE_LO/WORKSPACE_HI``; those in turn equal the union bbox of
  **all** raw point clouds of ``adroit_hammer_expert.zarr`` (verified:
  abs diff = 0). So ``mode="auto"`` reproduces the historical hammer run exactly.
* other tasks: derived from that task's own zarr, same "union bbox over
  frames" convention (``compute_workspace_from_zarr``).

CLI (prints a paste-ready table entry)::

    python -m diffusion_policy_3d.voxel.workspace \\
        --zarr /hy-tmp/3DP/data/adroit_door_expert.zarr --task adroit_door
"""
import argparse
import pathlib
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .voxelizer import (
    GRID_SIZE_DEFAULT, OBSTACLE_RGB, WORKSPACE_HI, WORKSPACE_LO, grid_dims)

# --- frozen table: task -> (lo, hi), verified against the task's dataset ------
TASK_WORKSPACE = {
    # == voxelizer.WORKSPACE_LO/HI == union bbox of adroit_hammer_expert.zarr
    "adroit_hammer": (
        (-0.24570936, -0.07013131, -0.09785966),
        (0.44392079, 0.45474687, 0.22984123),
    ),
    # derive from the task's own fullpc zarr until frozen here (see CLI above);
    # door: 45 deg transform + offset[0,0,1] + crop z>=-0.499
    # pen : transform=None + offset[0,0,0] + cam vil_camera + crop z>=-0.79
    #
    # First derivation (kept as comment, NOT frozen -> never stale if the dataset
    # is re-collected; re-run the CLI to refresh):
    #   adroit_door @ adroit_door_expert_fullpc.zarr (2026-09-15)
    #     lo = (-0.49403067, -0.46451172, -0.49152257)
    #     hi = ( 0.36615492,  0.61390732,  0.09907094)
    #     dims = [44, 54, 30] (71280 cells @2cm)
    #   metaworld_assembly @ metaworld_assembly_expert.zarr (2026-09-30, then named
    #     metaworld_assembly_expert_fullpc.zarr; single collector now writes _expert.zarr)
    #     lo = (-0.267715, -0.912606, -0.795000)     # camera-frame union bbox
    #     hi = ( 0.558204, -0.400000, -0.103649)
    #     dims = [42, 26, 35] (38220 cells @2cm)
    #     NOTE: metaworld raw point clouds already sit at the policy crop (z >=
    #     -0.795); the forger box covers the assembly table + peg + hole.
    "adroit_door": None,
    "adroit_pen": None,
    "metaworld_assembly": None,
}

MODES = ("auto", "hammer", "explicit")
FORGE_MODES = ("bbox", "occupied")


def hammer_lo_hi() -> Tuple[np.ndarray, np.ndarray]:
    """Historical hammer workspace constants (== the frozen table entry)."""
    return (np.array(WORKSPACE_LO, dtype=np.float64),
            np.array(WORKSPACE_HI, dtype=np.float64))


@dataclass(frozen=True)
class Workspace:
    """3D axis-aligned workspace box + voxel grid of the forger."""

    lo: np.ndarray
    hi: np.ndarray
    grid: float = float(GRID_SIZE_DEFAULT)
    rgb: np.ndarray = field(default_factory=lambda: np.array(OBSTACLE_RGB, np.float32))
    mode: str = "auto"
    source: str = ""
    task: Optional[str] = None

    def dims(self) -> np.ndarray:
        """Voxel grid dims = ceil((hi - lo) / grid)."""
        return np.asarray(grid_dims(self.lo, self.hi, self.grid), dtype=np.int64)

    def size(self) -> np.ndarray:
        return np.asarray(self.hi, np.float64) - np.asarray(self.lo, np.float64)

    def n_cells(self) -> int:
        return int(np.prod(self.dims()))

    def as_tuple(self) -> Tuple[np.ndarray, np.ndarray]:
        return (np.asarray(self.lo, dtype=np.float64),
                np.asarray(self.hi, dtype=np.float64))

    def to_dict(self) -> dict:
        return {
            "lo": [float(v) for v in np.asarray(self.lo, np.float64)],
            "hi": [float(v) for v in np.asarray(self.hi, np.float64)],
            "grid": float(self.grid),
            "rgb": [float(v) for v in np.asarray(self.rgb, np.float64)],
            "dims": [int(v) for v in self.dims()],
            "mode": str(self.mode),
            "source": str(self.source),
            "task": None if self.task is None else str(self.task),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Workspace":
        d = dict(d or {})
        lo = d.get("lo", None)
        hi = d.get("hi", None)
        if lo is None or hi is None:          # fall back to the frozen hammer box
            lo, hi = hammer_lo_hi()
        return cls(lo=np.asarray(lo, np.float64), hi=np.asarray(hi, np.float64),
                   grid=float(d.get("grid", GRID_SIZE_DEFAULT)),
                   rgb=np.asarray(d.get("rgb", OBSTACLE_RGB), np.float32),
                   mode=str(d.get("mode", "legacy")),
                   source=str(d.get("source", "ckpt-hyper")),
                   task=(None if d.get("task") is None else str(d["task"])))

    def describe(self) -> str:
        return ("workspace[%s] lo=%s hi=%s size=%s grid=%sg dims=%s "
                "cells=%d rgb=%s source=%s" % (
                    self.task or "-",
                    np.round(self.lo, 6).tolist(), np.round(self.hi, 6).tolist(),
                    np.round(self.size(), 6).tolist(), self.grid,
                    self.dims().tolist(), self.n_cells(),
                    np.round(self.rgb, 1).tolist(), self.source))


def compute_workspace_from_zarr(zarr_path, quantile=None,
                                frames: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Union bbox of every ``raw_point_cloud`` frame in a fullpc zarr.

    Same convention as ``test/voxelize_pointcloud.py::compute_workspace``
    (min/max over all frames) which produced the historical hammer constants.
    ``quantile=(q_lo, q_hi)`` uses robust percentiles instead of min/max.
    """
    import zarr
    root = zarr.open(str(zarr_path), mode="r")
    if "data" not in root.group_keys():
        raise FileNotFoundError(
            "{} is not a valid dataset (no 'data' group)".format(zarr_path))
    d = root["data"]
    if "raw_point_cloud" not in d or "raw_point_cloud_len" not in d:
        raise KeyError(
            "{} has no raw_point_cloud/raw_point_cloud_len -> not a fullpc zarr "
            "(the forger needs the un-sampled point clouds); collect it with "
            "scripts/gen_demonstration_adroit.sh".format(zarr_path))
    n = int(d["raw_point_cloud_len"].shape[0])
    if frames is not None:
        n = min(n, int(frames))
    if quantile is None:
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for i in range(n):
            k = int(np.asarray(d["raw_point_cloud_len"][i]))
            if k <= 0:
                continue
            xyz = np.asarray(d["raw_point_cloud"][i], np.float32).reshape(k, 6)[:, :3]
            lo = np.minimum(lo, xyz.min(axis=0))
            hi = np.maximum(hi, xyz.max(axis=0))
        return lo.astype(np.float64), hi.astype(np.float64)
    # robust mode: subsample frames, then percentiles
    step = max(1, n // 64)
    acc = []
    for i in range(0, n, step):
        k = int(np.asarray(d["raw_point_cloud_len"][i]))
        if k <= 0:
            continue
        acc.append(np.asarray(d["raw_point_cloud"][i], np.float32).reshape(k, 6)[:, :3])
    if not acc:
        raise ValueError("no raw points found in {}".format(zarr_path))
    pts = np.concatenate(acc, axis=0)
    return (np.percentile(pts, float(quantile[0]) * 100.0, axis=0).astype(np.float64),
            np.percentile(pts, float(quantile[1]) * 100.0, axis=0).astype(np.float64))


def check_zarr_matches_task(task: Optional[str], zarr_path) -> None:
    """Fail fast when a task is fed another task's dataset.

    Guards the historical footgun where ``task=adroit_door`` silently trained on
    ``adroit_hammer_expert.zarr`` (the python-side default).
    """
    if not task or not zarr_path:
        return
    base = pathlib.Path(str(zarr_path)).name
    token = str(task)
    if token in base or base.startswith(token.replace("adroit_", "")):
        return
    raise ValueError(
        "task/zarr mismatch: task='{t}' but zarr='{z}'. Pass the matching "
        "zarr (e.g. /hy-tmp/3DP/data/{t}_expert.zarr) or use --workspace "
        "hammer for a deliberate hammer-workspace run.".format(t=task, z=base))


def resolve_workspace(task=None, zarr=None, mode="auto", lo=None, hi=None,
                      grid=GRID_SIZE_DEFAULT, rgb=OBSTACLE_RGB,
                      quantile=None) -> Workspace:
    """Resolve the forger workspace for ``task`` (see module docstring).

    mode="auto"     frozen TASK_WORKSPACE table first, else derive from ``zarr``
    mode="hammer"   historical hammer constants (bit-identical to the old runs)
    mode="explicit" use the given ``lo``/``hi``
    """
    if mode not in MODES:
        raise ValueError(
            "workspace mode must be one of {}, got {!r}".format(MODES, mode))
    rgb_arr = np.asarray(rgb, np.float32)
    if mode == "explicit":
        if lo is None or hi is None:
            raise ValueError("workspace mode='explicit' needs both lo and hi")
        return Workspace(lo=np.asarray(lo, np.float64), hi=np.asarray(hi, np.float64),
                         grid=float(grid), rgb=rgb_arr, mode=mode,
                         source="cli-explicit", task=task)
    if mode == "hammer":
        hlo, hhi = hammer_lo_hi()
        return Workspace(lo=hlo, hi=hhi, grid=float(grid), rgb=rgb_arr, mode=mode,
                         source="hammer-const", task=task)

    # mode == "auto"
    entry = TASK_WORKSPACE.get(str(task), None) if task else None
    if entry is not None:
        return Workspace(lo=np.asarray(entry[0], np.float64),
                         hi=np.asarray(entry[1], np.float64),
                         grid=float(grid), rgb=rgb_arr, mode=mode,
                         source="table", task=task)
    if not zarr:
        raise ValueError(
            "workspace mode='auto' needs --zarr for task {!r} (no frozen table "
            "entry yet). Pass the task's fullpc zarr, or use --workspace hammer "
            "to deliberately reuse the hammer box.".format(task))
    check_zarr_matches_task(task, zarr)
    dlo, dhi = compute_workspace_from_zarr(zarr, quantile=quantile)
    src = "derived:" + pathlib.Path(str(zarr)).name
    if quantile is not None:
        src += "@q{}-{}".format(quantile[0], quantile[1])
    return Workspace(lo=dlo, hi=dhi, grid=float(grid), rgb=rgb_arr, mode=mode,
                     source=src, task=task)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--task", default=None)
    ap.add_argument("--mode", choices=MODES, default="auto")
    ap.add_argument("--quantile", nargs=2, type=float, default=None,
                    help="robust percentiles, e.g. 0.001 0.999")
    ap.add_argument("--grid", type=float, default=float(GRID_SIZE_DEFAULT))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    ws = resolve_workspace(task=a.task, zarr=a.zarr, mode=a.mode,
                           grid=a.grid, quantile=a.quantile)
    print(ws.describe())
    if a.json:
        import json
        print(json.dumps(ws.to_dict(), indent=2))
    print("\n# paste into voxel/workspace.py::TASK_WORKSPACE")
    print('    "{}": ('.format(a.task or "<task>"))
    print("        ({}),".format(", ".join("{:.8f}".format(v) for v in ws.lo)))
    print("        ({}),".format(", ".join("{:.8f}".format(v) for v in ws.hi)))
    print("    ),")


if __name__ == "__main__":
    main()


