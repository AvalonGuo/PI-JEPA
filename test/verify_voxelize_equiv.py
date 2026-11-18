#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Equivalence check: ``test/voxelize_pointcloud.py::voxelize_numpy`` vs the
package implementation ``diffusion_policy_3d.voxel.voxelizer.voxelize``.

Purpose: prove the script can **delegate its grid binning to the package**
(single source of truth: ``voxel/voxelizer.py`` + ``voxel/workspace.py``) before
the refactor lands. Read-only: no training, no sim, no writes except the report.

Checks per frame:
  * occupied cell set            (must be exactly equal; both are lex-sorted)
  * per-cell counts              (must be exactly equal; ``cap`` only scales
                                  ``occ`` and never truncates the template)
  * cell centers                 max |Δ|  (formula: (cell + 0.5) * grid + lo)
  * per-cell mean RGB            max |Δ|  (script: np.add.at / counts;
                                  package: add.reduceat(template)/counts)
  * dropped-outside bookkeeping  (SEMANTICS differ on purpose -> reported only)

Usage:
  .venv/bin/python test/verify_voxelize_equiv.py            # hammer + door
  .venv/bin/python test/verify_voxelize_equiv.py --frames 10 --out <json>
"""
import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[1]
_TEST_DIR = pathlib.Path(__file__).resolve().parent
_DP3_DIR = _REPO / "3D-Diffusion-Policy"
for _p in (str(_DP3_DIR), str(_REPO), str(_TEST_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from diffusion_policy_3d.voxel.voxelizer import voxelize  # noqa: E402
from diffusion_policy_3d.voxel.workspace import (  # noqa: E402
    compute_workspace_from_zarr)
import voxelize_pointcloud as vp  # noqa: E402

DEFAULT_ZARRS = [
    "/hy-tmp/3DP/data/adroit_hammer_expert_fullpc.zarr",
    "/hy-tmp/3DP/data/adroit_door_expert_fullpc.zarr",
]


def _raw_frame(d, f):
    k = int(np.asarray(d["raw_point_cloud_len"][f]))
    return np.asarray(d["raw_point_cloud"][f], np.float32).reshape(k, 6)


def pkg_voxelize(xyz, rgb, lo, hi, grid):
    """Package call mapped onto the script's dict contract."""
    pc = np.concatenate([xyz, rgb], axis=1) if rgb is not None and rgb.shape[1] == 3 \
        else xyz
    fld = voxelize(pc, lo=lo, hi=hi, grid_size=grid)
    if fld.n_cells == 0:
        return None, fld
    centers = (fld.cell_idx.astype(np.float64) + 0.5) * float(grid) + np.asarray(lo)
    avg_rgb = np.add.reduceat(fld.template[:, 3:6], fld.starts, axis=0) / \
        fld.counts[:, None].astype(np.float64)
    out = {"method": "numpy(pkg)", "centers": centers.astype(np.float32),
           "counts": fld.counts, "avg_rgb": avg_rgb.astype(np.float32),
           "grid_index": fld.cell_idx, "voxel_size": grid,
           "workspace_lo": lo, "workspace_hi": hi,
           "n_input": int(fld.n_input),
           "n_dropped_outside": int(fld.n_dropped_outside)}
    return out, fld


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zarr", nargs="*", default=DEFAULT_ZARRS)
    p.add_argument("--frames", type=int, default=6, help="frames probed per zarr")
    p.add_argument("--grid", type=float, default=0.02)
    p.add_argument("--out", default="/hy-tmp/3DP/vis/voxel_equiv/report.json")
    a = p.parse_args()

    import zarr
    rep = {"frames": int(a.frames), "grid": float(a.grid), "per_zarr": [],
           "verdict": {}}
    t0 = time.time()
    all_ok = True
    for zp in a.zarr:
        if not os.path.exists(zp):
            print(f"[equiv] skip (missing): {zp}")
            continue
        z = zarr.open(str(zp), "r")
        d = z["data"]
        T = int(d["raw_point_cloud_len"].shape[0])
        frames = np.linspace(0, T - 1, min(int(a.frames), T)).astype(np.int64).tolist()
        ws_script = vp.compute_workspace(str(zp))
        ws_pkg = compute_workspace_from_zarr(str(zp))
        lo, hi = ws_pkg
        ws_same = bool(np.array_equal(np.asarray(ws_script[0], np.float64),
                                      np.asarray(lo, np.float64)) and
                       np.array_equal(np.asarray(ws_script[1], np.float64),
                                      np.asarray(hi, np.float64)))
        print(f"\n[equiv] {pathlib.Path(zp).name} frames={frames} grid={a.grid} "
              f"| workspace(script)==workspace(pkg): {ws_same}")
        print(f"        lo={np.round(lo, 6).tolist()} hi={np.round(hi, 6).tolist()}")
        rows = []
        for f in frames:
            raw = _raw_frame(d, f)
            xyz, rgb = raw[:, :3], raw[:, 3:6]
            s = vp.voxelize_numpy(xyz, rgb, lo, hi, float(a.grid))
            q, fld = pkg_voxelize(xyz, rgb, lo, hi, float(a.grid))
            if s is None or q is None:
                rows.append({"frame": int(f), "raw_N": int(len(raw)),
                             "skipped": True})
                print(f"  frame {f:4d}: skipped (no occupied cell)")
                all_ok = False
                continue
            cells_same = bool(np.array_equal(s["grid_index"], q["grid_index"]))
            counts_same = bool(np.array_equal(s["counts"], q["counts"]))
            dcent = float(np.abs(np.asarray(s["centers"], np.float64) -
                                 np.asarray(q["centers"], np.float64)).max())
            drgb = float(np.abs(np.asarray(s["avg_rgb"], np.float64) -
                                np.asarray(q["avg_rgb"], np.float64)).max())
            ok = cells_same and counts_same and dcent == 0.0 and drgb <= 1e-3
            all_ok &= ok
            rows.append({"frame": int(f), "raw_N": int(len(raw)),
                         "n_cells_script": int(len(s["grid_index"])),
                         "n_cells_pkg": int(len(q["grid_index"])),
                         "cells_identical": cells_same,
                         "counts_identical": counts_same,
                         "max_center_delta": dcent,
                         "max_rgb_delta": drgb,
                         "n_dropped_script": int(s["n_dropped_outside"]),
                         "n_dropped_pkg": int(q["n_dropped_outside"]),
                         "total_points_script": int(s["counts"].sum()),
                         "total_points_pkg": int(q["counts"].sum()),
                         "ok": bool(ok)})
            print(f"  frame {f:4d}: cells {len(s['grid_index'])}/{len(q['grid_index'])} "
                  f"identical={cells_same} counts_identical={counts_same} "
                  f"max|Δcenter|={dcent:.3g} max|Δrgb|={drgb:.3g} "
                  f"| dropped script/pkg={s['n_dropped_outside']}/{q['n_dropped_outside']} "
                  f"-> {'OK' if ok else 'DIFF'}")
        rep["per_zarr"].append({"zarr": str(zp), "frames": rows,
                               "workspace_identical": ws_same,
                               "lo": np.round(lo, 9).tolist(),
                               "hi": np.round(hi, 9).tolist()})
    rep["verdict"]["all_frames_equivalent"] = bool(all_ok)
    rep["verdict"]["elapsed_s"] = round(time.time() - t0, 1)
    print("\n===== 结论面板 [voxelize 调包等价性] =====")
    print(f"  占用格集合 / 每格计数 / 格心坐标 / 格均值 RGB  全部一致 = {all_ok}")
    print("  注: n_dropped_outside 口径不同(脚本=过 bbox 但越 grid; 包内=不在 workspace 内, 含越界) "
          "-> 仅作参考, 不影响调包")
    print(f"  报告 -> {a.out}")
    os.makedirs(os.path.dirname(str(a.out)), exist_ok=True)
    with open(str(a.out), "w") as fh:
        json.dump(rep, fh, indent=2, ensure_ascii=False)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
