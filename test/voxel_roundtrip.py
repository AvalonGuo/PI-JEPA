#!/usr/bin/env python
"""
Bidirectional voxelizer <-> renderer validation (geometry only).

Subcommands (user runs them explicitly; agent does NOT run training / eval /
data-gen):

  roundtrip  geometry lossless check: voxelize(pc) -> reconstruct(field)
             on ID frames (pure numpy, no DP3).
  obstacle   geometry on clean vs real-cube polluted twin frames (fullpc_o):
             roundtrip coverage / recovered-cell counts of the obstacle half.

NOTE: the former DP3-oracle subcommands (`action`, `order`) and the action-gap
part of `obstacle` were removed together with the archived G/F adversarial line
and its frozen-DP3 oracle (`distill/oracle.py`). This tool no longer needs a
checkpoint.

Outputs: JSON summary written under --out (default /hy-tmp/3DP/voxel_out).
"""
import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import zarr

_REPO = pathlib.Path(__file__).resolve().parents[1]           # /root/3DP
_DP3_DIR = _REPO / "3D-Diffusion-Policy"
for _p in (str(_DP3_DIR), str(_REPO), str(pathlib.Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from diffusion_policy_3d.voxel import (  # noqa: E402
    voxelize, reconstruct, resolve_workspace,
)

DEFAULT_FULLPC = "/hy-tmp/3DP/data/adroit_hammer_expert_fullpc.zarr"
DEFAULT_FULLPC_O = "/hy-tmp/3DP/data/adroit_hammer_expert_fullpc_o.zarr"
DEFAULT_OUT = "/hy-tmp/3DP/voxel_out"


# ---------------------------------------------------------------------------
# zarr access
# ---------------------------------------------------------------------------
def _open_zarr(path):
    root = zarr.open(path, mode="r")
    data = root["data"]
    ends = np.asarray(root["meta"]["episode_ends"], dtype=np.int64)
    return data, ends


def _raw_frame(data, f):
    arr = np.asarray(data["raw_point_cloud"][f])
    return arr.reshape(-1, 6).astype(np.float32)


def _sample_indices(n, start, count, stride):
    end = min(n, start + count * stride)
    idx = np.arange(start, end, stride, dtype=np.int64)
    if len(idx) == 0:
        raise SystemExit(f"no frames selected (start={start} count={count} "
                         f"stride={stride}, n={n})")
    return idx


def _agg(vals):
    vals = list(vals)
    if not vals:
        return {"n": 0}
    v = np.asarray(vals, dtype=np.float64)
    return {"n": int(len(v)), "mean": float(v.mean()), "std": float(v.std()),
            "min": float(v.min()), "max": float(v.max())}


def _dump(out_dir, name, obj):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return path


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def _roundtrip_stats(pc, grid_size, cap):
    """voxelize -> reconstruct; returns per-frame geometry stats dict."""
    ws = resolve_workspace(mode="hammer")     # historical hammer box (unchanged)
    fld = voxelize(pc, lo=ws.lo, hi=ws.hi,
                   grid_size=grid_size, cap=cap)
    rec = reconstruct(fld)
    n_in = fld.n_points
    n_out = int(fld.n_dropped_outside)
    keep_per_cell = np.minimum(fld.counts, np.int64(cap))
    n_kept = int(keep_per_cell.sum())
    n_trunc_cells = int((fld.counts > cap).sum())
    coverage = n_kept / max(1, n_in)
    exact = None
    maxdiff = None
    if n_in > 0 and n_out == 0 and n_kept == n_in:
        # multiset identity: rec rows are a subset of the original rows
        order = np.lexsort((pc[:, 2], pc[:, 1], pc[:, 0]))
        a = pc[order]
        b = rec[np.lexsort((rec[:, 2], rec[:, 1], rec[:, 0]))]
        d = np.abs(a - b).max() if len(a) == len(b) else np.inf
        exact = bool(np.allclose(a, b, atol=1e-4))
        maxdiff = float(d)
    return {"n_raw": int(len(pc)), "n_out": n_out, "n_in": n_in,
            "n_rec": int(len(rec)), "n_kept": n_kept,
            "n_trunc_cells": n_trunc_cells,
            "coverage": float(coverage), "exact": exact, "maxdiff": maxdiff,
            "n_cells": int(fld.n_cells)}


def cmd_roundtrip(a):
    data, ends = _open_zarr(a.zarr)
    n_frames = int(len(data["raw_point_cloud"]))
    idx = _sample_indices(n_frames, a.start, a.frames, a.stride)
    t0 = time.time()
    rows = []
    for f in idx:
        pc = _raw_frame(data, int(f))
        rows.append({"frame": int(f),
                     **_roundtrip_stats(pc, a.grid, a.cap)})
    out = {"mode": "roundtrip", "zarr": a.zarr,
           "grid_size": a.grid, "cap": a.cap, "frames": idx.tolist(),
           "agg": {"coverage": _agg(r["coverage"] for r in rows),
                   "n_out": _agg(r["n_out"] for r in rows),
                   "n_trunc_cells": _agg(r["n_trunc_cells"] for r in rows),
                   "exact": _agg(r["exact"] for r in rows if r["exact"] is not None)},
           "rows": rows, "elapsed_s": round(time.time() - t0, 2)}
    path = _dump(a.out, "voxel_roundtrip.json", out)
    cov = out["agg"]["coverage"]
    print(f"[roundtrip] {len(idx)} frames, grid={a.grid}, cap={a.cap}")
    print(f"[roundtrip] coverage mean={cov['mean']:.4f} "
          f"min={cov['min']:.4f}  (target >0.95)")
    print(f"[roundtrip] exact-lossless frames = "
          f"{out['agg']['exact']['n'] if out['agg']['exact']['n'] else 0}")
    print(f"[roundtrip] saved -> {path}")


def cmd_obstacle(a):
    """Geometry-only: clean vs real-cube polluted twin frames (fullpc_o).

    Checks that the synthesized obstacle half round-trips through the voxel
    board as expected (coverage + recovered-cell delta). Needs no checkpoint.
    """
    data, ends = _open_zarr(a.zarr)
    n_eps = len(ends)
    assert n_eps % 2 == 0, "obstacle zarr must be clean+obstacle halves"
    clean_len = int(ends[n_eps // 2 - 1])
    idx = _sample_indices(int(clean_len), a.start, a.frames, a.stride)
    t0 = time.time()
    rows = []
    for f in idx:
        f = int(f)
        pc_c = _raw_frame(data, f)
        pc_p = _raw_frame(data, f + clean_len)
        n_add_gt = int(len(pc_p) - len(pc_c))
        rows.append({"frame": f, "twin_frame": f + clean_len,
                     "n_add_gt": n_add_gt,
                     "clean": _roundtrip_stats(pc_c, a.grid, a.cap),
                     "poll": _roundtrip_stats(pc_p, a.grid, a.cap)})
    agg = {
        "n_add_gt": _agg(r["n_add_gt"] for r in rows),
        "poll_coverage": _agg(r["poll"]["coverage"] for r in rows),
        "n_add_rt": _agg(r["poll"]["n_rec"] - r["clean"]["n_rec"] for r in rows),
    }
    out = {"mode": "obstacle", "zarr": a.zarr,
           "clean_len": clean_len, "grid_size": a.grid, "cap": a.cap,
           "agg": agg, "rows": rows, "elapsed_s": round(time.time() - t0, 2)}
    path = _dump(a.out, "voxel_obstacle.json", out)
    print(f"[obstacle] {len(idx)} twin frames; n_add_gt mean="
          f"{agg['n_add_gt']['mean']:.0f} (expect 47)")
    print(f"[obstacle] poll roundtrip coverage mean="
          f"{agg['poll_coverage']['mean']:.4f}")
    print(f"[obstacle] recovered-cell delta n_add_rt mean="
          f"{agg['n_add_rt']['mean']:.4f}")
    print(f"[obstacle] saved -> {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("roundtrip", help="geometry roundtrip on ID frames")
    pr.add_argument("--zarr", default=DEFAULT_FULLPC)
    pr.add_argument("--grid", type=float, default=0.02)
    pr.add_argument("--cap", type=int, default=16)
    pr.add_argument("--start", type=int, default=0)
    pr.add_argument("--frames", type=int, default=20)
    pr.add_argument("--stride", type=int, default=10)
    pr.add_argument("--out", default=DEFAULT_OUT)
    pr.set_defaults(func=cmd_roundtrip)

    po = sub.add_parser("obstacle",
                        help="clean vs real-cube twin frames (geometry only)")
    po.add_argument("--zarr", default=DEFAULT_FULLPC_O)
    po.add_argument("--grid", type=float, default=0.02)
    po.add_argument("--cap", type=int, default=16)
    po.add_argument("--start", type=int, default=0)
    po.add_argument("--frames", type=int, default=10)
    po.add_argument("--stride", type=int, default=50)
    po.add_argument("--out", default=DEFAULT_OUT)
    po.set_defaults(func=cmd_obstacle)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
