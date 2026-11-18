#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Offline verification for ``diffusion_policy_3d/cluster_distractor.py``.

Read-only: no sim, no training, no eval rollout. The injection hook sits right
before the FPS call in ``MetaWorldEnv.get_point_cloud``; this script replays that
exact step on the stored full clouds of a collected zarr
(``data/raw_point_cloud`` is already post-transform / post-crop, i.e. the same
array the wrapper feeds to FPS).

Checks
------
  T0  points=0        injector(cloud, fps_fn) == fps_fn(cloud)  (byte-identical)
  T1  enabled         fps_fn sees exactly len(cloud) + k*N rows, tail = injector rows
  T2  seed row        out[0] of the FPS result is unchanged (rows are *appended*)
  T3  shape           FPS output is (num_points, 6) for every N in --points
  T4  in-crop         every injected row over episodes 0..9 is inside the crop box
  T5  determinism     same (seed, episode) -> identical rows; other episode -> differ
  T6  radius law      r(0)=0, monotone, matches s_ref*sqrt(N/2pi)

Reported only (not asserted, not part of any training/eval metric): how many of
the injected rows happen to survive the FPS, and how many real rows changed.
For ``adroit_*`` tasks the stored zarr cloud is in the collector's frame, which
is not guaranteed to match the current eval chain, so the survival diagnostic is
marked ``n/a (frame-mismatch)``; T0-T3/T5/T6 are frame-independent, and the
adroit placement question is answered by ``cluster_distractor.py --check`` (R5).

Usage:
  .venv/bin/python test/verify_cluster_distractor.py
  .venv/bin/python test/verify_cluster_distractor.py --zarr <zarr> --out <json>
  .venv/bin/python test/verify_cluster_distractor.py --task adroit_hammer --num-points 512
"""
import argparse
import json
import pathlib
import sys

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[1]
_DP3 = _REPO / "3D-Diffusion-Policy"
for _p in (str(_DP3), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from diffusion_policy_3d.cluster_distractor import (  # noqa: E402
    ClusterInjector, load_spec, radius_for)

TASK = "metaworld_assembly"
_default_zarr = lambda task: f"/hy-tmp/3DP/data/{task}_expert.zarr"   # noqa: E731
DEFAULT_ZARR = _default_zarr(TASK)
NUM_POINTS = 1024



def _raw_frames(zarr_path, frames=3):
    """Post-transform / post-crop full clouds: exactly what the wrapper FPSes."""
    import zarr
    root = zarr.open(str(zarr_path), "r")
    d = root["data"] if "data" in root else root
    total = int(d["raw_point_cloud"].shape[0])
    idx = np.unique(np.linspace(0, total - 1, min(frames, total)).astype(int))
    out = []
    for f in idx:
        k = int(np.asarray(d["raw_point_cloud_len"][int(f)]))
        out.append(np.asarray(d["raw_point_cloud"][int(f)], np.float32).reshape(k, 6))
    return out


def _fps_fn(num_points):
    """Production sampler (``mjpc_wrapper.point_cloud_sampling``)."""
    def fn(pc):
        from diffusion_policy_3d.gym_util.mjpc_wrapper import point_cloud_sampling
        return point_cloud_sampling(np.asarray(pc, np.float32), num_points, "fps")
    return fn


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zarr", default=None)
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--out", default=None)
    ap.add_argument("--num-points", type=int, default=NUM_POINTS,
                    help="FPS budget of the task (1024 metaworld, 512 adroit)")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--points", type=int, nargs="+", default=[20, 50, 100, 200],
                    help="dose grid (candidate rows per cluster)")
    a = ap.parse_args(argv)
    a.zarr = a.zarr or _default_zarr(a.task)
    a.out = a.out or f"/hy-tmp/3DP/data/verify_cluster_{a.task}.json"
    grid = sorted({int(n) for n in a.points if int(n) > 0})

    spec = load_spec(a.task)
    spec_path = spec["spec_path"]
    adroit = str(spec.get("family", "metaworld")) == "adroit"
    clouds = _raw_frames(a.zarr)
    fps = _fps_fn(int(a.num_points))
    failures, report = [], {"spec": spec_path, "zarr": a.zarr,
                            "task": a.task, "num_points": int(a.num_points),
                            "cloud_rows": [int(len(c)) for c in clouds],
                            "delta_fps_cm": spec["delta_fps"],
                            "delta_raw_cm": spec["delta_raw"],
                            "family": str(spec.get("family", "metaworld"))}

    # -- T6 radius law --------------------------------------------------------
    r_tab = {n: radius_for(n, spec["delta_fps"]) for n in [0] + grid}
    _ns = list(r_tab)
    if r_tab[0] != 0.0 or any(r_tab[_ns[i]] > r_tab[_ns[i + 1]] for i in range(len(_ns) - 1)):
        failures.append("T6: r(N) is not monotone with r(0)=0")
    if not all(abs(r_tab[n] - spec["delta_fps"] * np.sqrt(n / (2 * np.pi))) < 1e-9
               for n in grid):
        failures.append("T6: r(N) deviates from the area law")
    report["radius_cm"] = {str(k): v for k, v in r_tab.items()}

    # -- T0 disabled injector -------------------------------------------------
    off = ClusterInjector(task=a.task, points=0, verbose=False)
    base = fps(clouds[0])
    if not np.array_equal(off(clouds[0], fps), base):
        failures.append("T0: points=0 changed the FPS output")
    if not np.array_equal(off(clouds[0], None), clouds[0]):
        failures.append("T0: points=0 changed the cloud")

    # -- T1/T2/T3 + diagnostics ----------------------------------------------
    diag = {}
    n_real = len(clouds[0])
    for n in grid:
        inj = ClusterInjector(task=a.task, points=n, seed=0, verbose=False,
                              num_points=int(a.num_points))
        rows = inj.begin(0)
        seen = {}

        def spy(pc, _rows=rows, _seen=seen):
            arr = np.asarray(pc)
            _seen["rows"] = arr.shape[0]
            _seen["tail"] = np.array_equal(arr[-_rows.shape[0]:],
                                          np.asarray(_rows, arr.dtype))
            _seen["head"] = np.array_equal(arr[:n_real], clouds[0])
            return fps(pc)

        out = inj(clouds[0], spy)
        if seen.get("rows") != n_real + n:
            failures.append(f"T1 N={n}: fps saw {seen.get('rows')} rows "
                            f"(expected {n_real + n})")
        if not seen.get("tail"):
            failures.append(f"T1 N={n}: injected rows are not the array tail")
        if not seen.get("head"):
            failures.append(f"T1 N={n}: real rows were modified")
        if out.shape != (int(a.num_points), 6):
            failures.append(f"T3 N={n}: FPS output {out.shape} != "
                            f"({int(a.num_points)}, 6)")
        if not np.array_equal(out[0], base[0]):
            failures.append(f"T2 N={n}: FPS seed row changed")

        # reported only: survivor count + real-row churn (not a metric).  On
        # adroit the zarr cloud sits in the collector frame, so how many injected
        # rows survive the FPS is not attributable -> reported as n/a.
        churn = int((~np.isclose(out[:, :3], base[:, :3], atol=1e-6)).all(1).sum())
        diag[n] = {"injected": int(n), "radius_cm": 100.0 * inj.radius,
                   "real_rows_changed": churn}
        if not adroit:
            surv = []
            for cloud in clouds:
                o = inj(cloud, fps)
                surv.append(int(np.isclose(o[:, None, :3], rows[None, :, :3],
                                           atol=1e-6).all(-1).any(0).sum()))
            diag[n]["survived_per_frame"] = surv

    # -- T4 in-crop over episodes (worst dose of the grid) --------------------
    n_worst = max(grid)
    inj = ClusterInjector(task=a.task, points=n_worst, seed=0, verbose=False,
                          num_points=int(a.num_points))
    lo, hi = inj.spec["crop_lo"], inj.spec["crop_hi"]
    worst = 0.0
    for ep in range(a.episodes):
        rows = inj.begin(ep)
        bad = np.any((rows[:, :3] <= lo) | (rows[:, :3] >= hi), axis=1)
        if bad.any():
            failures.append(f"T4 ep={ep}: {int(bad.sum())} rows outside the crop box")
        worst = max(worst, max(inj.history[-1]["center_shift_cm"]))
    report["max_center_shift_cm"] = worst

    # -- T5 determinism -------------------------------------------------------
    x = ClusterInjector(task=a.task, points=50, seed=7, verbose=False,
                        num_points=int(a.num_points))
    y = ClusterInjector(task=a.task, points=50, seed=7, verbose=False,
                        num_points=int(a.num_points))
    if not np.array_equal(x.begin(3), y.begin(3)):
        failures.append("T5: same (seed, episode) produced different rows")
    if np.array_equal(x.begin(3), x.begin(4)):
        failures.append("T5: different episodes produced identical rows")

    report["diagnostics"] = {str(k): v for k, v in diag.items()}
    report["failures"] = failures
    report["ok"] = not failures
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[verify] spec={spec_path}")
    print("[verify] r(N) cm: " + " ".join(f"N={k}->{v:.2f}" for k, v in r_tab.items()))
    for n, d in diag.items():
        print(f"[verify] N={n:3d} r={d['radius_cm']:4.2f}cm  "
              f"injected={d['injected']}  "
              f"survived={d.get('survived_per_frame', 'n/a (frame-mismatch)')} "
              f"(diagnostic only)  "
              f"real_rows_changed={d['real_rows_changed']}")
    print(f"[verify] max center shift = {worst:.2f} cm at N={n_worst} "
          f"over {a.episodes} episodes")
    if failures:
        print("[verify] FAILED\n  " + "\n  ".join(failures))
        print(f"[verify] report -> {a.out}")
        return 1
    print("[verify] OK (T0-T6)  report -> " + a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
