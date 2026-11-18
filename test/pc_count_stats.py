#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""点云数量统计（只读、无仿真、无训练）。

    .venv/bin/python test/pc_count_stats.py --zarr /hy-tmp/3DP/data/adroit_pen_expert_fullpc.zarr
    .venv/bin/python test/pc_count_stats.py        # 自动发现 *_expert_fullpc.zarr + metaworld_*_expert.zarr

每个 zarr 输出一行:

    <name>  frames= NNNN | raw N min=.. p50=.. mean=.. max=.. (XX.X%) | 512云有效行 A~B | N>512 帧 K/N

字段说明:
  * raw N           = data/raw_point_cloud_len 的真实点数统计
  * (XX.X%)         = raw N 均值 / 深度图像素上限(84x84=7056); 即"平均有多少比例的深度
                      像素被反投影成有效点" (door 21.2% / hammer 5.6% / pen 8.2%)
  * 512云有效行     = data/point_cloud 每帧的非全零行数 (采样后应恒 = num_points 或 =N)
  * N>512 帧 K/N    = 走 torch3d FPS 分支(名额竞争)的帧数; 其余为零填充(pad)

JSON 只保存同一行里的这些数字, 无其它字段。
"""
import argparse
import glob
import json
import os
import pathlib
import sys

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(_REPO / "3D-Diffusion-Policy"), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_DATA_ROOT = "/hy-tmp/3DP/data"
DEFAULT_OUT = "/hy-tmp/3DP/vis/pc_stats/report.json"
PIXEL_CAP_FALLBACK = 84 * 84


def one(zarr_path, num_points):
    """-> (printed_line, json_dict) ; json_dict=None 表示该 zarr 不可统计。"""
    import zarr
    name = os.path.basename(str(zarr_path))
    if not os.path.exists(str(zarr_path)):
        return "{}  MISSING".format(name), None
    z = zarr.open(str(zarr_path), "r")
    d = z["data"]
    if "raw_point_cloud_len" not in d:
        return "{}  -> no raw_point_cloud_len".format(name), None
    n = np.asarray(d["raw_point_cloud_len"][:], np.int64)
    cap = int(np.prod(d["depth"].shape[1:])) if "depth" in d else PIXEL_CAP_FALLBACK
    if "point_cloud" in d:
        pc = np.asarray(d["point_cloud"][:], np.float32)
        rows = np.any(pc != 0, axis=-1).sum(-1)
        rows_min, rows_max = int(rows.min()), int(rows.max())
    else:
        rows_min = rows_max = None
    n_gt = int((n > int(num_points)).sum())
    pct = 100.0 * float(n.mean()) / cap
    line = ("{:42s} frames={:5d} | raw N min={} p50={} mean={:.0f} max={} ({:.1f}%) | "
            "{}云有效行 {}~{} | N>{} 帧 {}/{}").format(
        name, len(n), int(n.min()), int(np.percentile(n, 50)), n.mean(), int(n.max()), pct,
        int(num_points), rows_min, rows_max, int(num_points), n_gt, len(n))
    rec = {"zarr": name, "frames": int(len(n)),
           "raw_n_min": int(n.min()), "raw_n_p50": int(np.percentile(n, 50)),
           "raw_n_mean": round(float(n.mean()), 1), "raw_n_max": int(n.max()),
           "raw_n_mean_pct": round(pct, 1),
           "pc_rows_min": rows_min, "pc_rows_max": rows_max,
           "n_gt_num_points": n_gt, "num_points": int(num_points)}
    return line, rec


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zarr", nargs="*", default=None,
                   help="fullpc zarr (可多个); 省略则自动发现 /hy-tmp/3DP/data/*_expert_fullpc.zarr "
                        "+ metaworld_*_expert.zarr")
    p.add_argument("--num-points", type=int, default=512,
                   help="分支判定阈值 / 采样后的行数 (默认 512)")
    p.add_argument("--out", default=DEFAULT_OUT, help="json 输出路径")
    p.add_argument("--no-json", action="store_true", help="只打印, 不写 json")
    a = p.parse_args()

    zarrs = a.zarr if a.zarr else sorted(
        glob.glob(os.path.join(DEFAULT_DATA_ROOT, "*_expert_fullpc.zarr")) +
        glob.glob(os.path.join(DEFAULT_DATA_ROOT, "metaworld_*_expert.zarr")))
    if not zarrs:
        raise SystemExit("[pc-stats] 没有可统计的 zarr (给 --zarr 或检查 {})".format(DEFAULT_DATA_ROOT))
    recs = []
    for zp in zarrs:
        line, rec = one(zp, a.num_points)
        print(line)
        if rec is not None:
            recs.append(rec)
    if not a.no_json:
        os.makedirs(os.path.dirname(str(a.out)), exist_ok=True)
        with open(str(a.out), "w") as f:
            json.dump(recs, f, indent=2, ensure_ascii=False)
        print("-> {}".format(a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
