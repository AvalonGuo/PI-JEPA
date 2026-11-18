#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""adroit_pen 的 dynsphere 候选格: 数据驱动绿区(优先) + 解析整桌网格(兜底)。

为什么 pen 需要单独处理(与 hammer/door 的 green_cells 差异):
  * hammer/door 的策略相机是**顶视** `top` -> 足迹可用闭式反投影
    (workspace_traffic_map.dataset_arm_footprint);
  * pen 的策略相机是 ~45° **斜视** `vil_camera`(DAPG_pen.xml 里没有 top 相机),
    必须逐点反投影: workspace_traffic_map.world_arm_footprint / green_cells_cam。
  两条路都沿用同一口径: 桌面 ±0.44 网格 / 手·物足迹膨胀 1 格 / 内缩 ±0.41。

两种候选来源(由是否拿得到 ID 数据集决定, 见 green_cells.load_dynsphere_cells):
  * "data" : ${DR3D_DATA_ROOT:-/hy-tmp/3DP/data}/adroit_pen_expert.zarr present -> real green cells
             (= green_cells.load_green_cells(task='pen'), 与 posgen/dynhand 共用);
  * "table": 数据集缺失(例如只拉了代码的机器)-> 整桌网格(不做手/物排除), 只保证
             suite 可跑: 球在候选格上是否可见/是否被手挡住, 仍由 dynsphere 自己的
             AABB 判据逐 episode 过滤(dynamic.py mode=frustum)。

缓存(仅数据路): dr3d/tmp/green_cells_pen.npz(+ .json 指纹); third_party 只读。

CLI(手动查看候选格): python dr3d/bench/green_cells_pen.py [--gs 0.02] [--show N]
"""
import pathlib
import sys

import numpy as np

DR3D = pathlib.Path(__file__).resolve().parents[1]
if str(DR3D) not in sys.path:
    sys.path.insert(0, str(DR3D))

# pen 的 ID 数据集与场景 XML(只读); 相机名以策略配置为准(见 cam_name())
import dataroot as _dr        # noqa: E402  数据集根目录(单一事实源)
import test_load as _TL       # noqa: E402  任务 base XML(单一事实源)
PEN_ZARR = _dr.adroit_expert_zarr("pen")
PEN_XML = _TL.BASE_XML["pen"]
CAM = "vil_camera"

# 与 workspace_traffic_map.green_cells 的默认口径保持一致(同一桌面网格原点)
PARAMS = dict(gs=0.02, table_half=0.44, inset=0.41)

def cam_name():
    """策略实际使用的相机名: 优先取真实策略配置(必须与观测一致), 否则用常量。"""
    try:
        import dynamic as _dyn
        cfg = _dyn.policy_cloud_config("pen")
        if cfg and cfg.get("cam_names"):
            return str(cfg["cam_names"][0])
    except Exception:
        pass
    return CAM


def table_grid(gs=None, table_half=None, inset=None):
    """整桌 (x,y) 候选格中心, 形状 (M,2); 与 green_cells 同一"网格原点 + 内缩"口径。

    不做手/物排除(dynsphere 允许球悬空/越过机械臂), 只保证格点落在桌面上、
    且 ±3cm 的球/方块不悬空。
    """
    gs = float(gs or PARAMS["gs"])
    half = float(table_half or PARAMS["table_half"])
    ins = float(inset or PARAMS["inset"])
    gg = np.arange(-half, half + 1e-9, gs)
    XX, YY = np.meshgrid(gg, gg)
    keep = (np.abs(XX) <= ins - 1e-9) & (np.abs(YY) <= ins - 1e-9)
    return np.stack([XX[keep], YY[keep]], axis=1)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gs", type=float, default=PARAMS["gs"])
    ap.add_argument("--show", type=int, default=5)
    a = ap.parse_args()
    cells = table_grid(gs=a.gs)
    print(f"[green_cells_pen] table_grid: n={len(cells)} cam={cam_name()} "
          f"zarr_exists={pathlib.Path(PEN_ZARR).exists()}")
    print(f"[green_cells_pen] x[{cells[:,0].min():+.3f},{cells[:,0].max():+.3f}] "
          f"y[{cells[:,1].min():+.3f},{cells[:,1].max():+.3f}]")
    for x, y in cells[:max(0, a.show)]:
        print(f"   x={x:+.3f} y={y:+.3f}")


if __name__ == "__main__":
    main()
