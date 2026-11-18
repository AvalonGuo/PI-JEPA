#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""只读校准：pen raw_point_cloud(PC 系, vil_camera 相机系) <-> 场景世界系 的刚性映射。

背景:
  * pen fullpc zarr 的 raw_point_cloud 由
    gym_util/mujoco_point_cloud.py::generatePointCloud 产出:
      p_cam = ((u-cx)*d/f, (v-cy)*d/f, d)     # depth 相机系(深度沿 +z)
      p_pc  = p_cam @ c2w_r.T                 # c2w_r = cam_mat0 @ diag(1,-1,-1)
    (test/visualize_pc_unoccupied.py 已实测 668/668 往返一致, 故 PC 系 = 该 c2w 结果)
  * 但世界级 <camera> 的 body 为世界(0,0,0), 因此 c2w **无平移** => PC 系并非场景世界系。
  * 本脚本: 从 XML 读 vil_camera 真实外参 (cam_pos / cam_xmat), 推出 PC->世界 刚性映射,
    并用 zarr 点云验证桌面平面落在世界 z=0 且跨度 ±0.45。

用法:
    .venv/bin/python test/calib_pen_pc_map.py [--zarr <pen_fullpc.zarr>]
"""
import argparse
import pathlib
import sys

import numpy as np

PEN_XML = ("/root/3DP/third_party/rrl-dependencies/mj_envs/mj_envs/"
           "hand_manipulation_suite/assets/DAPG_pen.xml")
DEFAULT_ZARR = "/hy-tmp/3DP/data/adroit_pen_expert_fullpc.zarr"
CAM = "vil_camera"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zarr", default=DEFAULT_ZARR)
    a = ap.parse_args()

    from mujoco_py import load_model_from_path
    m = load_model_from_path(PEN_XML)
    ci = m.camera_name2id(CAM)
    cam_pos = np.asarray(m.cam_pos[ci], np.float64)         # 世界系相机位置
    fovy = float(m.cam_fovy[ci])
    print(f"[calib] vil_camera: pos={cam_pos} fovy={fovy}deg")

    # 代码路径用的 c2w_r = cam_mat0 @ diag(1,-1,-1); 静态相机 cam_mat0 即世界系朝向
    cam_mat0 = np.asarray(m.cam_mat0[ci], np.float64).reshape(3, 3)
    c2w_r = cam_mat0 @ np.diag([1.0, -1.0, -1.0])
    print(f"[calib] cam_mat0=\n{cam_mat0}")

    # p_pc = c2w_r @ p_cam  <=>  p_cam = c2w_r.T @ p_pc
    # 真世界: p_world = cam_pos + cam_mat0 @ p_cam_sensor, 其中传感器系
    # p_cam_sensor = ((u-cx)*d/f, (v-cy)*d/f, -d) = diag(1,1,-1) @ p_cam
    # => p_world = cam_pos + cam_mat0 @ diag(1,1,-1) @ c2w_r.T @ p_pc
    R_pc2world = cam_mat0 @ np.diag([1.0, -1.0, -1.0]) @ c2w_r.T
    t_pc2world = cam_pos
    print(f"[calib] PC->world R=\n{R_pc2world}\n[calib] PC->world t={t_pc2world}")

    # ---- 用 zarr 帧0 验证: 映射后桌面点应落在世界 z=0, xy 跨 ±0.45 ----
    import zarr
    d = zarr.open(a.zarr, mode="r")["data"]
    L = int(np.asarray(d["raw_point_cloud_len"][0]))
    pc = np.asarray(d["raw_point_cloud"][0], np.float32)[:L * 6].reshape(-1, 6)
    pw = pc[:, :3] @ R_pc2world.T + t_pc2world
    print(f"[calib] frame0 PC bbox x[{pc[:,0].min():+.3f},{pc[:,0].max():+.3f}] "
          f"y[{pc[:,1].min():+.3f},{pc[:,1].max():+.3f}] "
          f"z[{pc[:,2].min():+.3f},{pc[:,2].max():+.3f}]")
    print(f"[calib] frame0 world bbox x[{pw[:,0].min():+.3f},{pw[:,0].max():+.3f}] "
          f"y[{pw[:,1].min():+.3f},{pw[:,1].max():+.3f}] "
          f"z[{pw[:,2].min():+.3f},{pw[:,2].max():+.3f}]")
    # 桌面点: 世界 z 最接近 0 的一层 (取 |z|<0.02 的表层)
    table = pw[np.abs(pw[:, 2]) < 0.02]
    if len(table):
        print(f"[calib] world table-layer n={len(table)} "
              f"x[{table[:,0].min():+.3f},{table[:,0].max():+.3f}] "
              f"y[{table[:,1].min():+.3f},{table[:,1].max():+.3f}] "
              f"z[{table[:,2].min():+.4f},{table[:,2].max():+.4f}]  "
              f"(expect z~0, x/y within ±0.45)")
    # 手/笔点: z 高于桌面 (>0.02) 的世界高度分布
    above = pw[pw[:, 2] > 0.02]
    print(f"[calib] world above-table n={len(above)} z range "
          f"[{above[:,2].min():+.3f},{above[:,2].max():+.3f}] "
          f"(hand/pen expected 0~0.5)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
