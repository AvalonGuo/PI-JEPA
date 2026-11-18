#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""metaworld assembly 的「绿区」候选位(供 posgen 随机放置干扰物)。

与 hammer/door 的 green_cells(workspace_traffic_map.py, top 相机)不同: assembly
的观测来自 corner2 相机 + 固定 point-cloud crop(MetaWorldEnv), 因此候选格 =
桌面格点中满足以下三条者(全部解析几何, 无渲染、无 sim 步进):

  1. in-crop : 世界点 -> corner2(cam_mat0 + wrapper 覆盖后的 cam_pos)
     -> pc_transform 后落在 MetaWorldEnv 的 [min_bound, max_bound] 内 ——
     与 eval 观测完全同一约定, 保证放置的 cube 一定进入策略点云;
  2. 远离机械臂工作区(|x|<=0.2 且 0.5<=y<=0.9, 实测值) + 安全边距;
  3. 远离物体/peg 邻域(不遮挡、不压住被操纵对象)。

缓存: dr3d/tmp/green_cells_assembly.npz (+ .json 指纹; 与 hammer/door 互不覆盖)。
非侵入: 只读 third_party 的模型 XML(取 corner2 相机矩阵), 产物全落 dr3d/tmp。

CLI:  python dr3d/bench/green_cells_assembly.py [--gs 0.03] [--force] [--show N]
"""
import hashlib
import json
import os
import pathlib
import sys

import numpy as np

DR3D = pathlib.Path(__file__).resolve().parents[1]
if str(DR3D) not in sys.path:
    sys.path.insert(0, str(DR3D))

# assembly 主 XML(只读; 取 corner2 相机矩阵)
ASSEMBLY_XML = ("/root/3DP/third_party/Metaworld/metaworld/envs/assets_v2/"
                "sawyer_xyz/sawyer_assembly_peg.xml")

# --- 与 MetaWorldEnv 一致的观测约定(env/metaworld/metaworld_wrapper.py) -------
CAM_POS_OVERRIDE = np.array([0.6, 0.295, 0.8], dtype=float)   # wrapper 覆盖值
CROP_MIN = np.array([-0.5, -1.5, -0.795], dtype=float)        # TASK_BOUDNS['default']
CROP_MAX = np.array([1.0, -0.4, 100.0], dtype=float)
_Z_PLACE = 0.03                                               # cube 中心高度
_ARM = dict(x=(-0.2, 0.2), y=(0.5, 0.9), margin=0.05)         # 机械臂工作区+边距
_OBJ = [(0.0, 0.6), (0.1, 0.8)]                               # RoundNut / peg
_OBJ_R = 0.16
PARAMS = dict(gs=0.03, x=(-0.67, 0.67), y=(0.23, 0.97))

CACHE = DR3D / "tmp" / "green_cells_assembly.npz"
CACHE_META = DR3D / "tmp" / "green_cells_assembly.json"


def _pc_transform():
    """MetaWorldEnv.pc_transform = Rx(61.4) @ Ry(-7)(scale=1, offset=0)。"""
    xa, ya = 61.4, -7.0
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(np.deg2rad(xa)), np.sin(np.deg2rad(xa))],
                   [0, -np.sin(np.deg2rad(xa)), np.cos(np.deg2rad(xa))]])
    Ry = np.array([[np.cos(np.deg2rad(ya)), 0, np.sin(np.deg2rad(ya))],
                   [0, 1, 0],
                   [-np.sin(np.deg2rad(ya)), 0, np.cos(np.deg2rad(ya))]])
    return Rx @ Ry


def _corner2_cam(xml_path=None):
    """只读加载 assembly 模型, 返回 corner2 的 (R_world2cam, cam_pos)。"""
    from mujoco_py import load_model_from_path
    m = load_model_from_path(str(xml_path or ASSEMBLY_XML))
    cid = int(m.camera_name2id("corner2"))
    Rcam = np.asarray(m.cam_mat0).reshape(-1, 3, 3)[cid].copy()
    return Rcam, CAM_POS_OVERRIDE.copy()


def _world_to_cloud(w, Rcam, cam_pos, pcT):
    w = np.asarray(w, float)
    return ((w - cam_pos) @ Rcam) @ pcT.T


def compute_cells(gs=None, xml_path=None):
    """(M,2) 世界 xy 候选格; 解析几何, 确定性。"""
    gs = float(gs or PARAMS["gs"])
    Rcam, cam_pos = _corner2_cam(xml_path)
    pcT = _pc_transform()
    xs = np.arange(PARAMS["x"][0], PARAMS["x"][1] + 1e-9, gs)
    ys = np.arange(PARAMS["y"][0], PARAMS["y"][1] + 1e-9, gs)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=1)
    world = np.concatenate([pts, np.full((len(pts), 1), _Z_PLACE)], axis=1)
    cloud = _world_to_cloud(world, Rcam, cam_pos, pcT)
    in_crop = (np.all(cloud >= CROP_MIN, axis=1) &
               np.all(cloud <= CROP_MAX, axis=1))
    ax1 = _ARM["x"][1]
    ay0, ay1 = _ARM["y"]
    mg = _ARM["margin"]
    in_arm = ((np.abs(pts[:, 0]) <= ax1 + mg) &
              (pts[:, 1] >= ay0 - mg) & (pts[:, 1] <= ay1 + mg))
    in_obj = np.zeros(len(pts), bool)
    for (ox, oy) in _OBJ:
        in_obj |= (np.hypot(pts[:, 0] - ox, pts[:, 1] - oy) < _OBJ_R)
    keep = in_crop & (~in_arm) & (~in_obj)
    return pts[keep].astype(np.float32)


def table_grid(gs=None):
    """整张 assembly 桌面的 (x,y) 网格(不做 crop/臂/物筛选); 供 dynsphere 用 ——
    球可悬空/越过机械臂, 候选只需是桌面点, 逐 episode 再按半径 r 做可见性过滤。"""
    gs = float(gs or PARAMS["gs"])
    xs = np.arange(PARAMS["x"][0], PARAMS["x"][1] + 1e-9, gs)
    ys = np.arange(PARAMS["y"][0], PARAMS["y"][1] + 1e-9, gs)
    XX, YY = np.meshgrid(xs, ys)
    return np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)


def _params_key(gs):
    p = dict(PARAMS)
    p["gs"] = float(gs)
    blob = json.dumps({"xml": ASSEMBLY_XML, "cam_pos": CAM_POS_OVERRIDE.tolist(),
                       "crop": [CROP_MIN.tolist(), CROP_MAX.tolist()],
                       "arm": _ARM, "obj": _OBJ, "obj_r": _OBJ_R,
                       "z": _Z_PLACE, "params": p}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_green_cells(force=False, verbose=True, gs=None):
    """读缓存(校验指纹)或就地计算并写缓存; 返回 (M,2) world xy。"""
    gs = float(gs or PARAMS["gs"])
    key = _params_key(gs)
    if not force and CACHE.exists() and CACHE_META.exists():
        try:
            meta = json.loads(CACHE_META.read_text(encoding="utf-8"))
            if meta.get("params_key") == key:
                cells = np.load(CACHE)["cells"]
                if verbose:
                    print(f"[green_cells_assembly] cache hit: {cells.shape[0]} "
                          f"cells <- {CACHE}")
                return cells
        except Exception as e:
            if verbose:
                print(f"[green_cells_assembly] cache unreadable ({e}); recomputing")
    cells = compute_cells(gs=gs)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    npz_tmp = CACHE.with_name(f".{CACHE.name}.tmp.{os.getpid()}")
    with open(npz_tmp, "wb") as f:
        np.savez_compressed(f, cells=cells)
    os.replace(str(npz_tmp), str(CACHE))
    meta_tmp = CACHE_META.with_name(f".{CACHE_META.name}.tmp.{os.getpid()}")
    meta_tmp.write_text(json.dumps(
        {"params_key": key, "n_cells": int(len(cells)), "gs": gs,
         "xml": ASSEMBLY_XML, "params": PARAMS}, indent=2), encoding="utf-8")
    os.replace(str(meta_tmp), str(CACHE_META))
    if verbose:
        print(f"[green_cells_assembly] computed {cells.shape[0]} cells -> {CACHE}")
    return cells


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gs", type=float, default=PARAMS["gs"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--show", type=int, default=5)
    a = ap.parse_args()
    cells = load_green_cells(force=a.force, gs=a.gs)
    print(f"[green_cells_assembly] n={len(cells)} "
          f"x[{cells[:,0].min():+.3f},{cells[:,0].max():+.3f}] "
          f"y[{cells[:,1].min():+.3f},{cells[:,1].max():+.3f}]")
    for x, y in cells[:max(0, a.show)]:
        print(f"   x={x:+.3f} y={y:+.3f}")


if __name__ == "__main__":
    main()
