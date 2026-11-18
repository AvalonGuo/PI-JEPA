#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第一帧点云占用足迹可视化（只读、无仿真、无训练）。

用法:
    .venv/bin/python test/visualize_pc_unoccupied.py
        # 默认 zarr=/hy-tmp/3DP/data/adroit_pen_expert_fullpc.zarr
    .venv/bin/python test/visualize_pc_unoccupied.py --zarr <xxx>.zarr --out <out.png>

背景 = zarr 的 data/img[0]（与 eval 第一帧同源: 均为 vil_camera 的 sim.render,
无上下翻转）。把全数据集每帧 raw_point_cloud（裁剪后、FPS 前）投影回该图像平面,
取所有帧的并集作为"曾占据过"的像素。

workspace 区域 = 全数据集 raw_point_cloud 的并集 3D 包围盒（代码库约定:
diffusion_policy_3d/voxel/workspace.py::compute_workspace_from_zarr; pen 实测
lo=[-0.1386,0.3167,-0.7167] hi=[0.2606,0.7306,-0.4769]）。把包围盒 6 面按 ~1cm
采样反投影成像素并填充, 得到 workspace 足迹(约图像中央 52%)。

只把 workspace 足迹内"从未被占据"的像素用半透明绿色(alpha, 默认 0.4)覆盖在第一帧
上输出, 并额外输出二值掩码（白=workspace 内未占据）。

绿色语义(已对齐污染伪造, 见 3D-Diffusion-Policy/diffusion_policy_3d/voxel/workspace.py
与 jepa/forge.py):
    workspace 就是 Phase1 的 --workspace auto 决议结果 —— hammer 命中冻结表项
    (TASK_WORKSPACE["adroit_hammer"] = hammer 全帧 raw 点云并集包围盒), door/pen 用
    各自 fullpc zarr 的 compute_workspace_from_zarr(同一"逐帧 min/max 并集"约定)。
    Phase1 的 forge_voxel_rows 在此盒内以 2cm 栅格(Grid=GRID_SIZE_DEFAULT)锻造障碍物
    色点作为污染, 因此"workspace 内从未被任何帧点云占据的像素" = 锻造障碍物时可用的
    空单元 = 可放置区。本图绿色(半透明)即该集合, 白色二值掩码与之相同。

相机模型（vil_camera, 已在 /root/3DP 用深度往返+颜色匹配实测验证, 668/668 吻合）:
    fovy=45° -> f = H/(2*tan(fovy/2));  cx=W/2, cy=H/2
    cam_mat0 = R_x(45°);  c2w_r = cam_mat0 @ diag(1,-1,-1);  平移 t=(0,0,0)
    逆投影: p_cam = p_world @ c2w_r; u=f*x/z+cx, v=f*y/z+cy;  img 像素=(H-1-v, u)
    (深度图相对 img 做了 verticalFlip; 详见
     3D-Diffusion-Policy/diffusion_policy_3d/gym_util/mujoco_point_cloud.py)
"""
import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

DEFAULT_ZARR = "/hy-tmp/3DP/data/adroit_pen_expert_fullpc.zarr"
DEFAULT_OUT = "/hy-tmp/3DP/vis/pen_unoccupied_green.png"
DEFAULT_MASK_OUT = "/hy-tmp/3DP/vis/pen_unoccupied_mask.png"
FOVY_DEG = 45.0  # DAPG_pen.xml vil_camera 未写 fovy -> MuJoCo 默认 45°


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zarr", default=DEFAULT_ZARR, help="fullpc zarr 路径")
    p.add_argument("--out", default=DEFAULT_OUT, help="绿色覆盖图输出路径")
    p.add_argument("--mask-out", default=DEFAULT_MASK_OUT, help="二值掩码输出路径")
    p.add_argument("--limit", type=int, default=0, help="只处理前 N 帧 (0=全部)")
    p.add_argument("--no-mask", action="store_true", help="不输出二值掩码")
    p.add_argument("--no-selfcheck", action="store_true", help="跳过 depth[0] 往返自检")
    p.add_argument("--alpha", type=float, default=0.4, help="绿色半透明度 (0~1, 默认 0.4)")
    p.add_argument("--workspace", nargs=6, type=float, default=None,
                   metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"),
                   help="workspace 包围盒 lo/hi; 省略则用全数据集并集包围盒")
    p.add_argument("--margin", type=float, default=0.0,
                   help="workspace 包围盒各向外扩 (m, 默认 0)")
    p.add_argument("--outline", action="store_true",
                   help="在图上画 workspace 足迹轮廓")
    p.add_argument("--outline-color", default="blue",
                   choices=("red", "blue", "green", "yellow", "white"),
                   help="workspace 轮廓颜色 (默认 blue)")
    return p


def cam_constants(H, W):
    """adroit_pen vil_camera 投影常量 (已在 /root/3DP 实测验证)。

    返回 (f, cx, cy, c2w_r):
        f     = H/(2*tan(fovy/2)) = 84/(2*tan22.5°) ≈ 101.397
        c2w_r = R_x(45°) @ diag(1,-1,-1)
              = [[1,0,0],[0,-√2/2,√2/2],[0,-√2/2,-√2/2]]
    """
    f = H / (2.0 * np.tan(np.deg2rad(FOVY_DEG) / 2.0))
    cx, cy = W / 2.0, H / 2.0
    s = np.sqrt(2.0) / 2.0
    c2w_r = np.array([[1.0, 0.0, 0.0],
                      [0.0, -s, s],
                      [0.0, -s, -s]], dtype=np.float64)
    return f, cx, cy, c2w_r


def project_to_pixels(points, f, cx, cy, c2w_r, H, W):
    """世界坐标点云 -> 图像像素 (点云生成时的 FLIPPED 深度图坐标)。

    返回 (u, v, ok): u/v 为浮点像素坐标, ok 为 z>0 且在 [0,W)x[0,H) 内的有效掩码。
    """
    pcam = points @ c2w_r                        # 世界 -> 相机(o3d 惯例), 平移 t=0
    x, y, z = pcam[:, 0], pcam[:, 1], pcam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = f * x / z + cx
        v = f * y / z + cy
    ok = (z > 0.0) & np.isfinite(u) & np.isfinite(v)
    ok &= (u >= 0.0) & (u < W) & (v >= 0.0) & (v < H)
    return u, v, ok


def workspace_mask(lo, hi, f, cx, cy, c2w_r, H, W, step=0.01):
    """workspace 3D 包围盒 -> 图像足迹掩码 (bool, (H,W))。

    把包围盒 6 个表面按 step 采样(默认 1cm), 反投影成像素并填充。
    包围盒在相机前为凸体, 其透视投影仍为凸多边形, 故 binary_fill_holes 即可。
    注意: 边缘采样点可能落在 u/v=84 边界, np.rint 后需 clip 防越界。
    """
    from scipy import ndimage
    xs = np.arange(float(lo[0]), float(hi[0]) + 1e-9, step)
    ys = np.arange(float(lo[1]), float(hi[1]) + 1e-9, step)
    zs = np.arange(float(lo[2]), float(hi[2]) + 1e-9, step)
    pts = []
    for z in (float(lo[2]), float(hi[2])):
        gx, gy = np.meshgrid(xs, ys)
        pts.append(np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, z)])
    for x in (float(lo[0]), float(hi[0])):
        gy, gz = np.meshgrid(ys, zs)
        pts.append(np.c_[np.full(gy.size, x), gy.ravel(), gz.ravel()])
    for y in (float(lo[1]), float(hi[1])):
        gx, gz = np.meshgrid(xs, zs)
        pts.append(np.c_[gx.ravel(), np.full(gx.size, y), gz.ravel()])
    P = np.concatenate(pts, axis=0).astype(np.float64)
    u, v, ok = project_to_pixels(P, f, cx, cy, c2w_r, H, W)
    ui = np.clip(np.rint(u[ok]).astype(np.int64), 0, W - 1)
    vi = np.clip(np.rint(v[ok]).astype(np.int64), 0, H - 1)
    m = np.zeros((H, W), dtype=bool)
    m[H - 1 - vi, ui] = True
    return ndimage.binary_fill_holes(m)


def selfcheck(d, f, cx, cy, c2w_r):
    """用 depth[0] 正向重建, 验证点云生成模型与 zarr 一致。

    返回 (重建保留点数, raw_point_cloud_len[0])。
    """
    H, W = d["depth"].shape[1], d["depth"].shape[2]
    depth = np.asarray(d["depth"][0], np.float64)
    U, V = np.meshgrid(np.arange(W), np.arange(H))
    valid = depth < 50.0                          # far=90m 背景深度
    xo = (U - cx) * depth / f
    yo = (V - cy) * depth / f
    zo = depth
    pw = np.stack([xo, yo, zo], axis=-1) @ c2w_r.T  # 正向: 世界 = 相机 @ c2w_r.T
    keep = (valid
            & (pw[..., 2] >= -0.79)                 # pen min_bound z
            & (pw[..., 2] < 10.0)
            & (np.abs(pw[..., 0]) < 10.0)
            & (np.abs(pw[..., 1]) < 10.0))
    return int(keep.sum()), int(np.asarray(d["raw_point_cloud_len"][0]))


def main():
    a = build_parser().parse_args()
    import zarr
    from PIL import Image

    t0 = time.time()
    z = zarr.open(str(a.zarr), mode="r")
    d = z["data"]
    for key in ("raw_point_cloud", "raw_point_cloud_len", "img", "depth"):
        if key not in d:
            raise SystemExit(f"[pc-occ] zarr 缺少 data/{key}")

    bg = np.asarray(d["img"][0], np.uint8)
    if bg.ndim != 3 or bg.shape[-1] != 3:
        raise SystemExit(f"[pc-occ] 预期 img[0] 为 (H,W,3), 实际 {bg.shape}")
    H, W = bg.shape[:2]

    lens = np.asarray(d["raw_point_cloud_len"][:], np.int64)
    T = int(len(lens))
    if a.limit and a.limit > 0:
        T = min(T, int(a.limit))
    f, cx, cy, c2w_r = cam_constants(H, W)
    print(f"[pc-occ] zarr={a.zarr} frames={T} img0={bg.shape} "
          f"f={f:.4f} cx={cx:.2f} cy={cy:.2f}")
    return a, d, H, W, T, f, cx, cy, c2w_r, bg, t0


def run(a, d, H, W, T, f, cx, cy, c2w_r, bg, t0):
    from PIL import Image

    if not a.no_selfcheck:
        n_recon, n_ref = selfcheck(d, f, cx, cy, c2w_r)
        tag = "OK" if n_recon == n_ref else "MISMATCH"
        print(f"[pc-occ] 自检 depth[0] 正向重建 {n_recon} 点 == "
              f"raw_point_cloud_len[0]={n_ref}  [{tag}]")
        if n_recon != n_ref:
            raise SystemExit("[pc-occ] 相机模型与 zarr 不符, 终止")

    # ---- 1. 汇总全部原始点云 (裁剪后、FPS 前) + 并集包围盒 ----
    lens = np.asarray(d["raw_point_cloud_len"][:], np.int64)
    M = int(lens[:T].sum())
    P = np.empty((M, 3), dtype=np.float32)
    lo = np.full(3, np.inf, dtype=np.float64)
    hi = -lo.copy()
    idx = 0
    n_zero = 0
    for t in range(T):
        k = int(lens[t])
        if k <= 0:
            n_zero += 1
            continue
        raw = np.asarray(d["raw_point_cloud"][t], np.float32).reshape(k, 6)
        P[idx:idx + k] = raw[:, :3]
        lo = np.minimum(lo, raw[:, :3].min(axis=0))
        hi = np.maximum(hi, raw[:, :3].max(axis=0))
        idx += k
    P = P[:idx]
    print(f"[pc-occ] 汇总原始点 {idx}/{M} (空帧={n_zero})")

    # ---- 2. 反投影取并集 -> 占用足迹 ----
    occupied = np.zeros((H, W), dtype=bool)
    u, v, ok = project_to_pixels(P.astype(np.float64), f, cx, cy, c2w_r, H, W)
    ui = np.rint(u[ok]).astype(np.int64)
    vi = np.rint(v[ok]).astype(np.int64)
    row = H - 1 - vi                          # 深度图相对 img 做了 verticalFlip
    col = ui
    occupied[row, col] = True
    n_out = int((~ok).sum())
    print(f"[pc-occ] 有效投影 {int(ok.sum())}/{idx} (越界/背面 {n_out})")

    # ---- 3. workspace(Phase1 auto 决议结果) 内未占据区 = 可放置区 -> 半透明绿色 ----
    if a.workspace is not None:
        ws_lo = np.asarray(a.workspace[:3], np.float64)
        ws_hi = np.asarray(a.workspace[3:], np.float64)
    else:
        ws_lo, ws_hi = lo, hi
    if a.margin and a.margin > 0:
        ws_lo = ws_lo - float(a.margin)
        ws_hi = ws_hi + float(a.margin)
    ws = workspace_mask(ws_lo, ws_hi, f, cx, cy, c2w_r, H, W)
    unocc = ws & ~occupied          # 可放置区: workspace 内从未被数据集点云占据的像素
    print(f"[pc-occ] workspace lo={np.round(ws_lo, 4).tolist()} "
          f"hi={np.round(ws_hi, 4).tolist()} 足迹 {int(ws.sum())}px "
          f"({100.0 * ws.sum() / (H * W):.1f}%)")

    alpha = float(np.clip(a.alpha, 0.0, 1.0))
    vis = bg.astype(np.float64).copy()
    vis[unocc] = (1.0 - alpha) * vis[unocc] + alpha * np.array([0.0, 255.0, 0.0])
    vis = np.clip(vis, 0, 255).astype(np.uint8)
    if a.outline:
        from scipy import ndimage
        OUTLINE_RGB = {"red": (255, 0, 0), "blue": (0, 0, 255),
                       "green": (0, 255, 0), "yellow": (255, 255, 0),
                       "white": (255, 255, 255)}
        boundary = ws & ~ndimage.binary_erosion(ws)
        vis[boundary] = OUTLINE_RGB[str(a.outline_color)]
    os.makedirs(os.path.dirname(str(a.out)) or ".", exist_ok=True)
    Image.fromarray(vis).save(str(a.out))

    if not a.no_mask:
        os.makedirs(os.path.dirname(str(a.mask_out)) or ".", exist_ok=True)
        Image.fromarray((unocc.astype(np.uint8)) * 255).save(str(a.mask_out))

    occ_n = int(occupied.sum())
    print(f"[pc-occ] workspace 内被占据 {occ_n}/{H * W} "
          f"({100.0 * occ_n / (H * W):.1f}%)  可放置区(workspace内未占据) "
          f"{int(unocc.sum())} ({100.0 * unocc.sum() / (H * W):.1f}%)")
    print(f"[pc-occ] 绿色覆盖图 -> {a.out}")
    if not a.no_mask:
        print(f"[pc-occ] 二值掩码   -> {a.mask_out}")
    print(f"[pc-occ] 耗时 {time.time() - t0:.1f}s")

    rep = {"zarr": str(a.zarr), "frames": int(T), "img0_shape": list(bg.shape),
           "green_label": "可放置区 = Phase1 workspace(auto决议) 内未被数据集点云占据的空单元",
           "workspace_lo": np.round(ws_lo, 6).tolist(),
           "workspace_hi": np.round(ws_hi, 6).tolist(),
           "workspace_px": int(ws.sum()),
           "workspace_pct": round(100.0 * ws.sum() / (H * W), 2),
           "occupied_px": occ_n, "occupied_pct": round(100.0 * occ_n / (H * W), 2),
           "unoccupied_px": int(unocc.sum()),
           "unoccupied_pct": round(100.0 * unocc.sum() / (H * W), 2),
           "alpha": alpha, "total_points": int(idx), "out": str(a.out),
           "mask_out": str(a.mask_out)}
    rep_json = pathlib.Path(str(a.out)).with_suffix(".json")
    with open(rep_json, "w") as fh:
        json.dump(rep, fh, indent=2, ensure_ascii=False)
    print(f"[pc-occ] 报告 -> {rep_json}")
    return 0


if __name__ == "__main__":
    sys.exit(run(*main()))
