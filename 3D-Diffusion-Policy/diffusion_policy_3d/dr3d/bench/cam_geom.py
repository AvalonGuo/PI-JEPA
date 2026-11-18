#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""dr3d bench: 通用相机几何 —— eval 观测链(world <-> 深度像素 <-> 点云 crop)的最小数学。

只依赖 numpy, 被 dynamic.py(动态干扰物可见性判据) / workspace_traffic_map.py
(depth 反投影足迹) / test/verify_pen_dynsphere_vis.py(自检) 共用, 一处事实源。

所有公式与 gym_util/mujoco_point_cloud.py + mjpc_diffusion_wrapper.py 的实际代码
路径逐字对应, 并用**真实 sim.render 深度帧**实测验证(见
test/verify_pen_dynsphere_vis.py: 桌面平面反投影 -> world z=0 / xy 跨 ±0.45;
球体 AABB 判据 <-> 真实点云逐帧一致):

  相机(worldbody 内的静态相机):
      p_cam = cam_mat0.T @ (p_world - cam_pos) = (x_s, y_s, z_s)   # z_s<0 = 前方
      f     = img/(2*tan(fovy/2));  cx = cy = img/2
      u     = cx + f*x_s/(-z_s)      # sim.render 列索引(左->右)
      v     = cy - f*y_s/(-z_s)      # sim.render 行索引(上->下)
      Z     = -z_s                   # depthimg2Meters() 之后的深度(m)
  深度图/点云在 captureImage 里被 verticalFlip => 存储侧行索引 r 即
      p_o3d = ((u-cx)*Z/f, (r-cy)*Z/f, Z)          # cx = cy = img/2
  crop 系(use_point_crop 的 min/max_bound 作用处):
      cloud = pcT @ (cam_mat0 @ diag(1,-1,-1) @ p_o3d + cam_body_pos)  [*scale +offset]
  恒等式(★, 本文件核心): diag(1,-1,-1) @ p_o3d == p_cam =>
      cloud = pcT @ (p_world - cam_pos + cam_body_pos)
  对静态相机(cam_body_pos == 0, 如 adroit pen 的 vil_camera)即:

      cloud = p_world - cam_pos                                        (★★)

  pen 的 crop(min_bound z=-0.79, transform=None, cam_pos=(0,-0.8,0.8))因此等价于
  "world z > 0.01": 裁掉桌面与其上 1cm 以内的点(与实测 pen fullpc zarr 的
  raw_point_cloud z∈[-0.7167,-0.4769] <=> world z∈[0.083,0.323] 完全一致)。

注意: cam_pos 与 cam_body_pos 是两回事 —— 代码里 o3d 的平移用的是
body_pos[cam_bodyid](见 mujoco_point_cloud.py:172-182); 相机挂在 worldbody 上时
该值为 0, 所以"无平移"只对那类场景成立, 不能当通例。
"""
import numpy as np

IMG_DEFAULT = 84          # PointCloudGenerator(img_size=84) == mjpc wrapper 默认


def camera_names(model):
    """模型里所有相机名(顺序即 camera_name2id 的 id 顺序)。"""
    return [str(n) for n in model.camera_names]


def geometry(model, cam_name, img=IMG_DEFAULT):
    """只读提取一台相机的外参/内参 -> dict(R, pos, fovy, f, img, loc, body_pos)。

    cam_name 不存在时**报错**(绝不沿用 camera_name2id 的 -1: cam_pos[-1] 会静默
    取到最后一台相机, 例如 pen 场景里会误取 front 相机)。
    """
    cid = int(model.camera_name2id(str(cam_name)))
    if cid < 0:
        raise ValueError(f"camera {cam_name!r} not found in model; available: "
                         f"{camera_names(model)}")
    img = int(img)
    fovy = float(np.ravel(model.cam_fovy)[cid])
    bid = int(np.asarray(model.cam_bodyid).ravel()[cid])
    # NOTE: np.array(...) (copy) is required: model.cam_mat0/cam_pos are views into
    # the mujoco_py MjModel buffers, so a view would dangle as soon as the caller
    # drops the model (use-after-free -> segfault when the cached zarr path is used).
    return dict(
        cam=str(cam_name),
        R=np.array(model.cam_mat0[cid], dtype=float).reshape(3, 3),
        pos=np.array(model.cam_pos[cid], dtype=float).reshape(3),
        fovy=fovy,
        f=img / (2.0 * np.tan(np.radians(fovy) / 2.0)),
        img=img,
        loc=np.array([img / 2.0, img / 2.0]),
        body_id=bid,
        body_pos=np.array(model.body_pos[bid], dtype=float).reshape(3),
    )


def find_geometry(model, cam_names, img=IMG_DEFAULT):
    """按 cam_names 顺序取第一台存在的相机(多相机任务的相机挑选)。"""
    for nm in cam_names:
        if int(model.camera_name2id(str(nm))) >= 0:
            return geometry(model, nm, img=img)
    raise ValueError(f"none of cameras {list(cam_names)} in model "
                     f"(available: {camera_names(model)})")


def to_camera(points, g):
    """world (...,3) -> p_cam (...,3); z<0 表示在相机前方。"""
    p = np.asarray(points, dtype=float)
    return (p - g["pos"]) @ g["R"]                      # == R.T @ (p-pos) 行向量写法


def project(points, g):
    """world (...,3) -> (u, v, z_cam); u/v 为 sim.render 像素坐标(浮点)。"""
    pc = to_camera(points, g)
    z = pc[..., 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        inv = np.where(z < -1e-9, -1.0 / z, np.nan)     # 1/(-z) = 1/depth
        u = g["loc"][0] + g["f"] * pc[..., 0] * inv
        v = g["loc"][1] - g["f"] * pc[..., 1] * inv
    return u, v, z


def to_cloud(points, g, pc_transform=None, scale=None, offset=None):
    """world (...,3) -> crop 系点云(★恒等式, 见模块 docstring)。"""
    c = np.asarray(points, dtype=float) - g["pos"] + g["body_pos"]
    if pc_transform is not None:
        c = c @ np.asarray(pc_transform, dtype=float).T
    if scale is not None:
        c = c * np.asarray(scale, dtype=float)
    if offset is not None:
        c = c + np.asarray(offset, dtype=float)
    return c


def pixels_to_world(rows, cols, depth, g):
    """存储深度像素 -> world (...,3)。

    rows/cols 是**存储侧**索引(与 zarr/depth 图和点云同向: captureImage 已把
    sim.render 的深度帧上下翻转), depth 为 depthimg2Meters() 之后的米制深度。

    逐字对应 mujoco_point_cloud.generateCroppedPointCloud 的 o3d 路径:
        p_o3d = ((u-cx)*Z/f, (r-cy)*Z/f, Z)      # cx=cy=img/2 (cammat2o3d)
        p_cam = diag(1,-1,-1) @ p_o3d            # o3d 相机 -> MuJoCo 相机(见 docstring)
        world = cam_pos + cam_mat0 @ p_cam
    ★注意 y 的符号必须配 diag(1,-1,-1): p_cam.y = -(r-cy)*Z/f(r 向下增大 -> 相机 y 向上)。
    """
    rows = np.asarray(rows, dtype=float)
    cols = np.asarray(cols, dtype=float)
    d = np.asarray(depth, dtype=float)
    x_s = (cols - g["loc"][0]) * d / g["f"]
    y_s = -(rows - g["loc"][1]) * d / g["f"]
    p_cam = np.stack([x_s, y_s, -d], axis=-1)
    return p_cam @ g["R"].T + g["pos"]                   # == pos + R @ p_cam


def aabb_visible(xs, ys, h, r, g, min_bound, max_bound,
                 pc_transform=None, scale=None, offset=None, margin_px=1.0):
    """(xs,ys) 处 AABB(x±r, y±r, h±r) **整块可见** 的 bool 数组。

    逐角点同时查两项(与 eval 观测链严格一致):
      1. crop : pcT/scale/offset 作用后的 cloud 逐分量落在 (min_bound, max_bound)
                (wrapper 用严格不等式, 边界点会被剔除 -> 这里同样取严格);
      2. 视锥 : 角点投影落在 [margin_px, img-1-margin_px] 且 z_cam < 0(在前方)。
    两者都过才算"球完整进入策略点云/图像"(不被裁切)。
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n = int(xs.shape[0])
    S = np.array([[-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
                  [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]], float) * float(r)
    w = np.stack([xs[:, None] + S[None, :, 0],
                  ys[:, None] + S[None, :, 1],
                  np.full((n, 8), float(h)) + S[None, :, 2]], axis=2)   # (n,8,3)
    cloud = to_cloud(w.reshape(-1, 3), g, pc_transform, scale, offset)
    lo = np.asarray(min_bound, dtype=float)
    hi = np.asarray(max_bound, dtype=float)
    crop_ok = (np.all(cloud > lo, axis=1) & np.all(cloud < hi, axis=1))
    u, v, z = project(w.reshape(-1, 3), g)
    m = float(margin_px)
    hi_px = g["img"] - 1.0 - m
    frus_ok = (np.isfinite(u) & np.isfinite(v) & (z < 0.0) &
               (u >= m) & (u <= hi_px) & (v >= m) & (v <= hi_px))
    return (crop_ok & frus_ok).reshape(n, 8).all(axis=1)
