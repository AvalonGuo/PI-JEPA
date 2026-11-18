#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
workspace_traffic_map.py
========================
基于 adroit ID 数据(全点云+逐点语义)统计 "workspace 轨迹经过" 并输出:
  * 红 = hand/hammer 点实际落到桌面格(2cm) 本身, 默认不外扩(--spread=0)
  * 绿 = 除红色外的全部桌面格(整桌 ±0.44 网格)
nail/nail_board 仅用于候选挑选时过滤(离静态物 > --static-margin), 图像保持红/绿两色。

坐标系: 点云为 PC 系; 输出图渲染在 hammer 世界系顶视首帧(512)。
PC<->world(xy) 用静态 nail_board 质心 + 已知 obstacle 对应点估计 R/t(只读, 见
estimate_xy_map), 因此绿/红格子会被放到正确桌面上。

用法(本文件位于 diffusion_policy_3d/dr3d/bench/):
  python -m diffusion_policy_3d.dr3d.bench.workspace_traffic_map \\
      --zarr ${DR3D_DATA_ROOT:-/hy-tmp/3DP/data}/adroit_hammer_expert.zarr \\
      --out  /hy-tmp/3DP/distractor_placement/workspace_traffic_firstframe.png

输出: 两张 PNG —— <out 去掉.png>_orig.png(无 mask 原图) 与 <...>_mask.png(掩码叠加版)。
RGB 无 crop; point-cloud crop(use_point_crop) 只影响点云不影响本图。

方案1(--rect, 无标签禁放区):
  不去语义分割: 红 = 手/锤足迹(禁放), 绿 = 桌面 workspace 的其余部分(可放置)。
  默认 --rect-shape footprint: 直接读 zarr 的 depth(84x84) + top 相机模型, 取"高于桌面
  --arm-height(默认 5cm)"的像素反投影到 world xy 打格 —— 实测 616 格, 与语义标签
  hand/hammer 的 614 格几乎一致, 故**不需要 raw_point_cloud_label**;
  --rect-shape rect 则退化为 dataset-union bbox 外接矩形(实测 910 格, 多框 296 格, +48%)。
  PC<->world 用硬编码常量(MAP_R/MAP_T, 实测 R=-I, t≈0, 残差 0.2cm), world->像素 用
  渲染标定的单应 H。桌面 ±0.45 与 top 相机视野 ±0.4556m 一一对应(桌面占画面 98.8%,
  每侧仅多 5.6mm 地面边), 故"绿"即渲染图里桌面的剩余部分。
  用法:
    python -m diffusion_policy_3d.dr3d.bench.workspace_traffic_map --rect --size 512 \\
        --out /hy-tmp/3DP/distractor_placement/workspace_rect_firstframe.png
  输出: <...>_orig.png 与 <...>_rect.png(红=禁放足迹, 绿=可放置格, 橙框=外接矩形参考)。
"""
import argparse
import os
import pathlib
import sys

import numpy as np

# 本目录非包(平铺模块): 保证 `import cam_geom` 在本文件被 importlib 按路径加载时也能用
_BENCH = pathlib.Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))
import cam_geom  # noqa: E402  通用相机几何(已用真实渲染验证, 见 test/verify_pen_dynsphere_vis.py)

# 本文件位于 .../diffusion_policy_3d/dr3d/bench/ ⇒ parents[1] 即 dr3d 目录
DR3D_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(DR3D_DIR) not in sys.path:
    sys.path.insert(0, str(DR3D_DIR))
import dataroot as _dataroot  # noqa: E402  数据集根目录(单一事实源)
import test_load as _TL_MOD   # noqa: E402  任务 base XML(单一事实源)

DEFAULT_ZARR = _dataroot.adroit_expert_zarr("hammer")
# 与 dr3d/verify_render.py 相同: 同一 DAPG_hammer 场景(无 obstacle)
DEFAULT_HAMMER = _TL_MOD.BASE_XML["hammer"]
DEFAULT_OUT = "/hy-tmp/3DP/distractor_placement/workspace_traffic_firstframe.png"

_TL = None


def _test_load():
    global _TL
    if _TL is None:
        import sys
        if str(DR3D_DIR) not in sys.path:
            sys.path.insert(0, str(DR3D_DIR))
        import test_load as tl
        _TL = tl
    return _TL


def compose_distractor_xml(base_xml, obj_name, xml_path, pos):
    """把一个 dr3d distractor(自由体+网格+材质/纹理) 合入任意 hammer xml 文本,
    放置 x/y 用 pos, 自动着地 z = -min_z + 2mm + OBJ_LIFT(mug +1cm)。"""
    import re as _re
    TL = _test_load()
    obj_txt = pathlib.Path(xml_path).read_text(encoding="utf-8", errors="ignore")
    obj_txt = TL.abs_mesh_paths(obj_txt, pathlib.Path(xml_path).parent)
    assets_in, world_in = TL.extract_assets_and_worldbody(obj_txt)
    # 多物体同场景需要唯一 name(robosuite 模板里 body 'object'/bottom_site 等会重名)
    world_in = _re.sub(r'name="[^"]*"',
                       lambda m: 'name="dr_%s_%s"' % (obj_name, m.group(0)[6:-1]),
                       world_in)
    # hammer 默认 class mesh scale=0.001; 给注入 mesh 补显式 scale(无则 1 1 1)
    def _mesh_scale(mm):
        t = mm.group(0)
        m = _re.search(r'scale="([0-9.eE+-]+) ([0-9.eE+-]+) ([0-9.eE+-]+)"', t)
        base = [float(x) for x in m.groups()] if m else [1.0, 1.0, 1.0]
        ns = ' '.join('%.6g' % b for b in base)
        if m:
            t = t[:m.start()] + 'scale="%s"' % ns + t[m.end():]
        else:
            t = t[:-2] + ' scale="%s" />' % ns
        return t
    assets_in = _re.sub(r'<mesh\b[^>]*/>', _mesh_scale, assets_in)
    ground = TL.object_ground_z(str(xml_path))
    xy = pos.split()
    z = -ground + 0.002 + float(TL.OBJ_LIFT.get(obj_name, 0.0))
    xml = base_xml
    if "<asset>" in xml:
        xml = xml.replace("</asset>", assets_in + "</asset>", 1)
    else:
        m = _re.search(r"<mujoco[^>]*>", xml)
        xml = xml[:m.end()] + "<asset>" + assets_in + "</asset>" + xml[m.end():]
    body = (f'<body name="dr_{obj_name}" pos="{xy[0]} {xy[1]} {z:.4f}" quat="1 0 0 0">'
            f'<freejoint name="dr_{obj_name}_free"/>'
            '<inertial pos="0 0 0.02" mass="0.1" '
            'diaginertia="1e-4 1e-4 1e-4"/>' + world_in + "</body>")
    assert "</worldbody>" in xml
    xml = xml.replace("</worldbody>", body + "</worldbody>", 1)
    return xml


def render_with_verify_camera(xml_path, size):
    """相机逻辑与 dr3d/verify_render.py 一致:
       优先 camera 'top', 否则 cams[0]; load->sim.forward()->sim.render(size,size)."""
    from mujoco_py import load_model_from_path
    import mujoco_py
    model = load_model_from_path(xml_path)
    sim = mujoco_py.MjSim(model)
    sim.forward()
    cams = list(getattr(model, "camera_names", []) or [])
    cam = "top" if "top" in cams else (cams[0] if cams else None)
    if cam is not None and cam not in cams:
        cam = cams[0] if cams else None
    img = np.asarray(sim.render(size, size, camera_name=cam))
    if img.shape[-1] != 3:
        img = img.transpose(1, 2, 0)
    return img, model

LABEL_HAND, LABEL_HAMMER = 1, 2
LABEL_BOARD, LABEL_NAIL = 3, 4

# 已知对应点: 真实 obstacle 世界坐标 <-> 其点云(PC)坐标(合成方块实测)
OBS_WORLD = np.array([0.22, -0.15], dtype=float)
OBS_PC = np.array([-0.220, 0.157], dtype=float)

# ---------------------------------------------------------------------------
# 方案1(无标签): dataset-union bbox 矩形禁放区所需常量 / 工具
#   PC<->world xy 用硬编码常量, 不再依赖 raw_point_cloud_label:
#   estimate_xy_map 实测 board 对齐 + obstacle 对应点残差 0.2cm 时 R=-I, t=(0.0009,0.0093)
# ---------------------------------------------------------------------------
MAP_R = np.array([[-1.0, 0.0], [0.0, -1.0]])
MAP_T = np.array([0.0009, 0.0093])


def _render_first_frame(a, scene_xml):
    """渲染首帧(可选注入 distractor 场景), 返回 RGB 图。"""
    scene_tmp = None
    if scene_xml is not None:
        scene_tmp = os.path.join(os.path.dirname(a.hammer),
                                 f"_scene_tmp_{os.getpid()}.xml")
        pathlib.Path(scene_tmp).write_text(scene_xml, encoding="utf-8")
    scene_path = scene_tmp or a.hammer
    try:
        img, _ = render_with_verify_camera(scene_path, a.size)
    finally:
        if scene_tmp is not None:
            os.remove(scene_tmp)
    return img


def dataset_union_xy_bbox(zarr_path, step=1):
    """ID 全数据集 raw_point_cloud 的 xy dataset-union bbox (PC 系, 无需语义标签)。

    等价于 test/voxelize_pointcloud.py::compute_workspace 的 xy 部分(其实测值已硬编码
    为 WORKSPACE_DEFAULT)。返回 (x0, y0, x1, y1)。
    """
    import zarr
    d = zarr.open(zarr_path, mode="r")["data"]
    T = int(d["raw_point_cloud_len"].shape[0])
    lo = np.array([np.inf, np.inf])
    hi = -lo.copy()
    for i in range(0, T, step):
        L = int(d["raw_point_cloud_len"][i])
        if L <= 0:
            continue
        xyz = np.asarray(d["raw_point_cloud"][i])[:L * 6].reshape(-1, 6)[:, :2]
        lo = np.minimum(lo, xyz.min(axis=0))
        hi = np.maximum(hi, xyz.max(axis=0))
    if not np.isfinite(lo).all():
        raise SystemExit("dataset_union_xy_bbox: 没有可用点云")
    return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])


def _top_camera(hammer_path, depth_shape):
    """读 'top' 相机: 由 fovy/高度得到内参 f 与相机在桌面上方的高度 cam_z。
    (top: pos=(0,0,cam_z) 垂直俯视; 桌面 z=0, 故某点的 depth = cam_z - 其高度。)"""
    from mujoco_py import load_model_from_path
    m = load_model_from_path(hammer_path)
    ci = m.camera_name2id("top")
    fovy = float(m.cam_fovy[ci])
    cam_z = float(m.cam_pos[ci][2])
    h, w = depth_shape
    f = float(w) / (2.0 * np.tan(np.radians(fovy) / 2.0))
    return f, cam_z


def dataset_arm_footprint(zarr_path, hammer_path, gs, table_half,
                          min_height=0.05, step=1):
    """免语义标签的「手/锤 footprint」(2cm 桌面格, 返回 (N,N) bool)。

    逐帧取 depth(84x84) 中"高于桌面"的像素(depth <= cam_z - min_height), 按 top 相机
    模型反投影到 world xy 再打到桌面格。已实测: 该判据得到 616 格, 与语义标签
    hand(1)/hammer(2) 的 footprint 614 格几乎一致, world bbox 也完全相同 —— 因此
    方案1 的矩形红区可以换成这个精确足迹, 且**不需要 raw_point_cloud_label**。

    注意: zarr 里的 depth 相对 sim.render 上下翻转(mujoco_point_cloud.captureImage 里
    做了 verticalFlip), 故此处先用 v_unflip = H-1-v 还原。
    """
    import zarr
    D = zarr.open(zarr_path, mode="r")["data"]["depth"]
    T, H, W = D.shape if D.ndim == 3 else (len(D), *np.asarray(D[0]).shape)
    f, cam_z = _top_camera(hammer_path, (H, W))
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    vv = (H - 1) - vv                       # 反上下翻转 -> sim.render 帧
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    N = len(np.arange(-table_half, table_half + 1e-9, gs))
    grid = np.zeros(N * N, bool)
    zmax = cam_z - min_height               # depth 阈值(= 高于 min_height 的点)
    for i in range(0, T, step):
        dep = np.asarray(D[i], dtype=np.float64)
        m = dep <= zmax
        if not m.any():
            continue
        d = dep[m]
        wx = (cx - uu[m]) * d / f           # top 相机成像模型 -> world xy
        wy = (cy - vv[m]) * d / f
        ix = np.floor((wx + table_half) / gs).astype(int)
        iy = np.floor((wy + table_half) / gs).astype(int)
        ok = (ix >= 0) & (ix < N) & (iy >= 0) & (iy < N)
        grid[iy[ok] * N + ix[ok]] = True
    return grid.reshape(N, N)


def green_cells(zarr_path=None, hammer_path=None, gs=0.02, table_half=0.44,
                arm_height=0.05, margin_cells=1, inset=0.41):
    """返回「绿区」(可放置) 桌面格中心的 world xy, 形状 (M, 2)。

    绿区 = 桌面 ±table_half 去掉手/锤足迹(dataset_arm_footprint, depth 判据),
    再把足迹膨胀 margin_cells 格作安全边距(避免 cube 被手/锤擦到), 并内缩到
    |x|,|y| <= inset(保证 ±3cm 的 cube 不悬空)。与 --rect 的绿格定义完全一致。

    默认 margin_cells=1(2cm) / inset=0.41 ⇒ 实测 1022 个候选格。
    该函数被 dr3d/bench/green_cells.py::load_green_cells 复用(eval 端随机放置)。
    """
    from scipy import ndimage
    zarr_path = zarr_path or DEFAULT_ZARR
    hammer_path = hammer_path or DEFAULT_HAMMER
    gg = np.arange(-table_half, table_half + 1e-9, gs)
    foot = dataset_arm_footprint(zarr_path, hammer_path, gs, table_half,
                                 min_height=arm_height)
    if margin_cells:
        foot = ndimage.binary_dilation(foot, iterations=int(margin_cells))
    XX, YY = np.meshgrid(gg, gg)
    keep = ((~foot) & (np.abs(XX) <= inset - 1e-9) &
            (np.abs(YY) <= inset - 1e-9))
    return np.stack([XX[keep], YY[keep]], axis=1)


# --- 通用相机版(斜视相机, 如 adroit pen 的 vil_camera) -----------------------
# hammer/door 的策略相机是顶视 `top`, 上面的 dataset_arm_footprint/green_cells 用
# 闭式反投影(快)。pen 的 vil_camera 是 ~45° 斜视: 同一 depth 像素在不同行对应不同
# 的视线方向, 闭式不再成立, 必须逐点反投影 —— cam_geom.pixels_to_world 即该公式
# (实测: 桌面平面落在 world z=0 且 xy 跨 ±0.45)。
# 语义完全对齐: 足迹 = "高于桌面 arm_height 的点在世界 xy 上的落点"(斜视时手在
# 桌面上的投影足迹会被自然拉长, 正是"球放这里会被手挡住/碰手"的位置)。
def camera_geometry(xml_path, cam_name, img=84):
    """只读加载 XML -> cam_geom.geometry(该相机); cam_name 不存在则报错。"""
    from mujoco_py import load_model_from_path
    return cam_geom.geometry(load_model_from_path(str(xml_path)), cam_name, img=img)


def world_arm_footprint(zarr_path, xml_path, cam_name, gs, table_half,
                        min_height=0.05, step=1, img=84):
    """通用相机版「手/物足迹」(2cm 桌面格, 返回 (N,N) bool)。

    逐帧取 depth(84x84, 米) -> 反投影为 world 3D -> 保留 world z >= min_height 的
    点 -> 打到桌面格。dataset_arm_footprint 是本函数在 top 相机下的特例(保留其原
    实现, hammer/door 路径零改动)。
    """
    import zarr
    g = camera_geometry(xml_path, cam_name, img=img)
    D = zarr.open(zarr_path, mode="r")["data"]["depth"]
    T = int(D.shape[0]) if hasattr(D, "shape") else len(D)
    H, W = (tuple(int(v) for v in D.shape[1:3]) if hasattr(D, "shape")
            else np.asarray(D[0]).shape)
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    N = len(np.arange(-table_half, table_half + 1e-9, gs))
    grid = np.zeros(N * N, bool)
    for i in range(0, T, step):
        dep = np.asarray(D[i], dtype=np.float64)
        m = dep > 0                                  # 0 = 无效/未命中
        if not m.any():
            continue
        w = cam_geom.pixels_to_world(rr[m], cc[m], dep[m], g)
        m2 = np.isfinite(w[:, 2]) & (w[:, 2] >= min_height)   # 高于桌面者
        if not m2.any():
            continue
        w = w[m2]
        ix = np.floor((w[:, 0] + table_half) / gs).astype(int)
        iy = np.floor((w[:, 1] + table_half) / gs).astype(int)
        ok = (ix >= 0) & (ix < N) & (iy >= 0) & (iy < N)
        grid[iy[ok] * N + ix[ok]] = True
    return grid.reshape(N, N)


def green_cells_cam(zarr_path, xml_path, cam_name, gs=0.02, table_half=0.44,
                    arm_height=0.05, margin_cells=1, inset=0.41, img=84):
    """green_cells 的通用相机版(口径/参数完全一致, 只换足迹的算法)。

    绿区 = 桌面 ±table_half 去掉手/物足迹(膨胀 margin_cells 格) 且 |x|,|y| <= inset。
    pen: 由 dr3d/bench/green_cells.py::compute_green_cells(task='pen') 调用。
    """
    from scipy import ndimage
    gg = np.arange(-table_half, table_half + 1e-9, gs)
    foot = world_arm_footprint(zarr_path, xml_path, cam_name, gs, table_half,
                               min_height=arm_height, img=img)
    if margin_cells:
        foot = ndimage.binary_dilation(foot, iterations=int(margin_cells))
    XX, YY = np.meshgrid(gg, gg)
    keep = ((~foot) & (np.abs(XX) <= inset - 1e-9) &
            (np.abs(YY) <= inset - 1e-9))
    return np.stack([XX[keep], YY[keep]], axis=1)


def estimate_xy_map(board_pc_xy, board_world_xy):
    """PC<->world xy 仿射(仅 2x2 反射 + 平移):
       pc = R @ world + t,  R in {diag(±1, ±1)}.
    平移用 nail_board 点云质心<->世界中心对齐; 反射符号用 obstacle 对应点挑选。"""
    best = None
    cands = [np.array([[1, 0], [0, 1]], float),
             np.array([[-1, 0], [0, 1]], float),
             np.array([[1, 0], [0, -1]], float),
             np.array([[-1, 0], [0, -1]], float)]
    for R in cands:
        t = board_pc_xy - R @ board_world_xy
        pred = R @ OBS_WORLD + t
        err = float(np.linalg.norm(pred - OBS_PC))
        if best is None or err < best[0]:
            best = (err, R, t)
    err, R, t = best
    return R, t, err


def _fit_homography(src, dst):
    """src/dst: (N,2) 世界平面点 -> 像素点, DLT 求 3x3 H (h33=1)."""
    A, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        b.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b.append(v)
    sol, *_ = np.linalg.lstsq(np.array(A, float), np.array(b, float), rcond=None)
    return np.append(sol, 1.0).reshape(3, 3)


def _project_h(H, pts):
    pts = np.asarray(pts, float)
    ones = np.ones((len(pts), 1))
    hh = H @ np.concatenate([pts, ones], axis=1).T
    return (hh[:2] / hh[2:3]).T


def _rgb2hsv_vec(arr):
    r, g, b = arr[:, 0] / 255.0, arr[:, 1] / 255.0, arr[:, 2] / 255.0
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = mx - mn
    h = np.zeros_like(mx)
    ok = d > 1e-6
    rk = ok & (mx == r)
    gk = ok & (mx == g)
    bk = ok & (mx == b)
    h[rk] = 60 * (((g[rk] - b[rk]) / d[rk]) % 6)
    h[gk] = 60 * ((b[gk] - r[gk]) / d[gk] + 2)
    h[bk] = 60 * ((r[bk] - g[bk]) / d[bk] + 4)
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0.0)
    return (h + 360) % 360, s, mx


def calibrate_homography(hammer_path, size):
    """在桌面上放 8 个颜色编码小标记(±0.25 角/边), 渲染后按色相识别像素质心,
    拟合并返回 桌面世界xy -> 图像像素 的单应 H."""
    import colorsys
    from scipy import ndimage
    hues = np.linspace(80, 330, 8)                 # 避开皮肤橙/木柄棕/灰白低饱和
    wpos = [(-0.25, -0.25), (0.25, -0.25), (0.25, 0.25), (-0.25, 0.25),
            (0.25, 0.0), (-0.25, 0.0), (0.0, 0.25), (0.0, -0.25)]
    half = 0.02
    geoms = []
    for i, (mx_, my_) in enumerate(wpos):
        h = hues[i]
        r, g, b = [int(round(c * 255)) for c in colorsys.hsv_to_rgb(h / 360, 0.9, 1.0)]
        geoms.append(
            f'<geom name="cal{i}" type="box" size="{half} {half} 0.001" '
            f'pos="{mx_:.4f} {my_:.4f} 0.004" contype="0" conaffinity="0" '
            f'rgba="{r/255:.3f} {g/255:.3f} {b/255:.3f} 1"/>')
    xml = pathlib.Path(hammer_path).read_text(encoding="utf-8", errors="ignore")
    xml = xml.replace("</worldbody>", "\n".join(geoms) + "\n</worldbody>")
    tmp = os.path.join(os.path.dirname(hammer_path), f"_cal_tmp_{os.getpid()}.xml")
    pathlib.Path(tmp).write_text(xml, encoding="utf-8")
    try:
        img, _ = render_with_verify_camera(tmp, size)
    finally:
        os.remove(tmp)
    pix = img.reshape(-1, 3).astype(np.float64)
    hue, sat, val = _rgb2hsv_vec(pix)
    src, dst = [], []
    tol = 12.0
    for h, wp in zip(hues, wpos):
        dh = np.abs((hue - h + 540) % 360 - 180)
        m = (dh < tol) & (sat > 0.45) & (val > 0.2)
        lab, nlab = ndimage.label(m.reshape(img.shape[0], img.shape[1]))
        if nlab == 0:
            continue
        areas = ndimage.sum(m.reshape(img.shape[0], img.shape[1]),
                            lab, range(1, nlab + 1))
        best = int(np.argmax(areas)) + 1          # 取该色相最大连通块=标记
        ys, xs = np.where(lab == best)
        if len(xs) < 4:
            continue
        src.append(wp)
        dst.append([float(xs.mean()), float(ys.mean())])
    if len(src) < 4:
        raise SystemExit(f"[calib] 只识别到 {len(src)}/8 个标记, 无法标定 H")
    H = _fit_homography(np.array(src), np.array(dst))
    return H, len(src)


def run_rect_mode(a):
    """方案1(无标签矩形禁放区)。

    禁放区 = ID 全点云 dataset-union xy bbox(PC 系, 红), 框外 = 可放置区(绿)。
    只读 --zarr 的 depth / raw_point_cloud / raw_point_cloud_len, **不读 raw_point_cloud_label**;
    PC<->world 用硬编码常量(MAP_R/MAP_T), world->像素 用渲染标定的 H。
    默认红区 = 免标签 depth 推出的手/锤精确足迹(--rect-shape footprint);
    绿 = 桌面 workspace(table_half)去掉红区。因 top 相机垂直俯视且视野 ±0.4556m ≈ 桌面 ±0.45m
    (桌面占画面 98.8%, 每侧仅多 5.6mm 地面边), 该区域就是渲染图里桌面的剩余部分。
    """
    import imageio
    from PIL import Image, ImageDraw

    # --- 可选 distractor(rect 模式只支持显式 --at) ---------------------------
    objs = [o.strip() for o in a.objects.split(",") if o.strip()]
    place_xy = []
    if objs:
        TL = _test_load()
        bad = [o for o in objs if o not in TL.OBJ_XML]
        if bad:
            raise SystemExit(f"--objects 无效: {bad}, 可选 {sorted(TL.OBJ_XML)}")
        if not a.at:
            print("[warn] --rect 模式无红/绿格可自动选点, 请用 --at 'x,y;...'; 本次不加载对象")
            objs = []
        else:
            parts = [s.strip() for s in a.at.split(";") if s.strip()]
            if len(parts) != len(objs):
                raise SystemExit(f"--at 数量({len(parts)}) != --objects({len(objs)})")
            for s in parts:
                x, y = (float(v) for v in s.replace(",", " ").split())
                place_xy.append((x, y))

    scene_xml = None
    if objs:
        TL = _test_load()
        scene_xml = pathlib.Path(a.hammer).read_text(encoding="utf-8", errors="ignore")
        for obj, (x, y) in zip(objs, place_xy):
            scene_xml = compose_distractor_xml(
                scene_xml, obj, TL.DST / TL.OBJ_XML[obj], f"{x} {y}")
            print(f"[object] {obj} at world ({x:+.3f}, {y:+.3f})")

    img_orig = _render_first_frame(a, scene_xml)
    stem = pathlib.Path(a.out)
    if stem.suffix.lower() == ".png":
        stem = stem.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    tag = "+".join(objs)
    suf = f"_{tag}" if tag else ""
    out_orig, out_rect = f"{stem}{suf}_orig.png", f"{stem}{suf}_rect.png"
    imageio.imwrite(out_orig, img_orig)
    print(f"[save] orig: {out_orig}")

    # --- 禁放矩形(PC 系) -----------------------------------------------------
    if a.workspace_rect:
        x0, y0, x1, y1 = (float(v) for v in a.workspace_rect.replace(";", ",").split(","))
        src = "--workspace-rect"
    else:
        x0, y0, x1, y1 = dataset_union_xy_bbox(a.zarr)
        src = "dataset-union bbox"
    print(f"[rect] forbidden bbox ({src}): x[{x0:+.4f},{x1:+.4f}] "
          f"y[{y0:+.4f},{y1:+.4f}]  w={x1 - x0:.3f} h={y1 - y0:.3f}")

    # --- world-xy -> 像素 ----------------------------------------------------
    Hm, n_cal = calibrate_homography(a.hammer, a.size)
    print(f"[calib] homography fitted from {n_cal}/8 markers")

    # --- 桌面格: 禁放(红) / 可放置(绿) --------------------------------------
    TB, gs = a.table_half, a.grid
    gg = np.arange(-TB, TB + 1e-9, gs)
    XX, YY = np.meshgrid(gg, gg)
    wc = np.stack([XX.ravel(), YY.ravel()], axis=1)          # 桌面格中心(world xy)
    pc = (MAP_R @ wc.T).T + MAP_T                            # -> PC 系
    bbox = ((pc[:, 0] >= x0) & (pc[:, 0] <= x1) &
            (pc[:, 1] >= y0) & (pc[:, 1] <= y1))
    if a.rect_shape == "footprint":
        forbid = dataset_arm_footprint(
            a.zarr, a.hammer, gs, TB, min_height=a.arm_height).ravel()
        print(f"[footprint] 免标签(depth <= cam_z-{a.arm_height * 100:.0f}cm)推出的"
              f"手/锤足迹: {int(forbid.sum())} cells")
        print(f"[rect] 参考: dataset-union 外接矩形 = {int(bbox.sum())} cells "
              f"(多框 {int((bbox & ~forbid).sum())}, 漏 {int((forbid & ~bbox).sum())})")
    else:
        forbid = bbox
    uv = _project_h(Hm, wc)
    in_img = ((uv[:, 0] >= 0) & (uv[:, 0] < a.size) &
              (uv[:, 1] >= 0) & (uv[:, 1] < a.size))
    place = (~forbid) & in_img
    print(f"[grid] table ±{TB}m @ {gs * 100:.1f}cm ({a.rect_shape}): {len(wc)} cells | "
          f"forbidden={int(forbid.sum())}  placeable={int(place.sum())}")

    # --- 绘制 ----------------------------------------------------------------
    base = Image.fromarray(img_orig).convert("RGBA")
    ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    half = gs / 2.0

    def _quad(w):
        cs = [(w[0] - half, w[1] - half), (w[0] + half, w[1] - half),
              (w[0] + half, w[1] + half), (w[0] - half, w[1] + half)]
        p = _project_h(Hm, np.asarray(cs, float))
        return [(int(u), int(v)) for u, v in p]

    for i in np.where(place)[0]:                             # 可放置(绿)
        dr.polygon(_quad(wc[i]), fill=(40, 215, 70, 90))

    corners_pc = np.asarray([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], float)
    corners_w = (MAP_R.T @ (corners_pc - MAP_T).T).T          # PC -> world
    cuv = _project_h(Hm, corners_w)
    if a.rect_shape == "footprint":
        for i in np.where(forbid)[0]:                        # 禁放 = 精确足迹(红)
            dr.polygon(_quad(wc[i]), fill=(235, 50, 50, 110))
        dr.polygon([(int(u), int(v)) for u, v in cuv],       # 参考: 外接矩形轮廓(橙)
                   outline=(255, 140, 0, 255))
    else:
        dr.polygon([(int(u), int(v)) for u, v in cuv],
                   fill=(235, 50, 50, 70), outline=(235, 50, 50, 255))
    Image.alpha_composite(base, ov).convert("RGB").save(out_rect)
    print(f"[save] rect: {out_rect}")

    # --- top-5 可放置候选(离框心最远, 间距 >= 0.12m) -------------------------
    if place.any():
        ctr = pc[forbid].mean(axis=0) if forbid.any() else np.zeros(2)
        d = np.linalg.norm(pc[place] - ctr, axis=1)
        order = np.where(place)[0][np.argsort(-d)]
        picked = []
        for i in order:
            w = wc[i]
            if all(np.linalg.norm(w - np.array(p)) >= 0.12 for p in picked):
                picked.append((float(w[0]), float(w[1])))
            if len(picked) == 5:
                break
        print("[candidates] top-5 placeable (world xy):")
        for x, y in picked:
            print(f"   x={x:+.3f} y={y:+.3f}")
    else:
        print("[candidates] 无可放置格(检查 --workspace-rect / --table-half)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zarr", default=DEFAULT_ZARR)
    ap.add_argument("--hammer", default=DEFAULT_HAMMER)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--grid", type=float, default=0.02, help="格子宽度 m")
    ap.add_argument("--spread", type=float, default=0.01,
                    help="红区外扩半径 m(默认1cm; 0=不扩)")
    ap.add_argument("--objects", default="",
                    help="逗号分隔加载到场景的 distractor: can,bottle,cereal,mug (默认空=不加载)")
    ap.add_argument("--at", default="",
                    help="放置坐标, ';'分隔多个, 每项 'x,y' (缺省自动选绿格候选)")
    ap.add_argument("--static-margin", dest="static_margin", type=float,
                    default=0.03, help="候选绿格离 nail/board 的最小距离 m(内部过滤)")
    ap.add_argument("--size", type=int, default=84,
                    help="渲染边长(与 eval AdroitEnv.render 同为 84; 可 96/512)")
    # ---- 方案1: 无标签矩形禁放区(不读 raw_point_cloud_label) -----------------
    ap.add_argument("--rect", action="store_true",
                    help="方案1: 红=手/锤禁放足迹, 绿=桌面 workspace 其余(可放置); "
                         "默认走 depth 免标签足迹, 不依赖语义标签")
    ap.add_argument("--rect-shape", dest="rect_shape", choices=["footprint", "rect"],
                    default="footprint",
                    help="--rect 的红区形状: footprint=免标签 depth 推出的精确手/锤足迹(默认); "
                         "rect=dataset-union bbox 外接矩形")
    ap.add_argument("--arm-height", dest="arm_height", type=float, default=0.05,
                    help="footprint 判据: 高于桌面多少 m 视为手/锤(默认 0.05)")
    ap.add_argument("--workspace-rect", dest="workspace_rect", default="",
                    help="手动外接矩形(PC 系) 'xmin,ymin,xmax,ymax', 仅 --rect-shape rect 生效")
    ap.add_argument("--table-half", dest="table_half", type=float, default=0.44,
                    help="桌面半边长 m(默认 0.44 = 桌面 geom 0.45 的保守内缩)")
    a = ap.parse_args()

    if a.rect:
        run_rect_mode(a)
        return

    import zarr
    root = zarr.open(a.zarr, mode="r")
    d, ends = root["data"], root["meta"]["episode_ends"][:]
    n_eps, T = len(ends), int(d["raw_point_cloud_len"].shape[0])
    print(f"[data] {T} frames / {n_eps} episodes: {a.zarr}")

    # 1) 收集 swept(hand/hammer) 与 static(board/nail) 的 PC-xy
    swept, stat = [], []
    for i in range(T):
        raw = np.asarray(d["raw_point_cloud"][i])
        L = int(d["raw_point_cloud_len"][i])
        lab = np.asarray(d["raw_point_cloud_label"][i])[:L]
        pts = raw[:L * 6].reshape(-1, 6)[:, :2]
        m1 = np.isin(lab, [LABEL_HAND, LABEL_HAMMER])
        m2 = np.isin(lab, [LABEL_BOARD, LABEL_NAIL])
        if m1.any():
            swept.append(pts[m1])
        if m2.any():
            stat.append(pts[m2])
    swept = np.concatenate(swept) if swept else np.zeros((0, 2))
    stat = np.concatenate(stat) if stat else np.zeros((0, 2))
    if len(swept) == 0:
        raise SystemExit("no hand/hammer points found")

    # 2) PC<->world xy 估计(board 质心对齐 + obstacle 对应点选反射)
    from mujoco_py import load_model_from_path
    import mujoco_py
    from scipy import ndimage
    mm = load_model_from_path(a.hammer)
    sim = mujoco_py.MjSim(mm)
    sim.forward()
    board_world_xy = sim.data.geom_xpos[mm.geom_name2id("board")][:2].copy()
    board_pc_xy = stat[::max(1, len(stat) // 5000)].mean(axis=0) if len(stat) else swept.mean(0)
    R, t, obs_err = estimate_xy_map(board_pc_xy, board_world_xy)
    print(f"[map] pc = R@world + t  R=\n{R}  t={np.round(t,4)}  "
          f"obstacle_residual={obs_err*100:.1f}cm  board_pc_centroid={np.round(board_pc_xy,4)}")

    def to_world(pc):
        return (R.T @ (pc - t).T).T

    def to_pc(w):
        return (R @ np.asarray(w, float).T).T + t

    # 3) 桌面范围(world ±0.44)网格, 全部格子先按 PC 系排列
    gs = a.grid
    TB = 0.44
    w_corners = np.array([[-TB, -TB], [TB, -TB], [TB, TB], [-TB, TB]], float)
    p_corners = to_pc(w_corners)
    x0, y0 = p_corners.min(axis=0)
    x1, y1 = p_corners.max(axis=0)
    nx = int(np.ceil((x1 - x0) / gs)) + 1
    ny = int(np.ceil((y1 - y0) / gs)) + 1
    cols = x0 + (np.arange(nx) + 0.5) * gs
    rows = y0 + (np.arange(ny) + 0.5) * gs
    XX, YY = np.meshgrid(cols, rows)
    pc_centers = np.stack([XX.ravel(), YY.ravel()], axis=1)   # (nx*ny, 2) PC 系中心
    wc = to_world(pc_centers)                                 # world 中心
    on_table = (np.abs(wc[:, 0]) <= TB) & (np.abs(wc[:, 1]) <= TB)

    # 4) 占用统计: 手/锤(swept) 与 nail/board(static), 供红区与候选过滤
    occ_swept = np.zeros(len(pc_centers), dtype=np.int32)
    occ_stat = np.zeros(len(pc_centers), dtype=np.int32)
    for pts, occ in ((swept, occ_swept), (stat, occ_stat)):
        if len(pts) == 0:
            continue
        c = np.clip(((pts - [x0, y0]) / gs).astype(np.int64), 0, [nx - 1, ny - 1])
        idx = c[:, 1] * nx + c[:, 0]
        np.add.at(occ, idx, 1)

    # 5) 红 = 距任一 手/锤 点 <= spread(体积外扩); 绿 = 桌面内除红之外全部
    occB = (occ_swept > 0).reshape(ny, nx)
    redB = np.zeros((ny, nx), bool)
    if occB.any():
        redB = ndimage.distance_transform_edt(~occB) * gs <= a.spread
    red_flat = redB.ravel() & on_table
    green_flat = on_table & ~red_flat
    n_red = int(red_flat.sum())
    n_green = int(green_flat.sum())
    print(f"[grid] table ±{TB}m grid {gs*100:.1f}cm {nx}x{ny} cells  "
          f"red(spread={a.spread*100:.1f}cm)={n_red}  green(else)= {n_green}")

    # 6) 候选: 绿格且离静态物(nail/board) > static_margin (仅内部过滤, 不影响上图)
    statB = (occ_stat > 0).reshape(ny, nx)
    if statB.any():
        stat_dist_m = ndimage.distance_transform_edt(~statB) * gs
    else:
        stat_dist_m = np.full((ny, nx), np.inf)
    stat_ok = (stat_dist_m.ravel() > a.static_margin) & on_table
    cand_idx = np.where(green_flat & stat_ok)[0]

    # 6b) 候选排序(离红区距离降序) —— 供自动放置与 top-5 打印
    cand_rank = []
    if len(cand_idx):
        red_ctr = wc[red_flat]
        gc = wc[cand_idx]
        if len(red_ctr):
            dd = np.linalg.norm(gc[:, None, :] - red_ctr[None, :, :], axis=2)
            dist_c = dd.min(axis=1)
        else:
            dist_c = np.full(len(gc), np.inf)
        order = np.argsort(-dist_c)
        cand_rank = [(int(cand_idx[j]), float(dist_c[j])) for j in order]

    # 6c) distractor 场景: 解析/自动选点并合成 xml(复用 dr3d/test_load 资产)
    TL = _test_load()
    objs = [o.strip() for o in a.objects.split(",") if o.strip()]
    valid = set(TL.OBJ_XML.keys())
    bad = [o for o in objs if o not in valid]
    if bad:
        raise SystemExit(f"--objects 无效: {bad}, 可选 {sorted(valid)}")
    scene_xml = None
    if objs:
        if a.at:
            parts = [s.strip() for s in a.at.split(";") if s.strip()]
            if len(parts) != len(objs):
                raise SystemExit(f"--at 数量({len(parts)}) != --objects({len(objs)})")
            place_xy = []
            for s in parts:
                x, y = (float(v) for v in s.replace(",", " ").split())
                place_xy.append((x, y))
        else:
            place_xy = []
            LIM, SPAC = 0.32, 0.12
            for cand_i, _dc in cand_rank:                 # 首选限位内远红候选
                w = wc[cand_i]
                if (abs(w[0]) <= LIM and abs(w[1]) <= LIM and
                        all(np.linalg.norm(w - np.array(p)) >= SPAC for p in place_xy)):
                    place_xy.append((float(w[0]), float(w[1])))
                    if len(place_xy) == len(objs):
                        break
            if len(place_xy) < len(objs):                  # 放宽至全桌
                for cand_i, _dc in cand_rank:
                    w = wc[cand_i]
                    if all(np.linalg.norm(w - np.array(p)) >= 0.10 for p in place_xy):
                        place_xy.append((float(w[0]), float(w[1])))
                        if len(place_xy) == len(objs):
                            break
            if len(place_xy) < len(objs):
                print(f"[warn] 候选不足, 只放 {len(place_xy)}/{len(objs)} 个对象")
        scene_xml = pathlib.Path(a.hammer).read_text(encoding="utf-8", errors="ignore")
        for obj, (x, y) in zip(objs, place_xy):
            scene_xml = compose_distractor_xml(
                scene_xml, obj, TL.DST / TL.OBJ_XML[obj], f"{x} {y}")
            print(f"[object] {obj} at world ({x:+.3f}, {y:+.3f})")

    # 7) 输出: 原图 + 图像级红/绿透明网格叠加(场景可含 distractor)
    import imageio
    from PIL import Image, ImageDraw
    stem = pathlib.Path(a.out)
    if stem.suffix.lower() == ".png":
        stem = stem.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    tag = "+".join(objs) if objs else ""
    suf = f"_{tag}" if tag else ""
    out_orig = f"{stem}{suf}_orig.png"
    out_mask = f"{stem}{suf}_mask.png"

    img_orig = _render_first_frame(a, scene_xml)
    imageio.imwrite(out_orig, img_orig)
    print(f"[save] orig: {out_orig}")

    H, n_cal = calibrate_homography(a.hammer, a.size)
    print(f"[calib] homography fitted from {n_cal}/8 markers")

    base = Image.fromarray(img_orig).convert("RGBA")
    ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    alpha = {1: (235, 50, 50, 120), 0: (40, 215, 70, 90)}
    half = gs / 2
    for i in np.where(red_flat | green_flat)[0]:
        tg = 1 if red_flat[i] else 0
        cx, cy = wc[i]
        corners = [(cx - half, cy - half), (cx + half, cy - half),
                   (cx + half, cy + half), (cx - half, cy + half)]
        uv = _project_h(H, np.asarray(corners, float))
        dr.polygon([(int(u), int(v)) for u, v in uv], fill=alpha[tg])
    masked = Image.alpha_composite(base, ov).convert("RGB")
    masked.save(out_mask)
    print(f"[save] mask: {out_mask}")

    # 8) top-5 候选绿格(离红区最远, 世界坐标)
    if cand_rank:
        print("[candidates] top-5 green cell centers (world xy):")
        for cand_i, dc in cand_rank[:5]:
            w = wc[cand_i]
            print(f"   x={w[0]:+.3f} y={w[1]:+.3f}  d_red={dc*100:.1f}cm")
    else:
        print("[candidates] no green cell passing static margin")


if __name__ == "__main__":
    main()
