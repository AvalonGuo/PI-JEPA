#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""pen(vil_camera) dynsphere 可见性判据自检 —— 只读探针(不训练/不跑 eval/不读数据集)。

pen 的策略相机是**斜视** vil_camera(DAPG_pen.xml 里没有 top), 顶视闭式失效, 故
dynsphere 改走通用相机几何; 本探针用**真实 sim.render** 逐条验证该几何(不需要 /hy-tmp
数据集), 8 步 7 项判据:

  [1] 相机事实(cameras/cam_pos/cam_mat0/body_pos/crop) + 策略相机选取;
  [2] 深度帧反投影 -> 桌面平面落在 world z=0 且 xy 跨 ±0.45; crop(z>-0.79) <=> world z>0.01;
  [3] ★恒等式: cam_geom.to_cloud(world) 与 gym_util.PointCloudGenerator 的真实点云
      逐点一致(实测最近邻 p50=5.5e-17, max=1.6e-14 m) => 判据与观测链同源;
  [4] dynsphere 的 AABB(视锥 ∧ point-cloud crop) vs 物理真相(逐像素光线-球求交 +
      真实深度): fp(判是实际不可见)=0 且 beyond-interior=0 必须成立; fn/occluded 无害;
  [5] 各半径下可行放置点数量/高度范围/xy 范围(供文档与调参);
  [6] 真实 DynamicDistractor(task_name='pen'): mode=frustum, 每次放置都满足判据;
  [7] 回归: hammer 仍走旧顶视闭式(mode/cam_z/tan_half/visible_hmax 逐值一致);
  [8] 数据路自检(合成 zarr, 不需数据集): zarr depth -> world_arm_footprint ->
      green_cells_cam, 用模型真值(geom_xpos)交叉验证朝向/索引(镜像 IoU 必须为 0)。

用法(在 /root/3DP 下):
    .venv/bin/python test/verify_pen_dynsphere_vis.py
    .venv/bin/python -u test/verify_pen_dynsphere_vis.py --quick   # 跳过 [4] 网格扫描
输出: 终端报告 + /hy-tmp/3DP/vis/pen_dynsphere_probe.json
提示: 本机 EGL 多 render context 偶发 core dump(重跑即可); 加 -u 免得日志被缓冲吞掉。
"""
import argparse
import json
import os
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]          # /root/3DP
DP3 = REPO / "3D-Diffusion-Policy" / "diffusion_policy_3d"
BENCH = DP3 / "dr3d" / "bench"
for _p in (str(REPO), str(DP3), str(BENCH)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PEN_XML = (str(REPO / "third_party/rrl-dependencies/mj_envs/mj_envs/"
              "hand_manipulation_suite/assets/DAPG_pen.xml"))
ENV_NAME = "adroit_pen"          # ENV_POINT_CLOUD_CONFIG 键
IMG = 84
CROP_Z_MARGIN = 0.01             # world z 高于桌面 1cm 才进 crop(crop z>-0.79 + cam_z=0.8)
OUT_JSON = pathlib.Path("/hy-tmp/3DP/vis/pen_dynsphere_probe.json")


def policy_cloud_config(env_name=ENV_NAME):
    """读真实策略配置(transform/min/max_bound/cam_names), 判据必须与观测完全一致。"""
    from diffusion_policy_3d.gym_util.mjpc_diffusion_wrapper import ENV_POINT_CLOUD_CONFIG
    return dict(ENV_POINT_CLOUD_CONFIG[env_name])


def compose_pen_sphere_scene(pos=("-0.22", "-0.15"), mocap=True):
    """合成 pen + dr_sphere 场景(缓存到 dr3d/tmp/scenes, third_party 只读)。"""
    import compose as C
    pos_s = pos if isinstance(pos, str) else " ".join(str(p) for p in pos)
    path, sha, fresh = C.ensure_scene("sphere", pos_s, mocap=mocap, base="pen")
    return pathlib.Path(path), sha, fresh


def make_sim(xml_path):
    from mujoco_py import load_model_from_path, MjSim
    return MjSim(load_model_from_path(str(xml_path)))


def render_depth(sim, cam="vil_camera", img=IMG):
    """真实深度帧 -> 米(与 mujoco_point_cloud.captureImage 完全一致: 含 verticalFlip)。

    返回的行/列索引与 zarr/点云同向(captureImage 里的 verticalFlip 已在内部做掉),
    因此可直接配 cam_geom.pixels_to_world 使用。
    """
    _, raw = sim.render(img, img, camera_name=cam, depth=True, device_id=0)
    ext = sim.model.stat.extent
    near = sim.model.vis.map.znear * ext
    far = sim.model.vis.map.zfar * ext
    raw = np.flip(raw, axis=0)                       # == verticalFlip(见 mujoco_point_cloud)
    return near / (1.0 - raw * (1.0 - near / far))


def sphere_center(sim, name="dr_sphere", radius=None):
    """放置球(mocap 体) -> (body_id, mid, geom_id, 子树内全部 geom_id)."""
    bid = int(sim.model.body_name2id(name))
    if bid < 0:
        raise ValueError(f"body {name!r} not in model")
    ids, changed = {bid}, True
    while changed:                                   # body 子树(资产内嵌 <body object>)
        changed = False
        for b in range(sim.model.nbody):
            if b not in ids and int(sim.model.body_parentid[b]) in ids:
                ids.add(b)
                changed = True
    gids = [g for g in range(sim.model.ngeom)
            if int(sim.model.geom_bodyid[g]) in ids]
    if not gids:
        raise ValueError(f"{name}: no geom in body subtree")
    mid = int(sim.model.body_mocapid[bid])
    if radius is not None:
        for g in gids:
            if int(sim.model.geom_type[g]) == 2:     # mjGEOM_SPHERE
                sim.model.geom_size[g, 0] = float(radius)
    return bid, mid, int(gids[0]), gids


def place_sphere(sim, center, radius):
    bid, mid, _gid, gids = sphere_center(sim, radius=radius)
    if mid >= 0:
        sim.data.mocap_pos[mid] = np.asarray(center, float)
    sim.model.body_pos[bid] = np.asarray(center, float)   # mj_reset 会据此复位
    sim.forward()
    return bid, mid, gids


def ray_sphere_depth(cam_pos, R, f, img, center, radius, pad=40):
    """扩展像素网格上的光线-球求交 -> dict(u, v, z, hit, p).

    扩展 pad 像素是为了让"球伸出画幅"也能被物理检出(pad 需大于球投影半径)。
    """
    cs = np.arange(-pad, img + pad)
    U, V = np.meshgrid(cs, cs)                        # (N,N) 列/行索引
    # p_cam ∝ (u-cx, -(v-cy), -f); 方向(世界) = R @ 该向量
    d_cam = np.stack([U - img / 2.0, -(V - img / 2.0),
                      np.full(U.shape, -float(f))], axis=-1)
    d_world = d_cam @ R.T                             # (N,N,3)
    dn = d_world / np.linalg.norm(d_world, axis=-1, keepdims=True)
    oc = cam_pos - np.asarray(center, float)          # 从中心指向相机
    b = np.sum(dn * oc, axis=-1)
    c = float(np.sum(oc ** 2) - radius ** 2)
    disc = b ** 2 - c
    hit = disc > 0
    t = np.where(hit, -b - np.sqrt(np.maximum(disc, 0.0)), np.nan)
    hit &= t > 1e-6
    p = cam_pos[None, None, :] + t[..., None] * dn
    p_cam = (p - cam_pos) @ R
    return dict(u=U, v=V, hit=hit, p=p, z=-p_cam[..., 2],
                in_frame=(hit & (U >= 0) & (U <= img - 1) &
                          (V >= 0) & (V <= img - 1)))

def check_camera(model, cam_geom, cfg):
    """相机事实 + 静态相机判定(★恒等式适用性)。"""
    names = cam_geom.camera_names(model)
    g = cam_geom.find_geometry(model, cfg["cam_names"], img=IMG)
    c2w_z = float(g["pos"][2]) + float(np.asarray(cfg["min_bound"], float)[2])
    rep = dict(cameras=names, has_top=("top" in names), cam=g["cam"],
               cam_names=list(cfg["cam_names"]), pos=g["pos"].tolist(),
               fovy=g["fovy"], f=g["f"], body_id=g["body_id"],
               body_pos=g["body_pos"].tolist(),
               min_bound=list(np.asarray(cfg["min_bound"], float)),
               max_bound=list(np.asarray(cfg["max_bound"], float)),
               crop_z_threshold_world=c2w_z)
    print(f"[1] cameras={names}")
    print(f"    policy cam_names={list(cfg['cam_names'])} -> {g['cam']} "
          f"(no 'top' camera: {'top' not in names})")
    print(f"    cam pos={np.round(g['pos'], 4).tolist()} fovy={g['fovy']:g} "
          f"f={g['f']:.4f} cam_bodyid={g['body_id']} "
          f"body_pos={np.round(g['body_pos'], 4).tolist()}")
    print(f"    crop min={np.round(cfg['min_bound'], 3).tolist()} "
          f"max={np.round(cfg['max_bound'], 3).tolist()} "
          f"transform={'None' if cfg.get('transform') is None else 'set'}")
    print(f"    static camera(body_pos==0) => cloud = world - cam_pos; "
          f"crop 等价于 world z > {c2w_z:+.3f} (桌面 z=0 起算)")
    # 事实记录型检查: 相机必须存在、内参/外参自洽(cam_names 里的相机在模型里找得到)
    rep["pass"] = bool(g["cam"] in list(cfg["cam_names"]) and g["fovy"] > 0
                       and abs(g["f"] - IMG / (2 * np.tan(np.radians(g["fovy"]) / 2)))
                       < 1e-6)
    return g, rep


def check_depth_roundtrip(sim, g, cam_geom, cfg, table_half=0.45):
    """深度帧 -> world: 桌面平面必须落在 z≈0 且 xy 跨 ±table_half。"""
    Z = render_depth(sim, cam=g["cam"])
    rows, cols = np.meshgrid(np.arange(IMG), np.arange(IMG), indexing="ij")
    pw = cam_geom.pixels_to_world(rows, cols, Z, g)
    cloud = cam_geom.to_cloud(pw, g, cfg.get("transform"), cfg.get("scale"),
                              cfg.get("offset"))
    tb = np.abs(pw[..., 2]) < CROP_Z_MARGIN
    keep_cloud = cloud[..., 2] > np.asarray(cfg["min_bound"], float)[2]
    keep_world = pw[..., 2] > (float(g["pos"][2])
                               + np.asarray(cfg["min_bound"], float)[2])
    rep = dict(table_px=int(tb.sum()),
               table_xy=[float(pw[..., 0][tb].min()), float(pw[..., 0][tb].max()),
                         float(pw[..., 1][tb].min()), float(pw[..., 1][tb].max())],
               crop_equiv_mismatch=int((keep_cloud != keep_world).sum()))
    print(f"[2] depth round-trip: table(|z|<{CROP_Z_MARGIN}) px={rep['table_px']} "
          f"x=[{rep['table_xy'][0]:+.3f},{rep['table_xy'][1]:+.3f}] "
          f"y=[{rep['table_xy'][2]:+.3f},{rep['table_xy'][3]:+.3f}] (expect ±{table_half})")
    ok = (rep["table_px"] > 1000
          and abs(rep["table_xy"][0] + table_half) < 0.02
          and abs(rep["table_xy"][1] - table_half) < 0.02
          and rep["crop_equiv_mismatch"] == 0)
    print(f"    {'PASS' if ok else 'FAIL'}: table plane + crop<->world-z equivalence "
          f"(mismatch={rep['crop_equiv_mismatch']})")
    rep["pass"] = bool(ok)
    return rep


def check_cloud_identity(sim, g, cam_geom, cfg):
    """★恒等式 vs 真实 PointCloudGenerator(逐点最近邻)。"""
    from diffusion_policy_3d.gym_util.mujoco_point_cloud import PointCloudGenerator
    gen = PointCloudGenerator(sim=sim, cam_names=[g["cam"]], img_size=IMG)
    real, _d = gen.generateCroppedPointCloud()
    real = np.asarray(real, float)[:, :3]
    Z = render_depth(sim, cam=g["cam"])
    rows, cols = np.meshgrid(np.arange(IMG), np.arange(IMG), indexing="ij")
    ana = cam_geom.to_cloud(cam_geom.pixels_to_world(rows, cols, Z, g), g,
                            cfg.get("transform"), cfg.get("scale"), cfg.get("offset"))
    from scipy.spatial import cKDTree
    ana = ana.reshape(-1, 3)
    tree = cKDTree(real)
    dist, _i = tree.query(ana, k=1)
    lo = np.asarray(cfg["min_bound"], float)
    hi = np.asarray(cfg["max_bound"], float)
    inb = np.all(ana > lo, axis=1) & np.all(ana < hi, axis=1)   # 只比 crop 内的点
    dc, _j = tree.query(ana[inb], k=1) if inb.any() else (np.zeros(0), None)
    rep = dict(n_real=int(real.shape[0]), n_pixels=int(ana.shape[0]),
               n_crop=int(inb.sum()),
               nn_max=float(np.nanmax(dist)),
               nn_p999=float(np.nanpercentile(dist, 99.9)),
               nn_p50=float(np.nanmedian(dist)),
               nn_crop_max=float(np.nanmax(dc)) if len(dc) else float("nan"),
               nn_crop_p50=float(np.nanmedian(dc)) if len(dc) else float("nan"),
               real_z=[float(real[:, 2].min()), float(real[:, 2].max())],
               ana_z=[float(ana[:, 2].min()), float(ana[:, 2].max())])
    print(f"[3] cloud identity: points real={rep['n_real']} analytic={rep['n_pixels']} "
          f"| NN dist p50={rep['nn_p50']:.2e} p99.9={rep['nn_p999']:.2e} "
          f"max={rep['nn_max']:.2e}")
    print(f"    crop 内 {rep['n_crop']} 点: NN p50={rep['nn_crop_p50']:.2e} "
          f"max={rep['nn_crop_max']:.2e}  <-- 判据看这两个")
    print(f"    cloud z: real=[{rep['real_z'][0]:+.4f},{rep['real_z'][1]:+.4f}] "
          f"analytic=[{rep['ana_z'][0]:+.4f},{rep['ana_z'][1]:+.4f}]")
    rep["pass"] = bool(rep["n_real"] == rep["n_pixels"] and
                       rep["nn_crop_max"] < 1e-4 and rep["nn_p999"] < 1e-6)
    print(f"    {'PASS' if rep['pass'] else 'FAIL'}: * cloud = world - cam_pos "
          f"(+ transform/scale/offset)")
    return rep

def _cloud_in_crop(p, g, cam_geom, cfg):
    """world 点 -> 是否在 crop(严格不等式, 与 wrapper 一致)内。"""
    c = cam_geom.to_cloud(p, g, cfg.get("transform"), cfg.get("scale"),
                          cfg.get("offset"))
    lo = np.asarray(cfg["min_bound"], float)
    hi = np.asarray(cfg["max_bound"], float)
    return np.all(c > lo, axis=-1) & np.all(c < hi, axis=-1)


def depth_jump_mask(Z, tol=0.005):
    """渲染深度图里的不连续像素: True = 该像素的 4 邻域有 > tol 的深度跳变。

    手/球/桌的轮廓处(MSAA/亚像素混合)深度会落到前后物体之一, 与"球的解析深度"
    不可直接比对; 这些像素单独归类(与 AABB 用 margin_px 保守留边同源)。
    """
    m = np.zeros(np.asarray(Z).shape, bool)
    for ax in (0, 1):
        d = np.abs(np.diff(Z, axis=ax)) > tol
        sl = [slice(None)] * 2
        sl[ax] = slice(0, -1)
        m[tuple(sl)] |= d
        sl[ax] = slice(1, None)
        m[tuple(sl)] |= d
    return m


def check_sphere_visibility(sim, g, cam_geom, cfg, r_set, xs, ys, hs,
                            margin_px, tol=0.005, pad=40):
    """AABB 判据 vs 物理真相(逐像素光线求交 + 真实深度)的混淆矩阵。

    physical_visible  := 球面所有命中像素都在画幅内 且 都在 crop 内(不看遮挡);
    physical_rendered := physical_visible 且 每个**内部**命中像素的真实深度都等于球面
                         深度(tol_z, 见下) => 球确实被渲染出来(没被手/桌挡在后面)。

    两类像素单列(与 AABB 用 margin_px 保守留边同源, 不参与硬判据):
      * 轮廓边缘(dist_in < 2px): 球只有 ~8px 半径, 解析切线 vs 像素中心有亚像素差;
      * 渲染图的深度不连续像素(手/球/桌轮廓, depth_jump_mask)。
    掠射像素的深度容差取物理上界 tol_z = max(tol, sqrt(2*r*dpix)+dpix),
    dpix = 半个像素在该像素球距处的物理尺寸。

    fp(判是但实际不可见)与 beyond-interior(击中球内部却渲染更远)必须为 0;
    fn(AABB 保守) 与 occluded(被手挡住)无害, 单独统计。
    """
    lo = np.asarray(cfg["min_bound"], float)
    hi = np.asarray(cfg["max_bound"], float)
    from scipy import ndimage
    stat = dict(n=0, fp=0, fn=0, occluded=0, beyond=0, beyond_edge=0, max_dz=0.0,
                n_hit=[], examples=[])
    for r in r_set:
        for h in hs:
            for x in xs:
                for y in ys:
                    center = (float(x), float(y), float(h))
                    place_sphere(sim, center, float(r))
                    Z = render_depth(sim, cam=g["cam"])
                    pk = ray_sphere_depth(g["pos"], g["R"], g["f"], IMG, center,
                                          float(r), pad=pad)
                    hit = pk["hit"]
                    if not hit.any():
                        continue
                    # 轮廓内到边界的像素距离: <2px 的算“边缘像素”。球只有 ~8px 半径,
                    # 边缘一圈受光栅化/亚像素采样影响(解析切线 vs 像素中心), 不计入
                    # 硬判据 —— 与 AABB 用 margin_px 保守留边的理由一致。
                    dist_in = ndimage.distance_transform_edt(hit)
                    jump = depth_jump_mask(Z)      # 渲染图里的轮廓像素(含手/桌)
                    stat["n"] += 1
                    n_hit = int(hit.sum())
                    n_out = int((hit & ~pk["in_frame"]).sum())
                    crop_ok = _cloud_in_crop(pk["p"], g, cam_geom, cfg)
                    n_cut = int((hit & ~crop_ok).sum())
                    phys_vis = (n_hit > 0 and n_out == 0 and n_cut == 0)
                    sel = hit & pk["in_frame"] & crop_ok          # 预测应命中的像素
                    n_behind, n_beyond, n_beyond_edge, dmax = 0, 0, 0, 0.0
                    if sel.any():
                        # pk 的 u/v 已是**正立(存储)图像**坐标(与 render_depth 的翻转后
                        # 数组同向, 见 cam_geom 恒等式) -> 直接索引, 切勿再翻一次
                        vr = pk["v"][sel].astype(int)
                        uc = pk["u"][sel].astype(int)
                        dZ = Z[vr, uc] - pk["z"][sel]
                        edge = (dist_in[sel] < 2.0) | jump[vr, uc]   # 边缘(亚像素敏感)
                        # 掠射像素的深度容差: 射线偏半个像素(在该像素球距处 = dpix 米)
                        # 时, 切线附近可多深入 ~sqrt(2*r*dpix)(球面几何) -> 物理上界。
                        dpix = 0.5 * float(np.median(pk["z"][sel])) / g["f"]
                        tol_z = max(float(tol), float(np.sqrt(2.0 * r * dpix)) + dpix)
                        beyond = dZ > tol_z                   # 渲染比球更远
                        n_behind = int((dZ < -tol_z).sum())   # 球前面有东西(遮挡) 无害
                        n_beyond = int((beyond & ~edge).sum())     # 内部像素: 必须 0
                        n_beyond_edge = int((beyond & edge).sum())  # 边缘一圈: 记录
                        if n_beyond:
                            dmax = float(dZ[beyond & ~edge].max())
                            stat["max_dz"] = max(stat["max_dz"], dmax)
                        stat["tol_z"] = max(stat.get("tol_z", 0.0), tol_z)
                    phys_rend = phys_vis and n_beyond == 0
                    ana = bool(cam_geom.aabb_visible(
                        np.array([x]), np.array([y]), h, r, g, lo, hi,
                        cfg.get("transform"), cfg.get("scale"), cfg.get("offset"),
                        margin_px=margin_px)[0])
                    stat["n_hit"].append(n_hit)
                    if ana and not phys_vis:
                        stat["fp"] += 1
                        stat["examples"].append(dict(kind="fp", r=r, x=x, y=y, h=h,
                                                     n_hit=n_hit, n_out=n_out,
                                                     n_cut=n_cut))
                    elif not ana and phys_vis:
                        stat["fn"] += 1
                    elif ana and phys_vis and not phys_rend:
                        stat["beyond"] += 1
                        stat["beyond_edge"] += n_beyond_edge
                        stat["examples"].append(dict(kind="beyond", r=r, x=x, y=y,
                                                     h=h, n_hit=n_hit,
                                                     n_beyond=n_beyond,
                                                     n_edge=n_beyond_edge,
                                                     n_behind=n_behind, dmax=dmax))
                    elif phys_vis and n_behind:
                        stat["occluded"] += 1
    stat["n_hit_mean"] = float(np.mean(stat.pop("n_hit"))) if stat["n"] else 0.0
    print(f"[4] sphere AABB verdict vs physical truth: {stat['n']} placements "
          f"(r={list(r_set)} h={list(hs)})")
    print(f"    false-positive (verdict=visible but physically clipped) = "
          f"{stat['fp']}  <-- must be 0")
    print(f"    beyond-interior (射线命中球内部但渲染更远) = {stat['beyond']}  "
          f"<-- must be 0")
    print(f"    beyond-edge (轮廓/不连续边缘上的同类像素, 亚像素敏感) = "
          f"{stat['beyond_edge']}")
    print(f"    false-negative (AABB conservative) = {stat['fn']}   "
          f"occluded-by-scene (遮蔽, 无害) = {stat['occluded']}")
    print(f"    mean silhouette pixels = {stat['n_hit_mean']:.1f}; "
          f"max |predicted-observed| depth (interior) = "
          f"{stat['max_dz'] * 1000:.2f} mm <= 亚像素掠射上界 "
          f"{stat.get('tol_z', 0.0) * 1000:.2f} mm")
    for e in stat["examples"][:5]:
        print(f"      {e}")
    stat["pass"] = bool(stat["fp"] == 0 and stat["beyond"] == 0)
    print(f"    {'PASS' if stat['pass'] else 'FAIL'}")
    return stat

def report_placements(g, cam_geom, cfg, margin_px, r_set, h_step=0.02, h_max=0.62):
    """各半径下的可行放置空间(候选格 × 高度网格), 供文档/调参。"""
    import green_cells as _gc
    cells = np.asarray(_gc.load_dynsphere_cells("pen", verbose=False), float)
    src = _gc.dynsphere_cells_source("pen")
    hs = np.arange(h_step, h_max + 1e-9, h_step)
    lo = np.asarray(cfg["min_bound"], float)
    hi = np.asarray(cfg["max_bound"], float)
    rows = []
    for r in r_set:
        n, cells_ok, zs = 0, set(), []
        for h in hs:
            m = cam_geom.aabb_visible(cells[:, 0], cells[:, 1], float(h), float(r),
                                      g, lo, hi, cfg.get("transform"),
                                      cfg.get("scale"), cfg.get("offset"),
                                      margin_px=margin_px)
            n += int(m.sum())
            for i in np.where(m)[0]:
                cells_ok.add(int(i))
                zs.append(float(h))
        rows.append(dict(r=float(r), n_placements=int(n), n_cells=int(len(cells_ok)),
                         h_min=(min(zs) if zs else None),
                         h_max=(max(zs) if zs else None)))
        print(f"[5] r={r:.2f}: {n:6d} placements / {len(cells_ok):5d} cells "
              f"h=[{rows[-1]['h_min']},{rows[-1]['h_max']}]")
    print(f"    cells={len(cells)} (source={src}) h grid {h_step}..{h_max} step {h_step}")
    return dict(source=src, n_cells=int(len(cells)), h_grid=[h_step, h_max, h_step],
                per_radius=rows,
                cells_xy=[float(cells[:, 0].min()), float(cells[:, 0].max()),
                          float(cells[:, 1].min()), float(cells[:, 1].max())])


def check_distractor(sim, cam_geom, cfg, margin_px, seeds=(0, 1, 2), plans=3):
    """真实 DynamicDistractor(task_name='pen') 集成检查: 每次放置都满足判据。"""
    import dynamic as _dyn
    import green_cells as _gc
    cells = np.asarray(_gc.load_dynsphere_cells("pen", verbose=False), float)
    d = _dyn.DynamicDistractor(sim, cells, task_name="pen", verbose=True)
    lo = np.asarray(cfg["min_bound"], float)
    hi = np.asarray(cfg["max_bound"], float)
    n, bad_radius, bad_height, bad_vis, bad_infeas = 0, 0, 0, 0, 0
    samples = []
    for s in seeds:
        for ep in range(2):
            d.start_episode(s, ep)
            if float(d.r) not in [float(v) for v in _dyn.R_SET]:
                bad_radius += 1
            feas = np.asarray(d.placements(), float)
            placed = [tuple(d.pos)[:3]] + [tuple(d.plan())[:3]
                                           for _ in range(plans)]
            for (x, y, h) in placed:
                n += 1
                if h < float(d.r) - 1e-9:
                    bad_height += 1
                vis = bool(cam_geom.aabb_visible(
                    np.array([x]), np.array([y]), h, float(d.r), d.geom_info[0],
                    lo, hi, cfg.get("transform"), cfg.get("scale"), cfg.get("offset"),
                    margin_px=margin_px)[0])
                if not vis:
                    bad_vis += 1
                in_feas = (len(feas) > 0 and bool(np.any(np.all(
                    np.abs(feas - np.array([x, y, h])) < 1e-9, axis=1))))
                if not in_feas:      # 含"可行集为空 -> 走最近桌心兜底格"的情形
                    bad_infeas += 1
            samples.append(dict(seed=s, ep=ep, r=float(d.r),
                                placements=[[round(v, 3) for v in p] for p in placed],
                                n_feasible=int(len(feas)), n_cells=len(d.cells_r)))
    print(f"[6] DynamicDistractor(task_name='pen'): mode={d.mode} placements={n} "
          f"bad_radius={bad_radius} bad_height={bad_height} bad_visible={bad_vis} "
          f"not_in_feasible_set={bad_infeas}")
    print(f"    example: {samples[0]}")
    ok = (d.mode == "frustum" and bad_radius == bad_height == bad_vis == bad_infeas == 0)
    print(f"    {'PASS' if ok else 'FAIL'}")
    return dict(mode=d.mode, n=int(n), bad_radius=bad_radius, bad_height=bad_height,
                bad_visible=bad_vis, not_in_feasible_set=bad_infeas,
                samples=samples[:3], pass_=bool(ok))


def check_legacy_top():
    """hammer/door 的顶视闭式路径必须与旧行为逐值一致(无数据也可查)。"""
    import compose as C
    import dynamic as _dyn
    xml, _sha, _fresh = C.ensure_scene("sphere", "-0.22 -0.15", mocap=True,
                                       base="hammer")
    sim = make_sim(xml)
    gg = np.arange(-0.44, 0.44 + 1e-9, 0.02)
    XX, YY = np.meshgrid(gg, gg)
    cells = np.stack([XX.ravel(), YY.ravel()], axis=1)
    d = _dyn.DynamicDistractor(sim, cells, task_name="hammer", verbose=True)
    exp_tan = float(np.tan(np.radians(45.0) / 2.0))
    x, y, r = 0.1, -0.1, 0.05
    hmax = float(d.visible_hmax(x, y, r))
    exp = (1.1 - (max(abs(x), abs(y)) + r)
           / (exp_tan * (1.0 - _dyn.VIS_MARGIN_PX / 42.0)))
    ok = (d.mode == "top" and abs(float(d.cam_z) - 1.1) < 1e-9
          and abs(float(d.tan_half) - exp_tan) < 1e-9 and abs(hmax - exp) < 1e-9)
    print(f"[7] hammer legacy top-mode: mode={d.mode} cam_z={d.cam_z} "
          f"tan_half={d.tan_half:.6f} visible_hmax({x},{y},{r})={hmax:.4f} "
          f"(expect {exp:.4f}) -> {'PASS' if ok else 'FAIL'}")
    return dict(mode=d.mode, cam_z=float(d.cam_z), tan_half=float(d.tan_half),
                visible_hmax=float(hmax), expected=float(exp), pass_=bool(ok))


def check_data_path(sim, g, cam_geom, cfg, gs=0.02, table_half=0.44, inset=0.41,
                    tmp="/hy-tmp/3DP/data/_synth_pen_probe.zarr", keep=False):
    """[8] 数据路自检(合成 zarr): zarr depth -> world_arm_footprint -> green_cells_cam。

    深度取自真实 sim.render(**不需要 ID 数据集**), 再用**模型真值**(geom_xpos 高于
    阈值者投影到桌面格)交叉验证: 朝向/索引约定若搞错(多翻/少翻一次), 足迹会与真值
    上下镜像 -> IoU 骤降(实测 flip IoU=0, 正常 IoU≈0.28, 且越界格 <10%)。
    这正是 eval 端 pen 绿区(zarr -> 足迹)的那条路径, 故必须与 [3] 的恒等式一起过。
    """
    import shutil
    import zarr
    import workspace_traffic_map as W
    from scipy import ndimage

    # 确定性: 球放到固定位置(不沿用 [4] 最后一个放置点); 手臂保持 XML 默认姿 ——
    # 本探针全程不 step, 且**不用 sim.reset()**(mujoco_py 会丢弃 render context,
    # 重建 EGL context 在本机不稳定: 曾 core dump / 渲染成整幅背景)。
    place_sphere(sim, (0.30, 0.36, 0.10), 0.02)
    D = np.stack([render_depth(sim, cam=g["cam"]).astype(np.float32)
                  for _ in range(3)])
    print(f"[8] rendered {D.shape[0]} frames "
          f"[{float(D.min()):.3f},{float(D.max()):.3f}]m -> synthetic zarr")
    tmp_p = pathlib.Path(tmp)
    if tmp_p.exists():
        shutil.rmtree(tmp_p)
    dd = zarr.open(str(tmp_p), mode="w").create_group("data")
    dd.create_dataset("depth", data=D, chunks=D.shape)
    print(f"[8] zarr written -> {tmp_p}")

    foot = W.world_arm_footprint(str(tmp_p), PEN_XML, g["cam"], gs, table_half,
                                 min_height=0.05, img=IMG)
    print(f"[8] world_arm_footprint done: {int(foot.sum())} cells")
    cells = W.green_cells_cam(str(tmp_p), PEN_XML, g["cam"], gs=gs,
                              table_half=table_half, arm_height=0.05,
                              margin_cells=1, inset=inset)
    print(f"[8] green_cells_cam done: {len(cells)} cells")
    if not keep:
        shutil.rmtree(tmp_p, ignore_errors=True)

    gg = np.arange(-table_half, table_half + 1e-9, gs)
    N = len(gg)
    gt = np.zeros((N, N), bool)                   # 模型真值: 高于 5cm 的 geom 中心
    xyz = np.asarray(sim.data.geom_xpos)
    hi = xyz[:, 2] > 0.05
    for x, y in xyz[hi][:, :2]:
        ix, iy = int(np.floor((x + table_half) / gs)), int(np.floor((y + table_half) / gs))
        if 0 <= ix < N and 0 <= iy < N:
            gt[iy, ix] = True

    def _iou(a, b):
        u = int((a | b).sum())
        return float((a & b).sum()) / u if u else float("nan")

    iou = _iou(foot, gt)
    iou_flip = _iou(foot, gt[::-1, :])
    near = ndimage.binary_dilation(gt, iterations=2)
    far = int((foot & ~near).sum())

    # 绿区口径: 必须 = 桌面 ±inset − 膨胀 1 格足迹(与 green_cells 完全一致)
    dil = ndimage.binary_dilation(foot, iterations=1)
    XX, YY = np.meshgrid(gg, gg)
    exp = ((~dil) & (np.abs(XX) <= inset - 1e-9) & (np.abs(YY) <= inset - 1e-9))
    ec = np.stack([XX[exp], YY[exp]], axis=1)
    same = (len(cells) == len(ec)) and bool(np.allclose(
        np.unique(cells, axis=0), np.unique(ec, axis=0)))
    ok = bool(foot.any() and iou > 0.15 and iou_flip < 1e-9 and
              far < 0.25 * int(foot.sum()) and same)
    rep = dict(n_frames=int(D.shape[0]), depth_range=[float(D.min()), float(D.max())],
               foot_cells=int(foot.sum()), truth_cells=int(gt.sum()),
               iou=float(iou), iou_yflip=float(iou_flip), far_cells=int(far),
               n_cells=int(len(cells)), cells_formula_match=bool(same),
               keep_zarr=bool(keep), zarr=str(tmp_p if keep else ""))
    print(f"[8] data path (synthetic zarr): depth frames={D.shape[0]} "
          f"[{float(D.min()):.3f},{float(D.max()):.3f}]m "
          f"footprint={int(foot.sum())} cells (model truth={int(gt.sum())}) "
          f"IoU={iou:.3f} (y-flip {iou_flip:.3f}, must be 0) "
          f"far={far} 绿区={len(cells)} 口径一致={same}")
    print(f"    {'PASS' if ok else 'FAIL'}: zarr depth 朝向/索引 与 "
          f"world_arm_footprint/green_cells_cam 口径")
    rep["pass"] = ok
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="跳过 [4] 球体可见性网格扫描(渲染较慢)")
    ap.add_argument("--out", default=str(OUT_JSON))
    ap.add_argument("--pad", type=int, default=40)
    ap.add_argument("--keep-zarr", action="store_true",
                    help="保留 [8] 用的合成 zarr(默认删掉)")
    a = ap.parse_args()
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    import cam_geom
    import dynamic as _dyn

    cfg = policy_cloud_config()
    scene, sha, fresh = compose_pen_sphere_scene()
    print(f"[scene] pen + dr_sphere -> {scene} sha={sha[:12]}"
          f"{' (composed now)' if fresh else ' (cache hit)'}")
    sim = make_sim(scene)
    rep = dict(scene=str(scene), sha=sha, img=IMG, margin_px=float(_dyn.VIS_MARGIN_PX),
               r_set=[float(v) for v in _dyn.R_SET])

    g, rep["camera"] = check_camera(sim.model, cam_geom, cfg)
    rep["depth_roundtrip"] = check_depth_roundtrip(sim, g, cam_geom, cfg)
    rep["cloud_identity"] = check_cloud_identity(sim, g, cam_geom, cfg)
    if a.quick:
        print("[4] skipped (--quick)")
    else:
        rep["sphere_visibility"] = check_sphere_visibility(
            sim, g, cam_geom, cfg, r_set=(0.06, 0.04, 0.02),
            xs=np.arange(-0.4, 0.41, 0.2), ys=np.arange(-0.4, 0.41, 0.2),
            hs=np.arange(0.10, 0.56, 0.15), margin_px=float(_dyn.VIS_MARGIN_PX),
            pad=int(a.pad))
    rep["placements"] = report_placements(g, cam_geom, cfg,
                                          float(_dyn.VIS_MARGIN_PX), _dyn.R_SET)
    rep["distractor"] = check_distractor(sim, cam_geom, cfg,
                                         float(_dyn.VIS_MARGIN_PX))
    # [8] 先用同一个 pen sim 做数据路自检(渲染); [7] 要新建 hammer sim(另一个
    # render context), 放在最后 —— 本机 EGL 上多 context 来回切曾 core dump。
    rep["data_path"] = check_data_path(sim, g, cam_geom, cfg, keep=a.keep_zarr)
    rep["legacy_top"] = check_legacy_top()

    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=2, sort_keys=True)
    keys = ("camera", "depth_roundtrip", "cloud_identity", "sphere_visibility",
            "distractor", "legacy_top", "data_path")
    checks = {k: rep[k].get("pass", rep[k].get("pass_", None))
              for k in keys if k in rep}
    print(f"[report] {a.out}")
    print("[summary] " + ", ".join(f"{k}={'PASS' if v else 'FAIL'}"
                                   for k, v in checks.items()))
    return 0 if all(bool(v) for v in checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
