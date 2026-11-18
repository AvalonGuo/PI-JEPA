#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""pen(vil_camera) + dynhand 可见性判据自检 —— 只读探针(不训练/不跑 eval/不读数据集/不渲染)。

背景: `HandDistractor` 原先硬编码 `sim.model.camera_name2id("top")` 并假设找不到时返回 -1;
但 mujoco_py 对**不存在**的相机是**抛 ValueError**, 而 pen(DAPG_pen.xml)没有 'top' 相机
(策略相机是斜视 vil_camera) ⇒ 每次 `HandDistractor(...)` 构造即崩。本探针 7 步核对修复:

  [1] 相机事实(pen): camera_names 无 'top', 策略相机 = ENV_POINT_CLOUD_CONFIG['adroit_pen']
      的 cam_names(vil_camera), 内参/外参自洽;
  [2] 根因复现(只读): 裸调 camera_name2id('top') **必须抛 ValueError** —— 证明旧写法必崩、
      新的存在性判定(model.camera_names)是必需的;
  [3] HandDistractor(task_name='pen') 构造成功: mode=frustum, cam_names=['vil_camera'],
      crop 与策略配置逐值一致;
  [4] 判据**独立复核**: 对若干 (x,y,z) 把 `visible_frac`(frustum) 与"不复用 cam_geom 的
      手写实现"(直接读 model.cam_mat0/cam_pos/cam_fovy + crop) 比较, 必须逐位一致;
  [5] start_episode 全流程(pen 真实绿区): A/B 在内缩区/长度约束内, z 落在 [z_lo, z_hi],
      path_vis_min 记录, tick/reapply/plan/finish/history/spec 可用且 spec 可 json 序列化,
      单 episode 计时;
  [6] top 分支回归(无 'top' 相机的宽容性): task_name=None 时不再崩溃 -> mode=top + 兜底几何,
      visible_frac 与旧闭式逐值一致;
  [7] hammer 真实场景回归: mode=top, cam_z/tan_half 与直接从 model 读的 'top' 相机一致,
      visible_frac 与旧闭式逐值一致(hammer 行为零变化)。

用法(在 /root/3DP 下):
    .venv/bin/python -u test/verify_pen_dynhand_vis.py
输出: 终端报告 + /hy-tmp/3DP/vis/pen_dynhand_probe.json
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]          # /root/3DP
DP3 = REPO / "3D-Diffusion-Policy" / "diffusion_policy_3d"
BENCH = DP3 / "dr3d" / "bench"
for _p in (str(REPO), str(DP3), str(BENCH)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ENV_NAME = "adroit_pen"
IMG = 84
OUT_JSON = pathlib.Path("/hy-tmp/3DP/vis/pen_dynhand_probe.json")


def policy_cloud_config(env_name=ENV_NAME):
    """读真实策略配置(判据必须与观测链完全一致)。"""
    from diffusion_policy_3d.gym_util.mjpc_diffusion_wrapper import (
        ENV_POINT_CLOUD_CONFIG)
    return dict(ENV_POINT_CLOUD_CONFIG[env_name])


def composed_sim(type_name="inspire", base="pen", mocap=True, scale=1.0,
                 pos=None):
    """合成 base+dynhand 场景(缓存, third_party 只读) -> MjSim。"""
    import compose as C
    from mujoco_py import load_model_from_path, MjSim
    kw = {} if pos is None else {"pos": pos}
    path, sha, fresh = C.ensure_scene(type_name, mocap=mocap, scale=scale,
                                      base=base, **kw)
    print(f"[scene] {base}+{type_name} -> {path} sha={sha[:12]}"
          f"{' (composed now)' if fresh else ' (cache hit)'}")
    return MjSim(load_model_from_path(str(path))), sha


def model_camera(model, name):
    """(独立于 cam_geom) 直接读相机的外参/内参 + 存在性。"""
    names = [str(n) for n in model.camera_names]
    if name not in names:
        return None
    cid = int(model.camera_name2id(name))
    bid = int(np.asarray(model.cam_bodyid).ravel()[cid])
    return dict(name=name, id=cid, names=names,
                pos=np.asarray(model.cam_pos[cid], float).reshape(3),
                R=np.asarray(model.cam_mat0[cid], float).reshape(3, 3),
                fovy=float(np.ravel(model.cam_fovy)[cid]),
                body_id=bid,
                body_pos=np.asarray(model.body_pos[bid], float).reshape(3))


def manual_vis_mask(cam, cfg, pts_world, margin_px):
    """不复用 cam_geom 的判据实现: crop(严格不等式) ∧ 针孔视锥(含 margin)。

    与 gym_util/mujoco_point_cloud.py + mjpc_diffusion_wrapper.py 的观测链一一对应:
        cloud = pT @ (p_world - cam_pos + cam_body_pos) [* scale + offset]
        u,v   = 84×84 针孔投影;  z_cam < 0 表示点在相机前方
    """
    w = np.asarray(pts_world, float)
    cloud = w - cam["pos"] + cam["body_pos"]
    T, sc, off = cfg.get("transform"), cfg.get("scale"), cfg.get("offset")
    if T is not None:
        cloud = cloud @ np.asarray(T, float).T
    if sc is not None:
        cloud = cloud * np.asarray(sc, float)
    if off is not None:
        cloud = cloud + np.asarray(off, float)
    lo = np.asarray(cfg["min_bound"], float)
    hi = np.asarray(cfg["max_bound"], float)
    crop_ok = np.all(cloud > lo, axis=1) & np.all(cloud < hi, axis=1)
    pc = (w - cam["pos"]) @ cam["R"]                     # == R.T @ (p_world-pos)
    f = IMG / (2.0 * np.tan(np.radians(cam["fovy"]) / 2.0))
    z = pc[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        inv = np.where(z < -1e-9, -1.0 / z, np.nan)
        u = IMG / 2.0 + f * pc[:, 0] * inv
        v = IMG / 2.0 - f * pc[:, 1] * inv
    m = float(margin_px)
    hi_px = IMG - 1.0 - m
    frus_ok = (np.isfinite(u) & np.isfinite(v) & (z < 0.0) &
               (u >= m) & (u <= hi_px) & (v >= m) & (v <= hi_px))
    return crop_ok & frus_ok


def pose_points(d, x, y, z):
    """(独立复算) yaw 旋转 + 平移后的手表面世界坐标: 与 _pts_world 同式。"""
    p = np.asarray(d.geom["points_rel"], float)
    cy, sy = np.cos(d.yaw), np.sin(d.yaw)
    px = cy * p[:, 0] - sy * p[:, 1] + x
    py = sy * p[:, 0] + cy * p[:, 1] + y
    return np.stack([px, py, p[:, 2] + z], axis=1)


def check_camera(sim, cfg):
    """[1] pen 相机事实 + [2] 根因(裸调 camera_name2id 必须抛 ValueError)。"""
    names = [str(n) for n in sim.model.camera_names]
    cam = model_camera(sim.model, str(cfg["cam_names"][0]))
    raised = False
    try:
        sim.model.camera_name2id("top")
    except ValueError:
        raised = True
    print(f"[1] cameras={names}")
    print(f"    has_top={'top' in names}  policy cam_names={list(cfg['cam_names'])}"
          f" -> {None if cam is None else cam['name']}")
    if cam is not None:
        f_exp = IMG / (2 * np.tan(np.radians(cam["fovy"]) / 2))
        print(f"    cam pos={np.round(cam['pos'], 4).tolist()} "
              f"fovy={cam['fovy']:g} f={f_exp:.4f} body_pos="
              f"{np.round(cam['body_pos'], 4).tolist()}")
        print(f"    crop min={np.round(cfg['min_bound'], 3).tolist()} "
              f"max={np.round(cfg['max_bound'], 3).tolist()} "
              f"transform={'None' if cfg.get('transform') is None else 'set'}")
    ok1 = (cam is not None and "top" not in names and cam["fovy"] > 0)
    print(f"    {'PASS' if ok1 else 'FAIL'}")
    print(f"[2] camera_name2id('top') raises ValueError: {raised} -> 旧的 "
          f"`if cid >= 0` 写法在 pen 下必崩; 必须先用 model.camera_names 判存在性")
    print(f"    {'PASS' if raised else 'FAIL'}")
    return dict(cameras=names, has_top=("top" in names),
                policy_cam=None if cam is None else cam["name"],
                cam_pos=None if cam is None else cam["pos"].tolist(),
                cam_fovy=None if cam is None else cam["fovy"],
                raw_name2id_raises=bool(raised),
                pass_=bool(ok1), pass_root_cause=bool(raised))


def check_distractor_init(sim, cells, cfg):
    """[3] HandDistractor(task_name='pen') 构造成功且相机分派正确。"""
    import dynamic_hand as DH
    d = DH.HandDistractor(sim, cells, scale=DH.HAND_SCALE, task_name="pen",
                          verbose=True)
    ok = (d.mode == "frustum" and
          [str(n) for n in d.cam_names] == [str(n) for n in cfg["cam_names"]] and
          len(d.geom_info) == 1 and
          str(d.geom_info[0]["cam"]) == str(cfg["cam_names"][0]))
    print(f"[3] HandDistractor(task_name='pen'): mode={d.mode} "
          f"cam_names={list(d.cam_names)} crop="
          f"min{np.round(np.asarray(d.crop[0], float), 3).tolist()}/"
          f"max{np.round(np.asarray(d.crop[1], float), 3).tolist()} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return d, dict(mode=d.mode, cam_names=[str(n) for n in d.cam_names],
                   crop_min=[float(v) for v in np.ravel(d.crop[0])],
                   crop_max=[float(v) for v in np.ravel(d.crop[1])],
                   pass_=bool(ok))


def check_criterion(d, cam, cfg, poses):
    """[4] visible_frac(frustum) 与手写独立实现逐点一致(容差=2 个边界顶点的翻转)。"""
    import dynamic_hand as DH
    nv = int(len(d.geom["points_rel"]))
    tol = 2.0 / max(nv, 1)
    bad = 0
    rows = []
    for (x, y, z) in poses:
        w = pose_points(d, x, y, z)
        man = float(np.mean(manual_vis_mask(cam, cfg, w, DH.VIS_MARGIN_PX)))
        got = float(d.visible_frac(x, y, z))
        dv = abs(man - got)
        ok = dv <= tol
        bad += 0 if ok else 1
        rows.append(dict(x=float(x), y=float(y), z=float(z), got=got,
                         manual=man, diff=dv, ok=bool(ok)))
        print(f"    (x={x:+.3f}, y={y:+.3f}, z={z:.3f}) visible_frac="
              f"{got:.6f} manual={man:.6f} |diff|={dv:.2e} "
              f"{'ok' if ok else 'MISMATCH'}")
    print(f"[4] frustum 判据 vs 手写实现: n={len(poses)} verts={nv} tol={tol:.2e} "
          f"mismatch={bad} -> {'PASS' if bad == 0 else 'FAIL'}")
    return dict(n=len(poses), n_verts=nv, tol=float(tol), mismatch=int(bad),
                samples=rows[:4], pass_=bool(bad == 0))


def check_start_episode(d, seeds=(0, 1, 2), eps=(0, 1)):
    """[5] start_episode 约束/复现/接口/计时(pen 真实绿区, 不渲染)。"""
    import dynamic_hand as DH
    ts = np.linspace(0.0, 1.0, DH.PATH_PROBES)[:, None]
    z_lo_std = max(float(d.z_lo_cfg), -float(d.geom["zmin"]))
    bad_inset = bad_len = bad_z = bad_pose = 0
    n = 0
    samples = []
    t0 = time.time()
    for s in seeds:
        for ep in eps:
            d.start_episode(s, ep)
            n += 1
            seg = d.A[None, :] + (d.B - d.A)[None, :] * ts
            if not np.all(np.max(np.abs(seg), axis=1) <= DH.INSET + 1e-9):
                bad_inset += 1
            if d.L + 1e-9 < d.min_path:
                bad_len += 1
            if not (z_lo_std - 1e-9 <= d.z <= d.z_hi_cfg + 1e-9):
                bad_z += 1
            for k in (0, 1, 2):
                d.tick(k)
            d.reapply()
            if d.pose_err() > 1e-6:
                bad_pose += 1
            d.finish()
            samples.append(dict(
                seed=int(s), ep=int(ep), z=float(d.z),
                A=[round(float(v), 3) for v in d.A],
                B=[round(float(v), 3) for v in d.B], L=float(d.L),
                yaw=float(d.yaw), cells=len(d.cells), cells_g=len(d.cells_g),
                cells_full=int(d.cells_full),
                path_vis_min=(None if d.path_vis_min is None
                              else float(d.path_vis_min)),
                history_len=len(d.history())))
    dt = (time.time() - t0) / max(n, 1)
    spec = d.spec()
    spec_txt = json.dumps(spec)
    print(f"[5] start_episode x{n}: bad_inset={bad_inset} bad_len={bad_len} "
          f"bad_z={bad_z} bad_pose_err={bad_pose} ({dt:.2f} s/episode)")
    print(f"    example: {samples[0]}")
    print(f"    spec jsonable bytes={len(spec_txt)} "
          f"camera_mode={spec.get('camera_mode')} cam_names={spec.get('cam_names')}")
    ok = (bad_inset == bad_len == bad_z == bad_pose == 0 and len(spec_txt) > 0
          and spec.get("camera_mode") == "frustum")
    print(f"    {'PASS' if ok else 'FAIL'}")
    return dict(n=int(n), sec_per_episode=float(dt), bad_inset=int(bad_inset),
                bad_len=int(bad_len), bad_z=int(bad_z),
                bad_pose_err=int(bad_pose), spec_keys=sorted(spec.keys()),
                samples=samples[:4], pass_=bool(ok))


def check_legacy_top(sim, cells):
    """[6] 无 'top' 相机 + task_name=None: 不再崩溃, top 闭式逐值一致。"""
    import dynamic_hand as DH
    d = DH.HandDistractor(sim, cells, scale=DH.HAND_SCALE, task_name=None,
                          verbose=True)
    exp_tan = float(np.tan(np.radians(DH.FOVY_FALLBACK) / 2.0))
    exp_eff = exp_tan * (1.0 - DH.VIS_MARGIN_PX / (IMG / 2.0))
    x, y, z = 0.35, 0.0, 0.20          # 部分出画幅的位姿(能真正验证闭式, 而非全 1.0)
    w = pose_points(d, x, y, z)
    man = float(np.mean(np.maximum(np.abs(w[:, 0]), np.abs(w[:, 1])) <=
                        exp_eff * (DH.CAM_Z_FALLBACK - w[:, 2])))
    got = float(d.visible_frac(x, y, z))
    ok = (d.mode == "top" and abs(float(d.cam_z) - DH.CAM_Z_FALLBACK) < 1e-12 and
          abs(float(d.tan_half) - exp_tan) < 1e-12 and
          abs(float(d.tan_eff) - exp_eff) < 1e-12 and abs(got - man) < 1e-12)
    print(f"[6] task_name=None(pen sim, 无 'top' 相机): mode={d.mode} "
          f"cam_z={d.cam_z} tan_half={d.tan_half:.6f} visible_frac={got:.6f} "
          f"(expect {man:.6f}) -> {'PASS' if ok else 'FAIL'}")
    return dict(mode=d.mode, cam_z=float(d.cam_z), tan_half=float(d.tan_half),
                tan_eff=float(d.tan_eff), visible_frac=got, expected=man,
                pass_=bool(ok))


def check_hammer_legacy():
    """[7] hammer 真实场景: mode=top, 相机几何与旧闭式逐值一致(零行为变化)。"""
    import dynamic_hand as DH
    sim, sha = composed_sim("inspire", base="hammer", mocap=True,
                            scale=DH.HAND_SCALE, pos=("-0.22", "-0.15"))
    # 合成网格代替绿区: 本探针不依赖任何数据集(hammer 绿区缓存需 zarr)
    gg = np.arange(-0.44, 0.44 + 1e-9, 0.02)
    XX, YY = np.meshgrid(gg, gg)
    cells = np.stack([XX.ravel(), YY.ravel()], axis=1)
    d = DH.HandDistractor(sim, cells, scale=DH.HAND_SCALE, task_name="hammer",
                          verbose=True)
    top = model_camera(sim.model, "top")
    exp_z = float(top["pos"][2]) or DH.CAM_Z_FALLBACK
    exp_tan = float(np.tan(np.radians(top["fovy"]) / 2.0))
    x, y, z = 0.35, 0.0, 0.20
    w = pose_points(d, x, y, z)
    man = float(np.mean(np.maximum(np.abs(w[:, 0]), np.abs(w[:, 1])) <=
                        d.tan_eff * (exp_z - w[:, 2])))
    got = float(d.visible_frac(x, y, z))
    ok = (d.mode == "top" and [str(n) for n in d.cam_names] == ["top"] and
          abs(float(d.cam_z) - exp_z) < 1e-12 and
          abs(float(d.tan_half) - exp_tan) < 1e-12 and abs(got - man) < 1e-12)
    print(f"[7] hammer: mode={d.mode} cam_z={d.cam_z} (model top z={exp_z}) "
          f"tan_half={d.tan_half:.6f} (expect {exp_tan:.6f}) "
          f"visible_frac={got:.6f} (expect {man:.6f}) "
          f"-> {'PASS' if ok else 'FAIL'}")
    return dict(sha=sha, mode=d.mode, cam_z=float(d.cam_z), exp_cam_z=exp_z,
                tan_half=float(d.tan_half), exp_tan_half=exp_tan,
                visible_frac=got, expected=man, pass_=bool(ok))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT_JSON))
    ap.add_argument("--cells", type=int, default=0,
                    help=">0 时只用前 N 个绿区格(冒烟加速)")
    a = ap.parse_args()
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    import dynamic_hand as DH
    import green_cells as GC

    cfg = policy_cloud_config()
    cells = np.asarray(GC.load_green_cells(task="pen"), float)
    if a.cells > 0:
        cells = cells[:a.cells]
    print(f"[cells] pen green cells: {len(cells)}")

    sim, sha = composed_sim("inspire", base="pen", mocap=True,
                            scale=DH.HAND_SCALE)
    rep = dict(env=ENV_NAME, sha=sha, img=IMG,
               vis_margin_px=float(DH.VIS_MARGIN_PX), n_cells=int(len(cells)),
               hand_scale=float(DH.HAND_SCALE))
    rep["camera"] = check_camera(sim, cfg)
    d, rep["init"] = check_distractor_init(sim, cells, cfg)
    cam = model_camera(sim.model, str(cfg["cam_names"][0]))
    z_lo = max(float(d.z_lo_cfg), -float(d.geom["zmin"]))
    poses = [(0.0, -0.30, z_lo), (0.24, 0.0, z_lo + 0.05),
             (-0.24, 0.0, 0.25), (0.0, 0.30, 0.30), (0.15, 0.15, z_lo + 0.02)]
    rep["criterion"] = check_criterion(d, cam, cfg, poses)
    rep["start_episode"] = check_start_episode(d)
    rep["legacy_top"] = check_legacy_top(sim, cells)
    rep["hammer"] = check_hammer_legacy()

    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=2, sort_keys=True)
    checks = {
        "camera(no top -> vil_camera)": rep["camera"]["pass_"],
        "root_cause(name2id raises)": rep["camera"]["pass_root_cause"],
        "dynhand init(pen)": rep["init"]["pass_"],
        "criterion vs manual": rep["criterion"]["pass_"],
        "start_episode(pen cells)": rep["start_episode"]["pass_"],
        "legacy top(task_name=None)": rep["legacy_top"]["pass_"],
        "hammer regression": rep["hammer"]["pass_"],
    }
    print(f"[report] {a.out}")
    print("[summary] " + ", ".join(f"{k}={'PASS' if v else 'FAIL'}"
                                   for k, v in checks.items()))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
