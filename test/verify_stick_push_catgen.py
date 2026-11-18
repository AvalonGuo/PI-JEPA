#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""stick-push catgen 自检 —— 只读探针(不训练 / 不跑 eval / 不读数据集)。

背景: catgen 原先只支持 assembly。给 stick-push 打开 catgen 有两个坑:
  (a) 基座必须是 sawyer_stick_obj.xml —— sawyer_xyz_env 把被操纵物硬编码在
      qpos[9:12]; 另有 stick freejoint(qpos[9:15]) 与 thermos 两个 slide
      (qpos[16:18]), 用 assembly 的 MJCF 会把复位写坏;
  (b) 该 XML 的可滑动 thermos body 与 5 类 catgen 资产的内层 body 都叫 "object"
      -> 直接注入 mj_loadXML 报 "repeated name in body array"。

本探针用**真实观测链**逐条验证(MetaWorldEnv + PointCloudGenerator: corner2 +
use_point_crop([-0.5,-1.5,-0.795]..[1,-0.4,100]) + FPS512 + pc_transform):

  [1] compose.mw_scene: assembly / stick-push 映射 + 未登记任务必须报错;
  [2] 6 类资产(5 catgen + sphere)在 stick_push 上合成并 mj_loadXML: nq=18 /
      nmocap=1 / jnt_qposadr 与基座逐值一致 / 注入内层 body 改名 dr_<obj>_object /
      无重复 body 名 / 基座 object(thermos) 名未被改动;
  [3] 复位不变性(base env vs composed env, 同一 numpy seed): 18 维 qpos 逐值相同,
      注入体 subtree 在复位姿态下无任何接触, 且基座接触数不增加;
  [4] 真实观测 toggle diff(composed - base): 全量点云与 FPS512 策略点云各算一次
      新增点簇(= 注入体), 断言点数 / 尺度(物体量级) / 在 crop 内留有余量;
      注入体 world body_xpos 必须恰好等于 P0 且停在桌面之上; 再用
      assembly@P0 cube 做对照(质心一致 => P0 / 相机 / crop 可迁移);
  [5] 净空: 注入体 subtree geom 与 stick / thermos subtree geom 的 rbound 表面下界;
  [6] catgen 计划与 task 无关: sample.sample_run(seed, n) ==
      Suite('catgen', 'stick-push', seed, n, family='metaworld').ep_types。

用法(在 /root/3DP 下):
    .venv/bin/python -u test/verify_stick_push_catgen.py           # 全量
    .venv/bin/python -u test/verify_stick_push_catgen.py --quick   # 只 cube, 跳过对照
输出: 终端报告 + /hy-tmp/3DP/vis/stick_push_catgen_probe.json
提示: EGL 多 render context 偶发 core dump(重跑即可); 加 -u 免得日志被缓冲吞掉。
"""
import argparse
import gc
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]          # /root/3DP
DP3 = REPO / "3D-Diffusion-Policy" / "diffusion_policy_3d"
BENCH = DP3 / "dr3d" / "bench"
for _p in (str(REPO), str(DP3), str(DP3 / "dr3d"), str(BENCH)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT_JSON = pathlib.Path("/hy-tmp/3DP/vis/stick_push_catgen_probe.json")
NUM_POINTS = 512                 # MetaworldRunner / DP3 eval 的部署 N
CATGEN_TYPES = ["cube", "can", "bottle", "cereal", "mug"]
EXTRA_TYPES = ["sphere"]         # dynsphere 的资产: 同样要能注入(重名冲突同类)
PARITY_TOL = 0.03                # 同 P0 下 stick-push vs assembly 可见簇质心差上限(m)
CLEAR_MIN = 0.15                 # 注入体与 stick/thermos 的表面净空下限(m)
CROP_LO = np.array([-0.5, -1.5, -0.795])   # MetaWorldEnv.use_point_crop 的 min_bound
CROP_HI = np.array([1.0, -0.4, 100.0])     #                             max_bound

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append({"name": name, "ok": bool(ok), "detail": str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    return bool(ok)


def build_env(task, model_xml_path, num_points=NUM_POINTS):
    """真实部署链: MetaWorldEnv(可指定 self-contained 场景 XML)。"""
    from diffusion_policy_3d.env.metaworld.metaworld_wrapper import MetaWorldEnv
    # MetaWorldEnv 期待 "cuda:<id>" 字符串(它做 device.split(":"))
    return MetaWorldEnv(task_name=task, device="cuda:0", use_point_crop=True,
                        num_points=int(num_points), model_xml_path=model_xml_path)


def reset_env(env, seed):
    """定种复位 -> (obs, qpos, sim)。metaworld 复位走 np.random, 显式 seed 即可复现。"""
    np.random.seed(int(seed))
    obs = env.reset()
    sim = env.env.sim
    return obs, sim.get_state().qpos.copy(), sim


def clouds(env):
    """(策略 FPS 点云, 全量点云) —— 都是 policy 坐标系(crop 之后)。"""
    pol, _ = env.get_point_cloud()
    full, _ = env.get_full_point_cloud()
    return (np.asarray(pol, dtype=float)[:, :3],
            np.asarray(full, dtype=float)[:, :3])


def policy_to_world(pts, transform):
    """get_point_cloud 里 pc <- pc @ transform.T, 故世界系 = pc @ inv(T).T。"""
    pts = np.asarray(pts, dtype=float)
    if transform is None:
        return pts
    return pts @ np.linalg.inv(np.asarray(transform, dtype=float)).T


def new_cluster(ref, pts, tol):
    """pts 中"距 ref 最近点 > tol"的掩码 = 新增几何(注入体)可见的那部分点。"""
    pts = np.asarray(pts, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if len(pts) == 0 or len(ref) == 0:
        return np.zeros(len(pts), dtype=bool)
    try:
        from scipy.spatial import cKDTree
        return cKDTree(ref).query(pts, k=1)[0] > float(tol)
    except Exception:                      # 无 scipy 时退化为分块暴力(限内存)
        out = np.zeros(len(pts), dtype=bool)
        t2 = float(tol) ** 2
        for i in range(0, len(pts), 64):
            blk = pts[i:i + 64]
            d2 = ((blk[:, None, :] - ref[None, :, :]) ** 2).sum(axis=-1)
            out[i:i + 64] = d2.min(axis=1) > t2
        return out


def model_of(xml_path):
    from mujoco_py import load_model_from_path
    return load_model_from_path(str(xml_path))


def body_names_of(m):
    """body 名列表(按下标)。注意: mujoco_py 的 `m.body_names` 少了未命名 body
    (stick-push 的 stick/thermos 各有一个未命名子 body), len() 与 m.nbody 不一致,
    必须走 body_id2name。"""
    return [str(m.body_id2name(i)) for i in range(m.nbody)]


def _subtree(m, root_id):
    """root_id 及其所有后代的 body 下标(geom 常常挂在未命名的子 body 上)。"""
    out = []
    for b in range(m.nbody):
        x = b
        while True:
            if x == root_id:
                out.append(b)
                break
            if x == 0:
                break
            x = m.body_parentid[x]
    return out


def geom_clearance(sim, targets=("stick", "object", "hand")):
    """注入体(dr_* body 子树)与指定 body 子树的 geom 的 rbound 表面下界(m)。

    用 rbound 和做下界: ||c1-c2|| - rb1 - rb2 <= 真实表面距离, 因此 >= 阈值即可断言
    净空; 数值小只说明"可能近", 需看具体值。
    """
    m = sim.model
    names = body_names_of(m)
    inj = [b for b in range(m.nbody) if names[b].startswith("dr_")]
    inj_bodies = set()
    for b in inj:
        inj_bodies.update(_subtree(m, b))
    inj_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in inj_bodies]
    out = {}
    for tb in targets:
        if tb not in names:
            continue
        tg = [g for g in range(m.ngeom)
              if m.geom_bodyid[g] in set(_subtree(m, m.body_name2id(tb)))]
        best = None
        for g in inj_geoms:
            for t in tg:
                d = (float(np.linalg.norm(sim.data.geom_xpos[g] - sim.data.geom_xpos[t]))
                     - float(m.geom_rbound[g]) - float(m.geom_rbound[t]))
                best = d if best is None else min(best, d)
        out[tb] = best
    return out


def step1_mapping(rep):
    print("\n[1] task -> (base, P0) 映射 (compose.mw_scene)", flush=True)
    import compose as C
    import test_load as TL
    asm_base, asm_p0 = C.mw_scene("assembly")
    sp_base, sp_p0 = C.mw_scene("stick-push")
    check("mw_scene('assembly')", (asm_base, asm_p0) == ("assembly", C.POS_ASSEMBLY),
          f"base={asm_base} P0={asm_p0}")
    check("mw_scene('stick-push')",
          sp_base == "stick_push" and sp_p0 == C.POS_ASSEMBLY == C.POS_STICK_PUSH,
          f"base={sp_base} P0={sp_p0} (复用 assembly 的 P0)")
    base_xml = TL.BASE_XML.get(sp_base)
    check("stick_push 基座已注册且文件存在",
          base_xml is not None and pathlib.Path(base_xml).exists(), str(base_xml))
    check("基座名 == sawyer_stick_obj.xml",
          str(base_xml).endswith("sawyer_xyz/sawyer_stick_obj.xml"),
          pathlib.Path(str(base_xml)).name)
    try:
        C.mw_scene("door-open")
        check("未登记任务必须报错", False, "mw_scene('door-open') 没有报错")
    except ValueError as e:
        check("未登记任务必须报错", "unsupported task" in str(e), str(e)[:90])
    rep["step1"] = {"assembly": [asm_base, asm_p0], "stick-push": [sp_base, sp_p0]}
    return sp_base, sp_p0


def step2_models(rep, base_key, p0, types):
    """合成(走生产缓存路径 ensure_scene) + MJCF/body 名检查。"""
    print(f"\n[2] 合成 + mj_loadXML (base={base_key}, P0='{p0}')", flush=True)
    import compose as C
    import test_load as TL
    base_xml = TL.BASE_XML[base_key]
    m_base = model_of(base_xml)
    n0, names0 = m_base.nbody, body_names_of(m_base)
    print(f"  base: nbody={n0} nq={m_base.nq} nmocap={m_base.nmocap} "
          f"bodies[:4]={names0[:4]}", flush=True)
    rows = {}
    for t in types:
        path, sha, fresh = C.ensure_scene(t, p0, base=base_key, free=False)
        _, sha2, fresh2 = C.ensure_scene(t, p0, base=base_key, free=False)
        m = model_of(path)
        names = body_names_of(m)
        inj_names = [b for b in names if b.startswith("dr_")]
        row = {
            "scene": path.name, "sha": sha[:16], "fresh": bool(fresh),
            "nbody": int(m.nbody), "nq": int(m.nq), "nmocap": int(m.nmocap),
            "n_body_delta": int(m.nbody - n0), "inj_bodies": inj_names,
        }
        print(f"  {t:7s} fresh={str(fresh):5s} nbody={m.nbody}(+{m.nbody - n0}) "
              f"nq={m.nq} nmocap={m.nmocap} inj={inj_names} sha={sha[:8]}", flush=True)
        check(f"{t}: 建模型后 nq 与基座一致({m_base.nq})", m.nq == m_base.nq,
              f"nq={m.nq}")
        check(f"{t}: nmocap 与基座一致({m_base.nmocap})", m.nmocap == m_base.nmocap,
              f"nmocap={m.nmocap}")
        check(f"{t}: 基座 body 树前缀未被改动",
              names[:n0] == names0, f"len={len(names)} n0={n0}")
        check(f"{t}: jnt_qposadr 与基座逐值一致",
              np.array_equal(np.asarray(m.jnt_qposadr), np.asarray(m_base.jnt_qposadr)),
              f"nq={m.nq}")
        named = [n for n in names if n != "None"]      # 未命名 body(如 thermos 子体)
        check(f"{t}: body 名无重复(未命名 body 除外)",
              len(named) == len(set(named)),
              f"dups={[x for x in set(named) if named.count(x) > 1]}")
        check(f"{t}: 内层 body 已改名 dr_{t}_object 且 thermos 的 object 保留",
              names.count("object") == 1 and f"dr_{t}_object" in names,
              f"inj={inj_names}")
        check(f"{t}: 缓存二次调用命中(不重算)", fresh2 is False and sha2 == sha,
              f"fresh2={fresh2}")
        rows[t] = row
    rep["step2"] = rows
    return rows


_BASE_CACHE = {}


def _inj_bodies(m):
    """注入体: 所有 dr_* body 及其子树(基座里也叫 dr_ 的名字不存在)。"""
    names = body_names_of(m)
    roots = [b for b in range(m.nbody) if names[b].startswith("dr_")]
    sub = set()
    for b in roots:
        sub.update(_subtree(m, b))
    geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in sub}
    return names, roots, sub, geoms


def _inj_contacts(sim, inj_geoms):
    """复位姿态下与注入体 geom 相关的接触数(必须为 0)。"""
    try:
        n = 0
        for i in range(int(sim.data.ncon)):
            c = sim.data.contact[i]
            if int(c.geom1) in inj_geoms or int(c.geom2) in inj_geoms:
                n += 1
        return n
    except Exception:                      # 拿不到 contact 就别假装通过
        return None


def _base_meas(task, base_xml, seed):
    """base 场景的复位/点云(每个 (task, xml, seed) 只建一次 env)。"""
    key = (task, str(base_xml), int(seed))
    if key not in _BASE_CACHE:
        env = build_env(task, str(base_xml))
        _, qpos, sim = reset_env(env, seed)
        pol, full = clouds(env)
        _BASE_CACHE[key] = {"env": env, "qpos": qpos, "pol": pol, "full": full,
                            "transform": env.pc_transform,
                            "ncon": int(sim.data.ncon)}
        print(f"  base[{task}]: qpos[:3]={np.round(qpos[:3], 5)} "
              f"ncon={sim.data.ncon} pol={pol.shape} full={full.shape}", flush=True)
    return _BASE_CACHE[key]


def _comp_meas(task, scene_xml, seed, type_name):
    env = build_env(task, str(scene_xml))
    _, qpos, sim = reset_env(env, seed)
    pol, full = clouds(env)
    m = sim.model
    names, roots, _sub, geoms = _inj_bodies(m)
    outer = m.body_name2id(f"dr_{type_name}")
    if outer < 0:
        outer = min(roots)
    out = {"qpos": qpos, "pol": pol, "full": full, "ncon": int(sim.data.ncon),
           "clearance": geom_clearance(sim),
           "inj_ncon": _inj_contacts(sim, geoms),
           "inj_world": np.array(sim.data.body_xpos[outer], dtype=float),
           "inj_names": [names[b] for b in roots]}
    env.close()
    del env
    gc.collect()
    return out


def assert_obs(label, bm, cm, p0, tol_full=0.01, tol_pol=0.02, require_pts=True,
               strict_clear=True):
    """[3][4][5] 的逐条判据(标签 -> 检查名)。返回该类型的观测量。

    世界系锚点用 sim 真值(注入体 body_xpos), 不用相机反投影: mujoco_point_cloud 的
    create_from_depth_image 平移用的是 body_pos 而不是 cam_pos, metaworld 的
    corner2 挂在 worldbody 上(body_pos=0), 所以 return 出来的点云是"绕相机转过的
    差向量"系 —— 该系下不能直接断言世界坐标(见 cam_geom 模块 docstring: ★恒等式只
    对 cam_pos==body_pos 的静态相机成立)。因此这里:
      * 世界系: 注入体 body_xpos == P0(严谨, 1mm) 且 z 落在桌面之上;
      * 策略系: 用"与 assembly 同 P0 的对照"证明相机/crop 响应一致, 用点簇尺度
        (pc_transform 是正交阵, 距离保持)证明这是一个物体量级的紧凑点簇。
    """
    p0_xy = np.array([float(v) for v in p0.split()])
    dq = float(np.abs(np.asarray(cm["qpos"]) - np.asarray(bm["qpos"])).max())
    check(f"{label}: 复位 qpos 与 base 逐值相同", dq < 1e-12, f"max|dq|={dq:g}")
    inj_ncon, ncon_b, ncon_c = cm["inj_ncon"], bm["ncon"], cm["ncon"]
    check(f"{label}: 复位姿态注入体无任何接触", inj_ncon == 0,
          f"inj_ncon={inj_ncon} (base ncon={ncon_b}, comp ncon={ncon_c})")
    check(f"{label}: 未新增基座接触", ncon_c <= ncon_b,
          f"base={ncon_b} comp={ncon_c}")
    clr = {k: (None if v is None else round(float(v), 4))
           for k, v in cm["clearance"].items()}
    ok_hard = all(cm["clearance"].get(b) is None or cm["clearance"][b] >= CLEAR_MIN
                  for b in ("stick", "object"))
    hand = cm["clearance"].get("hand")
    if strict_clear:
        check(f"{label}: 与 stick/thermos 表面净空 >= {CLEAR_MIN}m", ok_hard, str(clr))
        check(f"{label}: 与手(hand) 净空 > 0.05m", hand is None or hand > 0.05,
              f"hand={hand}")
    else:
        # 对照任务(assembly)侧没有 stick/thermos, 'object' 是被操纵的 RoundNut, 只记录
        print(f"    (对照: 净空仅记录 {clr})", flush=True)
    iw = np.asarray(cm["inj_world"], dtype=float)
    dxy = float(np.linalg.norm(iw[:2] - p0_xy))
    check(f"{label}: 注入体 world body_xpos xy == P0({p0}) (<1mm)", dxy < 1e-3,
          f"inj_world={np.round(iw, 4)} d_xy={dxy:g}")
    check(f"{label}: 注入体停在桌面之上(0.005 < z < 0.15m)", 0.005 < iw[2] < 0.15,
          f"z={iw[2]:.4f}")
    sel_f = new_cluster(bm["full"], cm["full"], tol_full)
    sel_p = new_cluster(bm["pol"], cm["pol"], tol_pol)
    blob = cm["full"][sel_f]
    cen = blob.mean(axis=0) if sel_f.any() else None
    cen_w = None if cen is None else policy_to_world(cen[None, :], bm["transform"])[0]
    ext = None if cen is None else float(np.linalg.norm(blob.max(0) - blob.min(0)))
    lost = int(new_cluster(cm["full"], bm["full"], tol_full).sum())
    n_pol = int(sel_p.sum())
    check(f"{label}: 注入体在策略视野内可见(全量点云出现新增点簇)",
          cen is not None, f"full_new={int(sel_f.sum())} full_lost={lost}")
    if cen is not None:
        mgn = float(min(np.min(cen - CROP_LO), np.min(CROP_HI - cen)))
        check(f"{label}: 可见簇质心在 crop 内留有余量(> 0.02m)", mgn > 0.02,
              f"centroid_pol={np.round(cen, 4)} margin={mgn:.3f}")
        check(f"{label}: 可见簇尺度是物体量级(0.02 < diag < 0.30m, "
              f"pc_transform 正交 => 距离保持)",
              ext is not None and 0.02 < ext < 0.30, f"extent={ext:.4f}m")
    if require_pts:
        check(f"{label}: 策略 FPS{NUM_POINTS} 点云含注入体点 (>=10)",
              n_pol >= 10, f"n_pol={n_pol}")
    print(f"    -> inj_world={np.round(iw, 4)} full_new={int(sel_f.sum())} "
          f"full_lost={lost} pol_new={n_pol} extent="
          f"{None if ext is None else round(ext, 4)} "
          f"centroid_pol={None if cen is None else np.round(cen, 4)}", flush=True)
    return {"qpos_maxdiff": dq, "ncon_base": ncon_b, "ncon_comp": ncon_c,
            "inj_ncon": inj_ncon, "inj_world": [float(v) for v in iw],
            "clearance": clr, "full_new": int(sel_f.sum()), "full_lost": lost,
            "pol_new": n_pol, "extent": ext,
            "centroid_pol": None if cen is None else [float(v) for v in cen],
            "centroid_raw": None if cen_w is None else [float(v) for v in cen_w]}


def step3_observe(rep, task, seed, types, require_pts=True):
    """base vs composed 的真实观测 toggle diff(每个类型重建一次 env)。"""
    import compose as C
    import test_load as TL
    base_key, p0 = C.mw_scene(task)
    base_xml = TL.BASE_XML[base_key]
    print(f"\n[3-5] 复位不变性 + toggle diff: task={task} base={base_key} "
          f"P0='{p0}' seed={seed}", flush=True)
    bm = _base_meas(task, base_xml, seed)
    rows = {}
    for t in types:
        scene, sha, _ = C.ensure_scene(t, p0, base=base_key, free=False)
        print(f"  {t:7s} scene={pathlib.Path(scene).name}", flush=True)
        cm = _comp_meas(task, scene, seed, t)
        rows[t] = assert_obs(f"{task}/{t}", bm, cm, p0,
                             require_pts=(require_pts and t in CATGEN_TYPES),
                             strict_clear=(task == "stick-push"))
    rep[f"obs_{task}"] = rows
    return rows


def step6_plan(rep, seed, n=20):
    print("\n[6] catgen 计划与 task 无关(sample_run == Suite.ep_types)", flush=True)
    import sample as _S
    import suite as _SU
    _, seq = _S.sample_run(seed, n)
    su = _SU.Suite("catgen", "stick-push", seed, n, family="metaworld", verbose=False)
    check("Suite('catgen','stick-push').ep_types == sample_run(seed, n)",
          list(su.ep_types) == list(seq), f"n={n} first6={list(seq)[:6]}")
    check("类型全在 catgen 资产集合内", set(seq) <= set(CATGEN_TYPES), sorted(set(seq)))
    # 生产路径: Suite._compose(默认 pos=None) 必须落到本 task 的基座 + mw_scene 的 P0
    import compose as C
    scene, _sha, _fresh = su._compose("cube")
    base_key, p0 = C.mw_scene("stick-push")
    name = pathlib.Path(scene).name
    check(f"Suite('catgen','stick-push')._compose('cube') 用 {base_key} 基座",
          f"_{base_key}_scene_" in name, name)
    check(f"Suite._compose 默认位置 = mw_scene P0 ('{p0}')",
          C._pos_tag(p0, False, 1.0, False)[1] in name, name)
    rep["step6"] = {"ep_types": [str(x) for x in seq], "catgen_scene": name,
                    "base": base_key, "P0": p0}
    return list(seq)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true",
                    help="只 cube(跳过其余 4 类与 assembly 对照)")
    ap.add_argument("--seed", type=int, default=0, help="复位定种(默认 0)")
    args = ap.parse_args()

    types = ["cube"] if args.quick else CATGEN_TYPES
    extra = [] if args.quick else EXTRA_TYPES
    rep = {"args": vars(args)}
    print("=== stick-push catgen 自检(只读探针) ===", flush=True)

    base_key, p0 = step1_mapping(rep)
    step2_models(rep, base_key, p0, types + extra)
    rows = step3_observe(rep, "stick-push", args.seed, types + extra)

    if not args.quick:
        arows = step3_observe(rep, "assembly", args.seed, ["cube"])
        c_sp, c_as = rows["cube"]["centroid_pol"], arows["cube"]["centroid_pol"]
        d = None if (c_sp is None or c_as is None) else float(
            np.linalg.norm(np.asarray(c_sp) - np.asarray(c_as)))
        check(f"P0/相机可迁移: stick-push 与 assembly cube@P0 可见簇质心差 "
              f"< {PARITY_TOL}m", d is not None and d < PARITY_TOL,
              f"d={None if d is None else round(d, 4)}m")
        w_sp = np.asarray(rows["cube"]["inj_world"], dtype=float)
        w_as = np.asarray(arows["cube"]["inj_world"], dtype=float)
        check("两任务注入体 world 位置一致(<1mm)",
              float(np.abs(w_sp - w_as).max()) < 1e-3, str(np.round(w_sp, 4)))

    step6_plan(rep, args.seed)

    ok = all(c["ok"] for c in CHECKS)
    rep["checks"] = CHECKS
    rep["n_pass"] = sum(1 for c in CHECKS if c["ok"])
    rep["n_fail"] = sum(1 for c in CHECKS if not c["ok"])
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rep, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\n=== {'全部通过' if ok else '存在 FAIL'}: "
          f"{rep['n_pass']}/{len(CHECKS)} 项通过 -> {OUT_JSON}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
