#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""stick-push dynhand 自检 —— 只读探针(不训练 / 不跑 eval)。

口径(数据驱动 + 常量, 见 dr3d/bench/green_cells_metaworld.py + dynamic_hand_mw.py):

    候选格 = 桌面格 ∩ in-crop ∩ ¬footprint                        (590; 与 dynsphere 同源)
    安全格 = 候选格 ∩ dist_to_footprint ≥ hr + margin(0.03)        (手半宽 hr≈0.086)
    A→B    = 安全格内取对, 段上 201 采样点 dist ≥ hr + margin 且 L ≥ 0.405 m
    朝向   = 中轴指向 TASK_CENTER = 数据质心 (0.243, 0.595)
    高度   = U(z_lo=0.12, min(0.40, 路径各点都可见的最高 z))       (常量 + 解析可见性)
    运动   = 逐 substep 沿 A→B 三角波乒乓(0.9 m/s, dt=timestep×5), 零接触

本探针独立复核:
  [1] 数据: 候选格/`work_dist`(到足迹距离场)/`task_center`; 每个安全格 `dist-margin≥hr`;
      对照"assembly 常量口径"的泄漏格数(记录为 baseline 缺陷);
  [2] episode: (seed,ep) 各抽一次 -> L≥min_path / zone_ok / table_ok / path_vis_min≥vis_min /
      z∈[0.12,0.40] / 中轴 201 点独立重算 dist ≥ hr+margin / 手体净空 ≥ 0;
  [3] 真实渲染: 3 个 substep(tick) -> 手出现在点云里 / 零手接触 / 机械臂 qpos[0:9] 不变 /
      nq=30 / 关节顺序(thermos slide 16,17; 手 18..29) / nmocap=1;
  [4] 可复现: 同 (seed,ep) A/B/z/yaw 与逐 substep 位置逐位一致; 异 ep 不同;
  [5] 回归: assembly dynhand 仍 legacy(候选格 = table_grid 1125, work=None,
      task_center=(0.05, 0.70))。

用法(在 /root/3DP 下):
    .venv/bin/python -u test/verify_stick_push_dynhand.py
    # 报告: /hy-tmp/3DP/vis/stick_push_dynhand_probe.json
"""
import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
DP3 = REPO / "3D-Diffusion-Policy" / "diffusion_policy_3d"
BENCH = DP3 / "dr3d" / "bench"
for _p in (str(REPO), str(DP3), str(DP3 / "dr3d"), str(BENCH), str(REPO / "test")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT_JSON = pathlib.Path("/hy-tmp/3DP/vis/stick_push_dynhand_probe.json")
TASK = "stick-push"
EXPECT_CELLS = 590
EXPECT_TCENTER = (0.243, 0.595)      # 数据质心(stick+thermos com 均值)
MIN_HAND_PTS = 8                     # 抽样渲染: 手至少贡献的全量点数
MIN_POL_PTS = 5

import verify_stick_push_catgen as V   # noqa: E402
import compose as C                    # noqa: E402
import test_load as TL                 # noqa: E402
import suite as SU                     # noqa: E402
import green_cells_metaworld as G      # noqa: E402
import green_cells_assembly as GA      # noqa: E402  (仅作"旧口径"对照)
from dynamic_hand import MIN_PATH_ABS, MIN_PATH_TIME, SPEED, Z_RANGE   # noqa: E402


def _old_zone(xy, hr):
    """旧口径(assembly 常量)的"进工作区"判据(仅用于量化泄漏)。"""
    mg = float(GA._ARM["margin"]) + float(hr)
    ax1 = float(GA._ARM["x"][1])
    ay0, ay1 = float(GA._ARM["y"][0]), float(GA._ARM["y"][1])
    z = ((np.abs(xy[:, 0]) <= ax1 + mg) & (xy[:, 1] >= ay0 - mg) &
         (xy[:, 1] <= ay1 + mg))
    for (ox, oy) in GA._OBJ:
        z |= (np.hypot(xy[:, 0] - ox, xy[:, 1] - oy) < float(GA._OBJ_R) + float(hr))
    return z


def step1_data(rep, dyn):
    print("\n[1] 候选格 + 工作区距离场 + 任务中心", flush=True)
    check = V.check
    cells = np.asarray(dyn.cells, float)
    check(f"候选格 = in-crop ∧ ¬footprint: n={len(cells)}",
          len(cells) == EXPECT_CELLS, f"n={len(cells)}")
    w = dyn.work
    check("work 来自 zarr 足迹(不是 assembly 常量)", w is not None,
          "work=zarr-footprint" if w is not None else "assembly-const")
    check("work_dist 形状 = (ny,nx) 且 gs=0.03",
          w["dist"].ndim == 2 and abs(float(w["gs"]) - 0.03) < 1e-9,
          f"shape={w['dist'].shape} gs={w['gs']}")
    check(f"task_center ≈ {EXPECT_TCENTER}",
          np.allclose(dyn.task_center, EXPECT_TCENTER, atol=2e-3),
          str(np.round(dyn.task_center, 3)))
    hr = float(dyn.hr)
    marg = float(w["margin"])
    d = dyn._dist_raw(cells)                     # (590,)
    safe = d >= hr + marg - 1e-9
    check(f"安全格(dist ≥ hr+{marg:.2f}) = {int(safe.sum())}", int(safe.sum()) > 300,
          f"n={int(safe.sum())}/{len(cells)}")
    check("安全格逐格满足 dist-margin ≥ hr",
          bool((d[safe] - marg >= hr - 1e-6).all()), f"min={float((d[safe]-marg).min()):.4f}")
    leak = (~_old_zone(cells, hr)) & (d - marg < hr)     # 旧口径放行但实际进区
    old_safe = int((~_old_zone(cells, hr)).sum())
    print(f"  [对照] 旧口径(assembly 常量)放行 {old_safe}/{len(cells)} 格, "
          f"其中 {int(leak.sum())} 格其实压进数据足迹(<hr)", flush=True)
    check("旧口径确实会泄漏(>=20 格) —— 记录为 baseline 缺陷",
          int(leak.sum()) >= 20, f"leak={int(leak.sum())}")
    rep["step1"] = {"n_cells": int(len(cells)), "n_safe": int(safe.sum()),
                    "hr": hr, "margin": marg,
                    "dist_min": float(d.min()), "dist_max": float(d.max()),
                    "task_center": [float(v) for v in dyn.task_center],
                    "old_safe": old_safe, "old_leak": int(leak.sum())}


def step2_episodes(rep, dyn):
    print("\n[2] episode 可行性(A/B / z / 可见性 / 工作区审计)", flush=True)
    check = V.check
    hr = float(dyn.hr)
    marg = float(dyn.work["margin"])
    min_path = float(dyn.min_path)
    u = np.linspace(0.0, 1.0, 201)[:, None]
    rows = {}
    for seed, ep in ((0, 0), (0, 1), (1, 0)):
        dyn.start_episode(seed, ep)
        key = "s%de%d" % (seed, ep)
        A, B = dyn.A.copy(), dyn.B.copy()
        seg = A[None, :] + (B - A)[None, :] * u
        check(f"{key}: L ≥ min_path({min_path:.3f}m)", dyn.L >= min_path - 1e-9,
              f"L={dyn.L:.3f}")
        check(f"{key}: zone_ok ∧ table_ok",
              bool(dyn.zone_ok) and bool(dyn.table_ok),
              f"clear={dyn.work_clear_min:.3f} t_ok={dyn.table_ok}")
        check(f"{key}: 中轴 201 点独立重算 dist-margin ≥ hr",
              bool((dyn._dist_raw(seg) - marg >= hr - 1e-6).all()),
              f"min={float((dyn._dist_raw(seg)-marg).min()):.4f}")
        check(f"{key}: 手体净空(center ∧ body) ≥ 0",
              float(dyn.center_clear_min) >= 0.0 and float(dyn.body_clear_min) >= 0.0,
              f"center={dyn.center_clear_min:.3f} body={dyn.body_clear_min:.3f}")
        check(f"{key}: path_vis_min ≥ vis_min({dyn.vis_min:.2f})",
              float(dyn.path_vis_min) >= float(dyn.vis_min) - 1e-9,
              f"vis={dyn.path_vis_min:.2f}")
        check(f"{key}: z ∈ [0.12, {Z_RANGE[1]:.2f}]",
              0.12 - 1e-9 <= float(dyn.z) <= Z_RANGE[1] + 1e-9, f"z={dyn.z:.3f}")
        check(f"{key}: 安全格 ≥ 50", len(dyn.cells_g) >= 50,
              f"cells_g={len(dyn.cells_g)}")
        rows[key] = {"A": [float(v) for v in A], "B": [float(v) for v in B],
                     "L": float(dyn.L), "z": float(dyn.z), "yaw": float(dyn.yaw),
                     "vis": float(dyn.path_vis_min),
                     "clear": float(dyn.work_clear_min),
                     "cells_g": int(len(dyn.cells_g))}
    rep["step2"] = rows


def _hand_contacts(sim, names):
    n = 0
    for i in range(int(sim.data.ncon)):
        c = sim.data.contact[i]
        bn = [names[sim.model.geom_bodyid[int(c.geom1)]],
              names[sim.model.geom_bodyid[int(c.geom2)]]]
        if any(str(b).startswith(("l_", "left_hand")) for b in bn):
            n += 1
    return n


def step3_render(rep, dyn, env, bm, ts=(0, 10, 20)):
    print(f"\n[3] 真实渲染(ep=0, substep {list(ts)})", flush=True)
    check = V.check
    sim = env.env.sim
    m = sim.model
    names = V.body_names_of(m)
    check("nq == 30(18 + 12 手指 hinge)", int(m.nq) == 30, f"nq={int(m.nq)}")
    check("nmocap == 1(手是静态体)", int(m.nmocap) == 1, f"nmocap={int(m.nmocap)}")
    adr = {m.joint_id2name(j): int(m.jnt_qposadr[j]) for j in range(m.njnt)}
    check("thermos slide 仍在 qpos[16,17]",
          adr.get("obj_slidex") == 16 and adr.get("obj_slidey") == 17,
          str({k: adr[k] for k in ("obj_slidex", "obj_slidey") if k in adr}))
    hand_adr = sorted(v for k, v in adr.items()
                      if k is not None and str(k).startswith("joint_l_"))
    check("手指 hinge 追加在 qpos[18..29]", hand_adr == list(range(18, 30)),
          str(hand_adr))
    check("agent_pos 仍 9 维", env.get_robot_state().shape == (9,),
          str(env.get_robot_state().shape))
    dyn.start_episode(0, 0)
    q_arm = sim.get_state().qpos[:9].copy()
    out = []
    for t in ts:
        dyn.tick(int(t))
        pol, full = V.clouds(env)
        nf = int(V.new_cluster(bm["full"], full, 0.01).sum())
        npl = int(V.new_cluster(bm["pol"], pol, 0.02).sum())
        nc = _hand_contacts(sim, names)
        q_now = sim.get_state().qpos
        check(f"t={t}: 手出现在策略点云里", nf >= MIN_HAND_PTS and npl >= MIN_POL_PTS,
              f"full_new={nf} pol_new={npl}")
        check(f"t={t}: 零手接触", nc == 0, f"hand_contacts={nc}")
        check(f"t={t}: 机械臂 qpos[0:9] 不变", np.array_equal(q_now[:9], q_arm),
              f"max|dq|={float(np.abs(q_now[:9]-q_arm).max()):g}")
        check(f"t={t}: 手姿态未被改写(pose_err≈0)", float(dyn.pose_err()) <= 1e-9,
              f"pose_err={float(dyn.pose_err()):.2e}")
        out.append({"t": int(t), "pos": [round(float(v), 4) for v in dyn.pos],
                    "full_new": nf, "pol_new": npl, "hand_contacts": nc})
    rep["step3"] = out


def step4_determinism(rep, dyn):
    print("\n[4] 可复现性: A/B/z/yaw + 逐 substep 位置", flush=True)
    check = V.check

    def run(seed, ep, n=6):
        dyn.start_episode(seed, ep)
        key = (tuple(np.round(dyn.A, 9)), tuple(np.round(dyn.B, 9)),
               round(float(dyn.z), 9), round(float(dyn.yaw), 9))
        seq = [tuple(np.round(dyn.pos, 9))]
        for t in range(1, n + 1):
            dyn.tick(t)
            seq.append(tuple(np.round(dyn.pos, 9)))
        return key, seq

    k1, s1 = run(0, 0)
    k2, s2 = run(0, 0)
    k3, s3 = run(0, 1)
    check("同 (seed,ep): A/B/z/yaw 逐值一致", k1 == k2,
          f"z={round(k1[2], 3)} yaw={round(k1[3], 3)}")
    check("同 (seed,ep): 逐 substep 位置逐值一致", s1 == s2, f"n={len(s1)}")
    check("不同 ep: A/B/z/yaw 或轨迹不同", (k1, s1) != (k3, s3), "")
    span = max(float(np.hypot(a[0] - s1[0][0], a[1] - s1[0][1])) for a in s1)
    check("手在 substep 上确实移动(span > 0.05m)", span > 0.05, f"span={span:.3f}")
    rep["step4"] = {"A": list(k1[0]), "B": list(k1[1]), "z": k1[2],
                    "yaw": k1[3], "span": span,
                    "substeps": [list(v) for v in s1]}


def step5_regression(rep):
    print("\n[5] 回归: assembly dynhand 仍 legacy", flush=True)
    check = V.check
    asm = G.cells_for("assembly", "dynhand")
    check("assembly dynhand 候选格 = table_grid(1125)", len(asm) == 1125,
          f"n={len(asm)}")
    check("assembly cells_source == 'table'",
          G.cells_source("assembly", "dynhand") == "table",
          G.cells_source("assembly", "dynhand"))
    vox, mask = G.dyn_volumes("assembly", verbose=False)
    check("assembly dyn_volumes == (None, None)(work 走 legacy 常量)",
          vox is None and mask is None, f"vox={vox} mask={mask}")
    rep["step5"] = {"assembly_cells": int(len(asm))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--substeps", default="0,10,20",
                    help="渲染抽检的 substep 列表(默认 0,10,20)")
    args = ap.parse_args()
    ts = tuple(int(v) for v in str(args.substeps).split(",") if v.strip())
    rep = {"args": vars(args), "task": TASK}
    print("=== stick-push dynhand 自检(只读探针) ===", flush=True)

    su = SU.Suite("dynhand", TASK, 0, 3, family="metaworld", verbose=False)
    scene, _sha = C.compose_scene("inspire", C.POS_STICK_PUSH,
                                  out_dir=pathlib.Path("/tmp/dr3d_hand_probe"),
                                  base="stick_push", free=False)
    env = V.build_env(TASK, str(scene))
    V.reset_env(env, 0)
    dyn = su._mw_distractor(env)
    V.check("cells_source == 'zarr:stick-push:dynhand'",
            su.cells_source == "zarr:%s:dynhand" % TASK, su.cells_source)
    print(f"  cells={len(su.cells)} source={su.cells_source} "
          f"task_center={np.round(dyn.task_center, 3).tolist()}", flush=True)

    step1_data(rep, dyn)
    step2_episodes(rep, dyn)
    bm = V._base_meas(TASK, TL.BASE_XML["stick_push"], 0)
    step3_render(rep, dyn, env, bm, ts)
    step4_determinism(rep, dyn)
    step5_regression(rep)

    ok = all(c["ok"] for c in V.CHECKS)
    rep["checks"] = V.CHECKS
    rep["n_pass"] = sum(1 for c in V.CHECKS if c["ok"])
    rep["n_fail"] = sum(1 for c in V.CHECKS if not c["ok"])
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== {'全部通过' if ok else '存在 FAIL'}: "
          f"{rep['n_pass']}/{len(V.CHECKS)} 项通过 -> {OUT_JSON}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
