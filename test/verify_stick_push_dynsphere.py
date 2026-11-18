#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""stick-push dynsphere 自检 —— 只读探针(不训练 / 不跑 eval)。

口径(数据驱动 + 常量, 见 dr3d/bench/green_cells_metaworld.py + dynamic_sphere_mw.py):

    候选格 = 桌面格 ∩ in-crop ∩ ¬footprint                       (590, zarr depth)
    放置点 = 候选格 × 高度网格(0.02..0.80, h≥r)
             ∩ AABB 8 角点 ⊂ crop ∧ ⊂ 视锥                        (常量; 旧实现)
             ∩ ¬(AABB ∩ 数据 3D 占据)                             (数据; 新增)
             ∩ 屏幕圆盘不压任务物体(stick 两端盘 + thermos 盘)      (数据; 新增)
             ∩ 数据可见率 ≥ 0.5 且 复位帧可见                      (数据; 新增)

本探针独立复核:
  [1] 候选格 + 数据体积(3D 占据 / 屏幕足迹 / vis_frac): 形状/数量/与专家轨迹间距;
  [2] 逐半径: 可行放置点数量、diag 自洽(h≥r)、抽样独立重算 4 条判据全部通过;
  [3] 真实渲染: 每半径 3 个放置点 -> 写入 model.body_pos/geom_size -> MetaWorldEnv
      toggle diff: 球必须出现(≥阈值点)、qpos/nq/nmocap 不变、零接触(free 静态体);
  [4] 可复现性: 同 (seed, ep) 半径与放置序列逐值一致; 不同 ep 至少半径或位置不同;
  [5] 回归: assembly 仍走 legacy(table_grid 1125 / source='table' / 无过滤器)。

用法(在 /root/3DP 下):
    .venv/bin/python -u test/verify_stick_push_dynsphere.py           # 全量(15 次渲染)
    .venv/bin/python -u test/verify_stick_push_dynsphere.py --quick   # 每半径 1 个
    # 报告: /hy-tmp/3DP/vis/stick_push_dynsphere_probe.json
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

OUT_JSON = pathlib.Path("/hy-tmp/3DP/vis/stick_push_dynsphere_probe.json")
TASK = "stick-push"
EXPECT_CELLS = 590
# volumes follow the current zarr (regenerated 2026-10-30); verified identical to an
# independent recomputation of the same data (the pre-regen cache held 3512 / 1659)
EXPECT_VOX = 3450
EXPECT_SCREEN = 1591
MIN_FULL_PTS = 8          # 抽样渲染: 球至少贡献的全量点云点数
MIN_POL_PTS = 3           #                            FPS512 行数

import verify_stick_push_catgen as V   # noqa: E402
import compose as C                    # noqa: E402
import test_load as TL                 # noqa: E402
import suite as SU                     # noqa: E402
import green_cells_metaworld as G      # noqa: E402
import dynamic_sphere_mw as DSM        # noqa: E402
from scipy.spatial import cKDTree      # noqa: E402


def step1_data(rep):
    print("\n[1] 候选格 + 数据体积", flush=True)
    check = V.check
    cells = G.cells_for(TASK, "dynsphere")
    grid = G.desk_grid()
    Rcam, f, c = G._cam(G.base_xml(TASK))
    npz = np.load(G.cache_paths(TASK)[0])
    in_crop, blocked = npz["in_crop"], npz["blocked"]
    recomputed = grid[in_crop & ~blocked]
    d, _ = cKDTree(np.asarray(recomputed, float)).query(np.asarray(cells, float), k=1)
    check(f"候选格 = in-crop ∧ ¬footprint: n={len(cells)}", len(cells) == EXPECT_CELLS,
          f"n={len(cells)}")
    check("与缓存掩码重算逐格一致(<1e-6)", bool((d < 1e-6).all()), f"max_d={d.max():g}")
    check("cells_source == 'zarr:stick-push:dynsphere'",
          G.cells_source(TASK, "dynsphere") == "zarr:%s:dynsphere" % TASK,
          G.cells_source(TASK, "dynsphere"))

    vox, mask = G.dyn_volumes(TASK, verbose=False)
    check(f"3D 占据体素数 == {EXPECT_VOX}", int(vox["occ"].sum()) == EXPECT_VOX,
          f"n={int(vox['occ'].sum())}")
    check(f"屏幕足迹像素 == {EXPECT_SCREEN}", int(mask.sum()) == EXPECT_SCREEN,
          f"n={int(mask.sum())}")
    fr = vox["vis_frac"]
    check("vis_frac 形状 = (n_cells, n_heights, n_radii)",
          fr.shape == (len(cells), len(vox["heights"]), len(vox["radii"])),
          str(fr.shape))
    check("vis_reset 同形状且是 bool",
          vox["vis_reset"].shape == fr.shape and vox["vis_reset"].dtype == bool,
          str(vox["vis_reset"].shape))
    check("min_vis_frac == 0.5", float(vox["min_vis_frac"]) == 0.5,
          str(vox["min_vis_frac"]))

    av, am = G.dyn_volumes("assembly", verbose=False)
    check("assembly 仍 legacy(无数据体积)", av is None and am is None,
          f"vox={av} mask={am}")

    # 与专家 eef/stick/thermos 轨迹的最小间距
    import zarr
    Z = zarr.open(G._dr.mw_expert_zarr(TASK), "r")
    st = np.asarray(Z["data/state"])[:, :3]
    fs = np.asarray(Z["data/full_state"])
    traffic = np.vstack([st[:, :2], fs[:, 4:7][:, :2], fs[:, 11:14][:, :2]])
    dd, _ = cKDTree(traffic).query(np.asarray(cells, float), k=1)
    check("候选格与专家轨迹最小间距 >= 0.05m", dd.min() >= 0.05,
          f"min={dd.min():.3f} median={np.median(dd):.3f}")
    rep["step1"] = {"n_cells": int(len(cells)), "n_vox": int(vox["occ"].sum()),
                    "n_screen": int(mask.sum()), "vis_frac_shape": list(fr.shape),
                    "d_traffic_min": float(dd.min())}
    return cells


def _aabb_ok(xs, ys, hs, r, dyn):
    """独立重算 AABB 判据(8 角点 ⊂ crop ∧ ⊂ 视锥); hs 可为逐点数组。"""
    hs = np.asarray(hs, dtype=float)
    S = np.array([[-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
                  [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]], float) * r
    w = np.stack([xs[:, None] + S[None, :, 0], ys[:, None] + S[None, :, 1],
                  hs[:, None] + S[None, :, 2]], axis=2)
    cloud = w @ dyn.A + dyn.c
    crop = np.all(cloud >= dyn.cmi, axis=2) & np.all(cloud <= dyn.cma, axis=2)
    cam = (w - dyn.cam_pos) @ dyn.Rcam
    z = cam[:, :, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        depth = np.where(z < -1e-6, -z, np.nan)
        u = dyn.cx + dyn.f * cam[:, :, 0] / depth
        v = dyn.cx + dyn.f * cam[:, :, 1] / depth
    fr = (np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u <= dyn.img - 1.0) &
          (v >= 0) & (v <= dyn.img - 1.0))
    return np.all(crop & fr, axis=1)


def _hits_free(xs, ys, h, r, free):
    occ, lo, gs = free["occ"], free["lo"], free["gs"]
    nx, ny, nz = occ.shape
    iz0 = int(np.clip(np.floor((h - r - lo[2]) / gs), 0, nz - 1))
    iz1 = int(np.clip(np.floor((h + r - lo[2]) / gs), 0, nz - 1))
    out = np.zeros(len(xs), bool)
    for sx in (-r, 0.0, r):
        ix = np.clip(np.floor((xs + sx - lo[0]) / gs).astype(int), 0, nx - 1)
        for sy in (-r, 0.0, r):
            iy = np.clip(np.floor((ys + sy - lo[1]) / gs).astype(int), 0, ny - 1)
            out |= occ[ix, iy, iz0:iz1 + 1].any(axis=1)
    return out


def _screen_blocks(xs, ys, h, r, dyn):
    n = len(xs)
    p = (np.stack([xs, ys, np.full(n, float(h))], 1) - dyn.cam_pos) @ dyn.Rcam
    depth = -p[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        u = dyn.cx + dyn.f * p[:, 0] / depth
        v = dyn.cx + dyn.f * p[:, 1] / depth
        rpx = dyn.f * float(r) / np.maximum(depth, 1e-6) + dyn.screen_margin_px
    H, W = dyn.obj_dist.shape
    cu = np.clip(np.round(u).astype(int), 0, W - 1)
    cv = np.clip(np.round(v).astype(int), 0, H - 1)
    return dyn.obj_dist[cv, cu] <= rpx


def _vis_ok(xs, ys, h, r, dyn):
    """独立重算: 可见率 + 复位帧可见(按最近格/高度/半径索引)。"""
    i = np.argmin(np.hypot(dyn.cells[None, :, 0] - xs[:, None],
                           dyn.cells[None, :, 1] - ys[:, None]), axis=1)
    j = int(np.argmin(np.abs(dyn.vis["heights"] - float(h))))
    k = int(np.argmin(np.abs(dyn.vis["radii"] - float(r))))
    ok = dyn.vis["frac"][i, j, k] >= dyn.vis["min_frac"]
    if dyn.vis.get("reset") is not None:
        ok &= dyn.vis["reset"][i, j, k]
    return ok


def step2_placements(rep, dyn, n_sample=120):
    print("\n[2] 逐半径可行放置点 + 独立复核", flush=True)
    check = V.check
    rows = {}
    for r in dyn.r_set:
        n = dyn._build_placements(r)
        dg = dict(dyn.diag)
        fx, fy, fh = dyn._fx, dyn._fy, dyn._fh
        check(f"r={r:.2f}: 放置点 > 0", n > 0, f"n={n}")
        check(f"r={r:.2f}: diag 自洽(aabb-drops=final)",
              dg["n_aabb"] - dg["n_drop_free"] - dg["n_drop_screen"] -
              dg["n_drop_vis"] == dg["n_final"], str(dg))
        rng = np.random.RandomState(int(r * 1000))
        idx = rng.choice(len(fx), size=min(n_sample, len(fx)), replace=False)
        xs, ys, hs = fx[idx], fy[idx], fh[idx]
        ok_h = bool((hs >= r - 1e-9).all())
        ok_a = bool(_aabb_ok(xs, ys, hs, r, dyn).all()) if len(xs) else True
        ok_3 = bool((~_hits_free(xs, ys, hs[0], r, dyn.free)).all()) if len(xs) else True
        # 3D 判据逐点(高度可能不同)
        if len(xs):
            ok_3 = all(bool((~_hits_free(xs[i:i + 1], ys[i:i + 1], float(hs[i]), r,
                                         dyn.free)).all()) for i in range(len(xs)))
        ok_s = (all(bool((~_screen_blocks(xs[i:i + 1], ys[i:i + 1], float(hs[i]), r,
                                          dyn)).all()) for i in range(len(xs)))
                if len(xs) else True)
        ok_v = (all(bool(_vis_ok(xs[i:i + 1], ys[i:i + 1], float(hs[i]), r,
                                dyn).all()) for i in range(len(xs)))
                if len(xs) else True)
        check(f"r={r:.2f}: 抽样 {len(xs)} 点 h≥r", ok_h, f"min_h={hs.min():.3f}")
        check(f"r={r:.2f}: 抽样 AABB(crop∧视锥) 全过", ok_a, "")
        check(f"r={r:.2f}: 抽样 ¬3D 占据 全过", ok_3, "")
        check(f"r={r:.2f}: 抽样 屏幕不压物体 全过", ok_s, "")
        check(f"r={r:.2f}: 抽样 可见率+复位可见 全过", ok_v, "")
        rows[f"{r:.2f}"] = {"n": int(n), "diag": dg}
        print(f"   r={r:.2f}: n={n} diag={dg}", flush=True)
    rep["step2"] = rows


def step3_render(rep, dyn, env, bm, per_radius):
    print(f"\n[3] 真实渲染(每半径 {per_radius} 个放置点)", flush=True)
    check = V.check
    sim = env.env.sim
    qpos0 = sim.get_state().qpos.copy()
    nq0, nmocap0 = int(sim.model.nq), int(sim.model.nmocap)
    names, roots, sub, geoms = V._inj_bodies(sim.model)
    check("dr_sphere 是静态体(非 mocap)",
          int(sim.model.body_mocapid[dyn.body_id]) < 0,
          str(int(sim.model.body_mocapid[dyn.body_id])))
    rows = {}
    for r in dyn.r_set:
        n = dyn._build_placements(r)
        rng = np.random.RandomState(int(r * 1000) + 7)
        picks = [0] + [int(v) for v in rng.randint(0, n, size=max(0, per_radius - 1))]
        out = []
        for i in picks:
            x, y, h = float(dyn._fx[i]), float(dyn._fy[i]), float(dyn._fh[i])
            sim.model.body_pos[dyn.body_id] = [x, y, h]
            sim.model.geom_size[dyn.geom_id, 0] = r
            sim.forward()
            pol, full = V.clouds(env)
            nf = int(V.new_cluster(bm["full"], full, 0.01).sum())
            npl = int(V.new_cluster(bm["pol"], pol, 0.02).sum())
            ncon_inj = V._inj_contacts(sim, geoms)
            out.append({"pos": [round(x, 4), round(y, 4), round(h, 4)],
                        "full_new": nf, "pol_new": npl})
            check(f"r={r:.2f} @({x:+.2f},{y:+.2f},h={h:.2f}): 球出现在点云里",
                  nf >= MIN_FULL_PTS and npl >= MIN_POL_PTS,
                  f"full_new={nf} pol_new={npl}")
            check(f"r={r:.2f} @({x:+.2f},{y:+.2f},h={h:.2f}): 零接触 + qpos/nq/nmocap 不变",
                  ncon_inj == 0 and np.array_equal(sim.get_state().qpos, qpos0) and
                  int(sim.model.nq) == nq0 and int(sim.model.nmocap) == nmocap0,
                  f"inj_ncon={ncon_inj} nq={int(sim.model.nq)} nmocap={int(sim.model.nmocap)}")
        rows[f"{r:.2f}"] = out
    rep["step3"] = rows


def step4_determinism(rep, dyn):
    print("\n[4] 可复现性: mix(seed, ep, channel)", flush=True)
    check = V.check

    def run(seed, ep, k=6):
        dyn.start_episode(seed, ep)
        seq = [tuple(round(float(v), 6) for v in dyn.pos)]
        for _ in range(k):
            seq.append(tuple(round(float(v), 6) for v in dyn.plan()))
        return float(dyn.r), seq, list(dyn.history())

    r1, s1, h1 = run(0, 0)
    r2, s2, h2 = run(0, 0)
    r3, s3, h3 = run(0, 1)
    check("同 (seed,ep): 半径与放置序列逐值一致", (r1, s1) == (r2, s2), f"r={r1}")
    check("半径 ∈ R_SET", r1 in [float(v) for v in dyn.r_set], str(r1))
    check("不同 ep: 半径或序列不同", (r1, s1) != (r3, s3), f"ep0.r={r1} ep1.r={r3}")
    n = dyn._build_placements(r1)
    fx, fy, fh = dyn._fx, dyn._fy, dyn._fh
    bad = 0
    for (_k, x, y, h, _r) in h1:
        cand = np.where(np.hypot(fx - x, fy - y) <= 1e-6)[0]
        if not len(cand) or not bool((np.abs(fh[cand] - h) <= 1e-9).any()):
            bad += 1
    check("history 里每个放置点都在可行集合内", bad == 0,
          f"n={n} bad={bad}")
    rep["step4"] = {"r_ep0": r1, "r_ep1": r3, "seq_ep0": [list(v) for v in s1]}


def step5_regression(rep):
    print("\n[5] 回归: assembly dynsphere 仍 legacy", flush=True)
    check = V.check
    asm_cells = G.cells_for("assembly", "dynsphere")
    check("assembly dynsphere 候选格 = table_grid(1125)", len(asm_cells) == 1125,
          f"n={len(asm_cells)}")
    check("assembly cells_source == 'table'",
          G.cells_source("assembly", "dynsphere") == "table",
          G.cells_source("assembly", "dynsphere"))
    vox, mask = G.dyn_volumes("assembly", verbose=False)
    check("assembly dyn_volumes == (None, None)(无数据体积/过滤器)",
          vox is None and mask is None, f"vox={vox} mask={mask}")
    rep["step5"] = {"assembly_cells": int(len(asm_cells))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-radius", type=int, default=3,
                    help="每个半径渲染抽检的放置点数(默认 3)")
    ap.add_argument("--quick", action="store_true", help="每半径只渲染 1 个")
    args = ap.parse_args()
    per = 1 if args.quick else max(1, int(args.per_radius))
    rep = {"args": vars(args), "task": TASK}
    print("=== stick-push dynsphere 自检(只读探针) ===", flush=True)
    step1_data(rep)

    su = SU.Suite("dynsphere", TASK, 0, 4, family="metaworld", verbose=False)
    scene, _sha = C.compose_scene("sphere", C.POS_STICK_PUSH,
                                  out_dir=pathlib.Path("/tmp/dr3d_dyn_probe"),
                                  base="stick_push", free=False)
    env = V.build_env(TASK, str(scene))
    V.reset_env(env, 0)
    dyn = su._mw_distractor(env)
    step2_placements(rep, dyn)

    bm = V._base_meas(TASK, TL.BASE_XML["stick_push"], 0)
    step3_render(rep, dyn, env, bm, per)
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
