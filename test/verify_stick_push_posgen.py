#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""stick-push posgen 自检 —— 只读探针(不训练 / 不跑 eval / 不读训练流程)。

绿区口径(数据驱动, 见 dr3d/bench/green_cells_metaworld.py):

    绿区 = 桌面格(常量) ∩ in-crop(常量) ∩ ¬dilate(zarr depth 足迹) ∩ visible@reset

本探针**独立复核**这套口径(不直接相信缓存), 并下沉到真实渲染:

  [1] 绿区: 375 格 / bbox / 每格 in-crop 且不在足迹内 且 visible@reset(从 zarr 重算),
      与缓存 npz 逐格一致; 与专家轨迹(eef / stick / thermos)的最小间距;
  [2] 真实渲染: 抽 8 格(x/y 极值 + 随机 4 格) -> cube@格 的 MetaWorldEnv toggle diff:
      >=10 行 FPS 点 / 质心在 crop 内留余量 / body_xpos==(x,y,0.032) /
      复位 qpos 与 base 逐位相同 / full_lost==0(不遮挡 stick/thermos);
  [3] 计划复现: Suite('posgen','stick-push',seed,n).ep_pos ==
      sample_green_pos(seed,n,cells), 位置 ∈ cells, n 前缀稳定;
  [4] 回归: assembly 仍是 legacy 解析口径(560 格, 基线不变); 数据版只作报告(不写缓存)。

用法(在 /root/3DP 下):
    .venv/bin/python -u test/verify_stick_push_posgen.py            # 全量(8 格渲染)
    .venv/bin/python -u test/verify_stick_push_posgen.py --quick    # 只 2 格渲染
    # 报告: /hy-tmp/3DP/vis/stick_push_posgen_probe.json
"""
import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]          # /root/3DP
DP3 = REPO / "3D-Diffusion-Policy" / "diffusion_policy_3d"
BENCH = DP3 / "dr3d" / "bench"
for _p in (str(REPO), str(DP3), str(DP3 / "dr3d"), str(BENCH), str(REPO / "test")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT_JSON = pathlib.Path("/hy-tmp/3DP/vis/stick_push_posgen_probe.json")
TASK = "stick-push"
EXPECT_N = 375                  # 数据驱动绿区格数(口径变则同步改)
CLEAR_MIN = 0.05                # 与专家 eef/stick/thermos 轨迹的最小间距下限(m)

import verify_stick_push_catgen as V   # noqa: E402  复用真实观测链/判据工具
import compose as C                    # noqa: E402
import test_load as TL                 # noqa: E402
import green_cells_metaworld as G      # noqa: E402
from scipy.spatial import cKDTree      # noqa: E402


def _set_equal(a, b, tol=1e-6):
    """两个 (M,2) 点集是否逐点相同(不考虑顺序)。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        return False
    d, _ = cKDTree(b).query(a, k=1)
    return bool((d < tol).all())


def step1_cells(rep):
    print("\n[1] 绿区复核(独立重算; 不信缓存)", flush=True)
    import zarr
    cells = G.load_cells(TASK, verbose=False)
    grid = G.desk_grid()
    Rcam, f, c = G._cam(G.base_xml(TASK))
    zpath = G._dr.mw_expert_zarr(TASK)
    in_crop = G.crop_mask(grid, Rcam, f, c)
    blocked, foot = G.footprint_mask(zpath, Rcam, f, c)
    visible, per = G.visible_at_reset(grid, zpath, Rcam, f, c)
    keep = in_crop & ~blocked & visible
    recomputed = grid[keep]

    check = V.check
    check(f"格数 == {EXPECT_N}", len(cells) == EXPECT_N, f"n={len(cells)}")
    check("缓存与独立重算逐格一致", _set_equal(cells, recomputed),
          f"n_cache={len(cells)} n_recompute={len(recomputed)}")
    check("diag: grid 1125 / in-crop 849", int(len(grid)) == 1125 and
          int(in_crop.sum()) == 849, f"grid={len(grid)} in_crop={int(in_crop.sum())}")
    check("diag: footprint(dilate1) 336 / visible 569",
          int(blocked.sum()) == 336 and int(visible.sum()) == 569,
          f"blocked={int(blocked.sum())} visible={int(visible.sum())}")
    check("每格都在 in-crop 内且不在足迹内",
          bool(in_crop[keep].all()) and bool((~blocked[keep]).all()) and
          bool(visible[keep].all()), "all(good)")
    check("全部 episode 首帧可见性一致(复位姿态确定)",
          bool((per == per[0]).all()), f"n_episodes={per.shape[0]}")

    # 与专家轨迹的最小间距(只作记录/下限断言)
    Z = zarr.open(zpath, "r")
    st = np.asarray(Z["data/state"])[:, :3]
    fs = np.asarray(Z["data/full_state"])
    traffic = np.vstack([st[:, :2], fs[:, 4:7][:, :2], fs[:, 11:14][:, :2]])
    d, _ = cKDTree(traffic).query(cells, k=1)
    check(f"与专家 eef/stick/thermos 轨迹最小间距 >= {CLEAR_MIN}m", d.min() >= CLEAR_MIN,
          f"min={d.min():.3f} p5={np.percentile(d, 5):.3f} median={np.median(d):.3f}")
    print(f"  cells bbox x[{cells[:,0].min():+.3f},{cells[:,0].max():+.3f}] "
          f"y[{cells[:,1].min():+.3f},{cells[:,1].max():+.3f}]", flush=True)
    rep["step1"] = {"n": int(len(cells)),
                    "bbox": [[float(cells[:, 0].min()), float(cells[:, 0].max())],
                             [float(cells[:, 1].min()), float(cells[:, 1].max())]],
                    "diag": {"grid": int(len(grid)), "in_crop": int(in_crop.sum()),
                             "footprint": int(foot.sum()), "blocked": int(blocked.sum()),
                             "visible": int(visible.sum())},
                    "d_traffic": {"min": float(d.min()),
                                  "p5": float(np.percentile(d, 5)),
                                  "median": float(np.median(d))}}
    return cells


def _pick_samples(cells, n, seed):
    """极值 4 格 + 随机 (n-4) 格(去重), 保证覆盖 x/y 两端。"""
    idx = [int(np.argmin(cells[:, 0])), int(np.argmax(cells[:, 0])),
           int(np.argmin(cells[:, 1])), int(np.argmax(cells[:, 1]))]
    rng = np.random.RandomState(int(seed) + 1234)
    if n > len(idx):
        extra = [i for i in rng.permutation(len(cells)) if i not in set(idx)]
        idx += [int(i) for i in extra[:n - len(idx)]]
    return cells[sorted(set(idx))[:max(n, 1)]]


def step2_render(rep, cells, n, seed):
    print(f"\n[2] 真实渲染 toggle diff({n} 个抽样格)", flush=True)
    bm = V._base_meas(TASK, TL.BASE_XML["stick_push"], 0)
    out = pathlib.Path("/tmp/dr3d_posgen_probe")
    rows = {}
    for xy in _pick_samples(cells, n, seed):
        pos = "%.4f %.4f" % (float(xy[0]), float(xy[1]))
        print(f"  cell=({pos})", flush=True)
        scene, sha = C.compose_scene("cube", pos, out_dir=out, base="stick_push",
                                     free=False)
        cm = V._comp_meas(TASK, scene, 0, "cube")
        rows[pos] = V.assert_obs(f"posgen@{pos}", bm, cm, pos,
                                 strict_clear=False, require_pts=True)
    rep["step2"] = rows
    return rows


def step3_plan(rep, cells, seed, n=20):
    print("\n[3] 计划复现(sample_green_pos == Suite.ep_pos)", flush=True)
    import sample as S
    import suite as SU
    su = SU.Suite("posgen", TASK, seed, n, family="metaworld", verbose=False)
    exp = [tuple(p) for p in S.sample_green_pos(seed, n, cells)]
    got = [tuple(p) for p in su.ep_pos]
    V.check("Suite.ep_pos == sample_green_pos(seed, n, cells)", got == exp, f"n={n}")
    V.check("Suite.cells 来自数据驱动绿区", len(su.cells) == len(cells) and
            _set_equal(su.cells, cells), f"n={len(su.cells)}")
    V.check("cells_source == 'zarr:stick-push'",
            su.cells_source == "zarr:%s" % TASK, str(su.cells_source))
    d, _ = cKDTree(np.asarray(cells, float)).query(np.asarray(exp, float), k=1)
    V.check("抽到的位置全部来自绿区(<1e-6)", bool((d < 1e-6).all()), f"max_d={d.max():g}")
    su5 = SU.Suite("posgen", TASK, seed, 5, family="metaworld", verbose=False)
    V.check("n 前缀稳定(n=5 是 n=20 的前缀)",
            [tuple(p) for p in su5.ep_pos] == got[:5],
            f"{[(round(a, 3), round(b, 3)) for a, b in got[:3]]} ...")
    # 生产路径: Suite._compose 必须把该位置落到 stick_push 基座 + 该位置的缓存名
    scene, _sha, _fresh = su._compose("cube", tuple(got[0]))
    tag = C._pos_tag("%.4f %.4f" % (got[0][0], got[0][1]), False, 1.0, False)[1]
    V.check("Suite._compose(pos[0]) 落在 stick_push 基座且位置戳匹配",
            "_stick_push_scene_" in pathlib.Path(scene).name and tag in pathlib.Path(scene).name,
            pathlib.Path(scene).name)
    rep["step3"] = {"ep_pos": [[float(a), float(b)] for a, b in got],
                    "cells_source": str(su.cells_source)}
    return got


def step4_regression(rep):
    print("\n[4] 回归: assembly 仍走 legacy(解析)口径", flush=True)
    asm = G.load_cells("assembly", verbose=False)
    V.check("assembly legacy 格数 = 560(基线不变)", len(asm) == 560, f"n={len(asm)}")
    V.check("assembly cells_source == 'legacy'",
            G.cells_source("assembly") == "legacy", G.cells_source("assembly"))
    bak = dict(G.TASKS["assembly"])
    try:
        G.TASKS["assembly"]["cells"] = "data"
        d = G.compute_cells("assembly")            # 只算, 不写缓存/不改默认
        n = len(d["cells"])
        V.check("assembly 数据版可算(仅报告, 默认不启用)", n >= 200, f"n={n}")
        rep["step4"] = {"assembly_legacy": int(len(asm)), "assembly_data_report": int(n)}
    finally:
        G.TASKS["assembly"].clear()
        G.TASKS["assembly"].update(bak)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0, help="posgen 抽样种子(默认 0)")
    ap.add_argument("--n", type=int, default=8, help="渲染抽检格数(默认 8)")
    ap.add_argument("--quick", action="store_true", help="只渲染 2 格")
    args = ap.parse_args()
    n = 2 if args.quick else max(2, int(args.n))
    rep = {"args": vars(args), "task": TASK}
    print("=== stick-push posgen 自检(只读探针) ===", flush=True)
    cells = step1_cells(rep)
    step2_render(rep, cells, n, args.seed)
    step3_plan(rep, cells, args.seed)
    step4_regression(rep)
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
