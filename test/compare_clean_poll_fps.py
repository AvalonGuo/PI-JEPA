#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""clean vs poll 的 FPS-512 名额竞争：逐点分类 + 可视化 + 量化。

背景：训练期 poll = sample_512(concat(raw, 伪造点块))。
当 raw+blob > 512 时 sample_512 走 torch3d FPS —— 它是**全局贪心**：加入一小团污染点会
挤掉一批真实点，于是 poll 与 clean(=sample_512(raw)=zarr.stored) 的差异**并非只有污染**。
本脚本把一帧的两个 512 视图按**索引**精确分类并画出来：

  🟢 真实-保留   clean ∩ poll
  🔵 真实-被换掉 clean \\ poll
  🟠 真实-新选入 poll \\ clean
  🔴 合成污染点  concat 中 raw 之后的那些行

用法（产物统一落 /hy-tmp，符合本仓库约定）：
  python test/compare_clean_poll_fps.py                         # door 默认 8 帧
  python test/compare_clean_poll_fps.py --frames 12 --seed 0
  python test/compare_clean_poll_fps.py --hammer_ref            # 追加 hammer 对照
  python test/compare_clean_poll_fps.py --no-plot               # 只打印统计
"""
import argparse
import json
import os
import pathlib
import sys

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[1]          # /root/3DP
_DP3 = _REPO / "3D-Diffusion-Policy"
for _p in (str(_DP3), str(_REPO), str(pathlib.Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from diffusion_policy_3d.common.pc_sample import sample_512          # noqa: E402
from diffusion_policy_3d.jepa.forge import forge_voxel_rows          # noqa: E402
from diffusion_policy_3d.voxel.workspace import resolve_workspace    # noqa: E402

DEFAULT_ZARR = "/hy-tmp/3DP/data/adroit_door_expert_fullpc.zarr"
DEFAULT_HAMMER_ZARR = "/hy-tmp/3DP/data/adroit_hammer_expert_fullpc.zarr"
DEFAULT_OUT = "/hy-tmp/3DP/vis/clean_vs_poll"
MATCH_TOL = 1e-5

COL = {"kept": (0.10, 0.70, 0.20),       # 绿
       "dropped": (0.15, 0.35, 0.95),    # 蓝
       "added": (1.00, 0.55, 0.05),      # 橙
       "forged": (0.90, 0.10, 0.10)}     # 红


# --------------------------------------------------------------------------
# FPS / 分类
# --------------------------------------------------------------------------
def fps_with_idx(pts, num_points=512):
    """与 ``sample_512`` 完全相同的 FPS/pad 逻辑，但额外返回被选中的行索引。

    N <= num_points -> 零填充（返回 idx = 0..N-1，其余无）；否则 torch3d FPS。
    返回 (out_pc (num_points,6), idx (K,) 被选中的原始行索引, branch)
    """
    pts = np.asarray(pts, dtype=np.float32)
    n = pts.shape[0]
    if n <= num_points:
        out = np.zeros((num_points, pts.shape[1]), dtype=np.float32)
        out[:n] = pts
        return out, np.arange(n, dtype=np.int64), "pad"
    import torch
    import pytorch3d.ops
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    xyz = torch.from_numpy(pts[:, :3].copy()).float().unsqueeze(0).to(dev)
    K = torch.tensor([num_points], dtype=torch.long, device=dev)
    _, inds = pytorch3d.ops.sample_farthest_points(points=xyz, K=K)
    idx = inds[0].cpu().numpy().astype(np.int64)
    return pts[idx], idx, "fps"


def _keys(pc):
    """用于集合比较：(x,y,z) 四舍五入到 MATCH_TOL → tuple。"""
    pc = np.asarray(pc, dtype=np.float64)
    q = np.round(pc[:, :3] / MATCH_TOL).astype(np.int64)
    return set(map(tuple, q))


def classify_frame(raw, stored, blob):
    """返回 (clean_pc, poll_pc, info)。info 内含四类点的点集与索引。"""
    clean, idx_c, br_c = fps_with_idx(raw, 512)          # = zarr.stored
    cat = np.concatenate([raw, blob], axis=0)
    poll, idx_p, br_p = fps_with_idx(cat, 512)
    n_raw = int(raw.shape[0])
    forged_rows = idx_p[idx_p >= n_raw]                  # poll 里的污染行
    real_rows = idx_p[idx_p < n_raw]                     # poll 里的真实行
    clean_real = idx_c[idx_c < n_raw]                    # clean 里的真实行（pad 时=全部）

    kept = sorted(set(real_rows.tolist()) & set(clean_real.tolist()))
    dropped = sorted(set(clean_real.tolist()) - set(kept))
    added = sorted(set(real_rows.tolist()) - set(kept))
    info = {
        "n_raw": n_raw, "n_blob": int(blob.shape[0]), "branch": br_p if br_p == "pad" else (
            "pad" if (n_raw + blob.shape[0]) <= 512 else "fps"),
        "branch_clean": br_c,
        "n_clean_real": int(clean_real.shape[0]),
        "n_poll_real": int(real_rows.shape[0]),
        "n_kept": len(kept), "n_dropped": len(dropped), "n_added": len(added),
        "n_forged_in_512": int(forged_rows.shape[0]),
        "n_pad_in_512": int(512 - real_rows.shape[0] - forged_rows.shape[0]),
        "retention": (len(kept) / max(1, clean_real.shape[0])),
        "clean_eq_stored": bool(np.allclose(clean, stored, atol=1e-4)),
        "pts": {"kept": raw[kept][:, :3] if kept else np.zeros((0, 3)),
                "dropped": raw[dropped][:, :3] if dropped else np.zeros((0, 3)),
                "added": raw[added][:, :3] if added else np.zeros((0, 3)),
                "forged": cat[forged_rows][:, :3] if len(forged_rows) else np.zeros((0, 3))},
    }
    return clean, poll, info


def load_frame(zarr_path, f):
    import zarr
    d = zarr.open(str(zarr_path), mode="r")["data"]
    k = int(np.asarray(d["raw_point_cloud_len"][f]))
    raw = np.asarray(d["raw_point_cloud"][f], np.float32).reshape(k, 6)
    stored = np.asarray(d["point_cloud"][f], np.float32)
    state = np.asarray(d["state"][f], np.float32)
    return raw, stored, state


# --------------------------------------------------------------------------
# 画图
# --------------------------------------------------------------------------
def _scatter3d(ax, pts, color, label, s=6, alpha=0.9):
    if pts is None or len(pts) == 0:
        return
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=s, c=[color], label=label,
               alpha=alpha, depthshade=False)


def _scatter2d(ax, pts, color, label, s=6, alpha=0.9):
    if pts is None or len(pts) == 0:
        return
    ax.scatter(pts[:, 0], pts[:, 1], s=s, c=[color], label=label, alpha=alpha)


def _limits(pts_list):
    allp = np.concatenate([p for p in pts_list if p is not None and len(p)], axis=0)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    c = (lo + hi) / 2.0
    r = max(float((hi - lo).max()) / 2.0, 1e-3)
    return c, r


def plot_frame(out_png, tag, f, info, ws=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P = info["pts"]
    c, r = _limits([P["kept"], P["dropped"], P["added"], P["forged"]])

    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    ax1.set_title(f"[clean] 512 ({info['n_clean_real']} real, branch={info['branch_clean']})")
    _scatter3d(ax1, P["kept"], COL["kept"], f"kept {info['n_kept']}")
    _scatter3d(ax1, P["dropped"], COL["dropped"], f"dropped {info['n_dropped']}")
    ax2 = fig.add_subplot(2, 3, 2, projection="3d")
    ax2.set_title(f"[poll] 512 ({info['n_poll_real']} real + "
                  f"{info['n_forged_in_512']} forged, branch={info['branch']})")
    _scatter3d(ax2, P["kept"], COL["kept"], f"kept {info['n_kept']}")
    _scatter3d(ax2, P["added"], COL["added"], f"added {info['n_added']}")
    _scatter3d(ax2, P["forged"], COL["forged"], f"forged {info['n_forged_in_512']}")
    ax3 = fig.add_subplot(2, 3, 3, projection="3d")
    ax3.set_title("poll: kept vs forged (only)")
    _scatter3d(ax3, P["kept"], COL["kept"], "kept")
    _scatter3d(ax3, P["forged"], COL["forged"], "forged")
    for ax in (ax1, ax2, ax3):
        ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.legend(loc="upper left", fontsize=7)

    ax4 = fig.add_subplot(2, 3, 4); ax4.set_title("poll top view (x-y)")
    _scatter2d(ax4, P["kept"], COL["kept"], "kept")
    _scatter2d(ax4, P["added"], COL["added"], "added")
    _scatter2d(ax4, P["forged"], COL["forged"], "forged", s=14)
    ax4.set_xlabel("x"); ax4.set_ylabel("y"); ax4.legend(fontsize=7)
    ax5 = fig.add_subplot(2, 3, 5); ax5.set_title("poll side view (x-z)")
    _scatter2d(ax5, P["kept"], COL["kept"], "kept")
    _scatter2d(ax5, P["added"], COL["added"], "added")
    _scatter2d(ax5, P["forged"], COL["forged"], "forged", s=14)
    ax5.set_xlabel("x"); ax5.set_ylabel("z"); ax5.legend(fontsize=7)

    if ws is not None and getattr(ws, "lo", None) is not None:
        for ax, i, j in ((ax4, 0, 1), (ax5, 0, 2)):
            x0, y0 = float(ws.lo[i]), float(ws.lo[j])
            x1, y1 = float(ws.hi[i]), float(ws.hi[j])
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                       ec="gray", ls="--", lw=0.8))
            ax.set_title(ax.get_title() + "  (dashed = workspace bbox)")

    ax6 = fig.add_subplot(2, 3, 6); ax6.axis("off")
    txt = [f"zarr frame {f}  ({tag})", "",
           f"raw N            = {info['n_raw']}",
           f"forged blob N    = {info['n_blob']}",
           f"branch           = {info['branch']}",
           f"clean real rows  = {info['n_clean_real']}",
           f"poll real rows   = {info['n_poll_real']}",
           f"forged in 512    = {info['n_forged_in_512']}",
           f"pad rows in 512  = {info['n_pad_in_512']}", "",
           f"kept    = {info['n_kept']}",
           f"dropped = {info['n_dropped']}  (churn)",
           f"added   = {info['n_added']}",
           f"retention = {info['retention'] * 100:.1f}%",
           f"clean == stored : {info['clean_eq_stored']}"]
    if "z" in info:
        z = info["z"]
        txt += ["", "z-space L1 (128-d):",
                f"  gap_enc  = {z['gap_enc']:.4f}",
                f"  eff_sel  = {z['eff_sel']:.4f}",
                f"  eff_blob = {z['eff_blob']:.4f}",
                f"  noop     = {z['noop']:.4f}",
                f"  pred     = {z['pred']:.4f}" + ("  OK" if z["pred"] < z["noop"]
                                                  else "  HARMFUL")]
    ax6.text(0.0, 0.98, "\n".join(txt), va="top", ha="left",
             family="monospace", fontsize=10)
    fig.suptitle("clean vs poll : FPS-512 名额竞争"
                 "（绿=真实保留 蓝=真实被换掉 橙=真实新选入 红=合成污染）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


def plot_summary(out_png, rows):
    """rows: list of dict(tag, frame, retention, churn, added, forged)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tags = sorted(set(r["tag"] for r in rows))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, key, title, ylab in ((axes[0], "retention", "clean 真实点被保留比例", "%"),
                                 (axes[1], "churn", "被换掉的真实点行数", "rows"),
                                 (axes[2], "forged", "进入 512 的污染点行数", "rows")):
        data = [[(r[key] * 100 if key == "retention" else r[key])
                 for r in rows if r["tag"] == t] for t in tags]
        ax.boxplot(data, labels=[t.replace("_", "\n") for t in tags], showmeans=True)
        ax.set_title(title); ax.set_ylabel(ylab); ax.grid(alpha=0.3)
    fig.suptitle("多帧统计（clean vs poll 的 FPS-512 名额竞争）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


def run_case(tag, zarr_path, task, args, out_dir, n_frames):
    import zarr
    ws = resolve_workspace(task=task, zarr=zarr_path, mode=args.workspace,
                           lo=args.workspace_lo, hi=args.workspace_hi)
    print(f"\n[{tag}] {zarr_path}")
    print(f"        {ws.describe()}")
    d = zarr.open(str(zarr_path), "r")["data"]
    T = int(d["raw_point_cloud_len"].shape[0])
    frames = sorted(np.random.default_rng(int(args.seed)).choice(
        T, size=min(n_frames, T), replace=False).tolist())
    rng_poll = np.random.default_rng(int(args.poll_seed))    # 与训练相同的伪造 rng
    rows = []
    for f in frames:
        raw, stored, _ = load_frame(zarr_path, f)
        blob = forge_voxel_rows(rng_poll, p_min=int(args.voxel_min),
                                p_max=int(args.voxel_max), ws=ws)
        clean, poll, info = classify_frame(raw, stored, blob)
        assert poll.shape == (512, 6) and clean.shape == (512, 6)
        assert info["n_kept"] + info["n_dropped"] == info["n_clean_real"]
        assert info["n_kept"] + info["n_added"] == info["n_poll_real"]
        assert info["n_poll_real"] + info["n_forged_in_512"] <= 512
        assert info["n_pad_in_512"] == 512 - info["n_poll_real"] - info["n_forged_in_512"]
        rows.append({"tag": tag, "frame": int(f), "retention": float(info["retention"]),
                     "churn": int(info["n_dropped"]), "added": int(info["n_added"]),
                     "forged": int(info["n_forged_in_512"])})
        msg = (f"  frame {f:5d}: raw={info['n_raw']:5d} blob={info['n_blob']:3d} "
               f"branch={info['branch']:3s} | clean_real={info['n_clean_real']:3d} "
               f"kept={info['n_kept']:3d} dropped={info['n_dropped']:3d} "
               f"added={info['n_added']:3d} forged_in_512={info['n_forged_in_512']:3d} "
               f"pad={info['n_pad_in_512']:3d} "
               f"retention={info['retention'] * 100:5.1f}% "
               f"(clean==stored:{info['clean_eq_stored']})")
        print(msg)
        if not args.no_plot:
            plot_frame(os.path.join(out_dir, f"{tag}_frame{f:05d}.png"),
                       tag, f, info, ws)
    ret = np.array([r["retention"] for r in rows])
    churn = np.array([r["churn"] for r in rows], dtype=float)
    forg = np.array([r["forged"] for r in rows], dtype=float)
    print(f"  ---- {tag} 汇总 ({len(rows)} 帧): 留存={ret.mean() * 100:.1f}%"
          f"±{ret.std() * 100:.1f} | 被换掉={churn.mean():.1f}±{churn.std():.1f} 行"
          f" | 污染进 512={forg.mean():.1f}±{forg.std():.1f} 行"
          f" | 被换掉/污染={churn.mean() / max(1e-9, forg.mean()):.1f}x")
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--zarr", default=DEFAULT_ZARR)
    ap.add_argument("--task", default="adroit_door")
    ap.add_argument("--frames", type=int, default=8, help="随机抽多少帧")
    ap.add_argument("--seed", type=int, default=0, help="抽帧 rng")
    ap.add_argument("--poll-seed", type=int, default=0, help="伪造 rng（与训练一致）")
    ap.add_argument("--voxel-min", type=int, default=10)
    ap.add_argument("--voxel-max", type=int, default=50)
    ap.add_argument("--workspace", choices=["auto", "hammer", "explicit"],
                    default="auto")
    ap.add_argument("--workspace-lo", nargs=3, type=float, default=None)
    ap.add_argument("--workspace-hi", nargs=3, type=float, default=None)
    ap.add_argument("--hammer-ref", action="store_true",
                    help="再加一组 hammer fullpc 对照（hammer 走 pad 分支, 无 FPS 重选）")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = run_case("door", a.zarr, a.task, a, a.out, a.frames)
    if a.hammer_ref:
        rows += run_case("hammer", DEFAULT_HAMMER_ZARR, "adroit_hammer", a,
                         a.out, a.frames)
    if not a.no_plot:
        p = plot_summary(os.path.join(a.out, "summary.png"), rows)
        meta = {"zarr": a.zarr, "task": a.task, "frames": a.frames,
                "poll_seed": a.poll_seed, "voxel_min": a.voxel_min,
                "voxel_max": a.voxel_max, "workspace": a.workspace,
                "rows": rows, "summary_png": p}
        with open(os.path.join(a.out, "summary.json"), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"\n[out] 图与统计 -> {a.out}")
    return rows


if __name__ == "__main__":
    main()




