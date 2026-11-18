#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
z-TSNE: DP3 obs-encoder 特征 vs 灵巧手 state(真实闭环 eval 无障碍采集)
========================================================================
与 z-TSNE 分析管线相同(z 提取 + t-SNE), 但数据来自"真实 eval 回放"
(策略闭环 rollout, 干净场景), 等价于这条链路:

  bash scripts/eval_policy.sh dp3 adroit_hammer 0112 0 0 false 20 false
  (AdroitEnv -> MujocoPointcloudWrapperAdroit(fps/pad->512)
   -> MultiStepWrapper; 确定性 seed, 每 episode env.seed(seed+i))

差异: 本脚本的 env 栈去掉 SimpleVideoRecordingWrapper(仅录制, 不改 obs),
并在每个策略决策点(每 n_action_steps 子步)把窗口末帧的
(point_cloud(512,6), agent_pos(24)) + 两种 z 一起落盘, 供离线重画。

  * z_pn_max : obs_encoder 点云分支 max-pool 全局特征(256 维, 投影前)
  * z_obs_enc: obs_encoder 完整输出(0112 ckpt 实测 128 维/帧)

子命令:
  collect  跑真实闭环 eval(无障碍)并把每步观测+z 存成 npz + 自动出图
  tsne     只载入 npz 重新出图(可用于 clean_dataset 的 npz 对照)

输出(全部写 /hy-tmp):
  eval_rollout_z_embeddings.npz / eval_rollout_z_tsne_compare.png /
  eval_rollout_z_tsne_rho.png / eval_rollout_z_tsne_summary.json /
  eval_rollout_collect_meta.json

用法示例:
  # 采集 20 个干净场景 episode(等价 eval_policy.sh ... false 20 false);
  # 默认 success_goal_threshold=10 + extra_steps_after_success=15,
  # 即成功后再跑 15 个子步即停(与 VRL3 fullpc 演示采集同语义);
  # --every-step 逐子步存帧(--extra-steps-after-success 0 则关闭截断跑到底)
  python test/z_tsne_eval.py collect --ckpt <0112 latest.ckpt> --episodes 20 \
      --seed 0 --perplexity 30 [--every-step] [--catgen] [--posgen] \
      [--save-video]
  # --catgen: dr3d 类别泛化(固定位置 P0, 每 episode 随机 cube/can/bottle/cereal/mug;
  #           类型序列 = RandomState(seed).randint, 每 episode 换场景并重建 env)
  # --posgen: dr3d 位置泛化('cube' 每 episode 放在随机"可放置绿区"格;
  #           候选来自 dr3d/bench/green_cells.py, 位置 = RandomState(seed+7919),
  #           每 episode 换场景并重建 env; meta 里写 episode_positions)
  # --save-video: 每 episode 存 <out>/episode_XXX.mp4 (fps=10, 同 eval_policy.sh 规格)
  # 仅重画(可喂文件1 clean npz)
  python test/z_tsne_eval.py tsne --npz \
      /hy-tmp/3DP/z_tsne_out/eval_rollout_z_embeddings.npz
"""
import argparse
import json
import os
import pathlib
import sys

import numpy as np
import torch

_REPO = pathlib.Path(__file__).resolve().parents[1]       # /root/3DP
_DP3_DIR = _REPO / "3D-Diffusion-Policy"
_BENCH_DIR = _DP3_DIR / "diffusion_policy_3d" / "dr3d" / "bench"  # catgen suite
for _p in (str(_DP3_DIR), str(_REPO), str(pathlib.Path(__file__).resolve().parent),
           str(_BENCH_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_CKPT = ("/hy-tmp/data/outputs/adroit_hammer-dp3-0112_seed0/"
                "checkpoints/latest.ckpt")
DEFAULT_OUT = "/hy-tmp/3DP/z_tsne_out"
PN_MAX_DIM = 256
SOURCE_NAMES = ["pn_max", "obs_enc"]
NPZ_KEYS = ("z_pnmax", "z_obsenc", "state", "phase", "episode")


def load_model(ckpt_path, device):
    """Load the exact eval checkpoint (EMA weights + fitted normalizer stored
    in the state dict), same as eval.py::eval()'s checkpoint loading. The
    obs_encoder is a plain DP3Encoder (`obs_encoder.extractor.*` keys).
    """
    import dill
    import hydra
    from omegaconf import OmegaConf
    with open(str(ckpt_path), "rb") as f:
        payload = torch.load(f, pickle_module=dill, map_location="cpu",
                             weights_only=False)
    cfg = payload["cfg"]
    try:
        OmegaConf.register_new_resolver("eval", eval, replace=True)
    except ValueError:
        pass
    model = hydra.utils.instantiate(cfg.policy)
    key = "ema_model" if bool(cfg.training.use_ema) else "model"
    sd = payload["state_dicts"][key]
    model.load_state_dict(sd, strict=True)
    model.eval()
    model.to(device)
    return model, cfg


@torch.no_grad()
def encode_batch(model, pcs, states, device):
    """Normalize -> slice xyz -> obs_encoder, exactly like the policy path.
    Returns z_pn_max (M,256), z_obs_enc (M,Do)."""
    pc = torch.from_numpy(np.asarray(pcs, dtype=np.float32)).to(device)
    st = torch.from_numpy(np.asarray(states, dtype=np.float32)).to(device)
    nobs = model.normalizer.normalize({"point_cloud": pc, "agent_pos": st})
    nobs["point_cloud"] = nobs["point_cloud"][..., :3]
    z_obs_enc = model.obs_encoder(nobs)
    # 取 PointNet 分支的 mlp(max 输入): RepairEncoder 包装时先解包到 .enc
    base_enc = getattr(model.obs_encoder, "enc", model.obs_encoder)
    per_point = base_enc.extractor.mlp(nobs["point_cloud"])
    z_pn_max = torch.max(per_point, dim=1)[0]
    return z_pn_max.detach().cpu().numpy(), z_obs_enc.detach().cpu().numpy()


def _pick_device(arg):
    if arg:
        return torch.device(arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def standardize(a):
    a = np.asarray(a, dtype=np.float64)
    mu = a.mean(axis=0, keepdims=True)
    sd = a.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (a - mu) / sd


def spearman(a, b):
    """Rank correlation; returns 0.0 when degenerate (constant input)."""
    from scipy.stats import spearmanr
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size < 3 or b.size < 3:
        return 0.0
    if np.all(a == a[0]) or np.all(b == b[0]):
        return 0.0
    r, _ = spearmanr(a, b)
    return float(r) if np.isfinite(r) else 0.0


def pca2(z):
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=0).fit_transform(standardize(z))


def tsne2d(z, perplexity, seed):
    from sklearn.manifold import TSNE
    n = int(z.shape[0])
    ppx = int(max(2, min(int(perplexity), n - 1)))
    if n < 4:
        raise ValueError(f"need >=4 samples for TSNE, got {n}")
    return TSNE(n_components=2, perplexity=ppx, init="pca",
                learning_rate="auto", n_iter=1000,
                random_state=int(seed)).fit_transform(
                    np.asarray(z, dtype=np.float64))


def prepare_fields(state, phase, episode):
    from sklearn.decomposition import PCA
    spca = PCA(n_components=3, random_state=0).fit_transform(standardize(state))
    fields = {
        "state_pc1": spca[:, 0],
        "phase": np.asarray(phase, dtype=float),
        "episode": np.asarray(episode, dtype=float),
    }
    return fields, spca


def build_metrics(z_dict, state, phase, spca):
    std_state = standardize(state)
    rows = [f"statePC{i + 1}" for i in range(3)] + ["phase"]
    scalars = [spca[:, 0], spca[:, 1], spca[:, 2], np.asarray(phase, dtype=float)]
    mat, cols, top_state_dims = [], [], {}
    for src in SOURCE_NAMES:
        zp = pca2(z_dict[src])
        for c in range(2):
            cols.append(f"{src} zPC{c + 1}")
            mat.append([spearman(zp[:, c], s) for s in scalars])
        rhos = [spearman(zp[:, 0], std_state[:, j])
                for j in range(std_state.shape[1])]
        top_state_dims[src] = [int(j) for j in np.argsort(-np.abs(rhos))[:5]]
    M = np.asarray(mat, dtype=float).T
    return {"rows": rows, "cols": cols, "M": M,
            "top_state_dims": top_state_dims}


def plot_rho(M, rows, cols, path):
    fig, ax = plt.subplots(figsize=(9.5, 5))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=15, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="black" if abs(M[i, j]) < 0.6 else "white",
                    fontsize=8)
    ax.set_title("Spearman corr: z-PC1/2  vs  hand-state scalar "
                 "(state PC1..3 / phase)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_compare(emb, fields, path, suptitle):
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10))
    col_defs = [
        ("state_pc1", "coolwarm", "state PC1"),
        ("phase", "viridis", "phase in episode"),
        ("episode", "tab20", "episode id"),
    ]
    for r, src in enumerate(SOURCE_NAMES):
        y = emb[f"{src}_tsne"]
        dim = int(np.asarray(emb[f"{src}_z"]).shape[1])
        for c, (key, cmap, label) in enumerate(col_defs):
            ax = axes[r][c]
            sc = ax.scatter(y[:, 0], y[:, 1], c=fields[key], cmap=cmap,
                            s=8, alpha=0.9, edgecolors="none")
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
            cb.set_label(label)
            ax.set_title(f"{src} ({dim}d) | colored by {label}")
            ax.set_xlabel("TSNE-1")
            ax.set_ylabel("TSNE-2")
    axes[0][0].set_ylabel("pn_max   (max-pool 256d)", fontsize=11, rotation=0,
                          labelpad=120, va="center")
    axes[1][0].set_ylabel("obs_enc  (pn+state)", fontsize=11, rotation=0,
                          labelpad=140, va="center")
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _seed_all(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_video(frames, out_dir, episode_idx, fps=10):
    """(T, C, H, W) uint8 -> <out_dir>/episode_{idx:03d}.mp4。

    与 diffusion_policy_3d.env_runner.base_runner.save_episode_video 同规格
    (相同的 (T,C,H,W)->(T,H,W,C) 与 imageio fps 写法), 只是直接写到 <out_dir>,
    以对齐 eval_policy.sh 的 <run_dir>/media/<mode>/episode_XXX.mp4 布局。
    """
    import imageio
    from termcolor import cprint
    if frames is None or not isinstance(frames, np.ndarray):
        cprint(f"[WARN] episode {episode_idx} video frames unavailable, skip",
               "yellow")
        return None
    if frames.ndim == 5:
        frames = frames[:, 0]           # select first frame
    frames = np.transpose(frames, (0, 2, 3, 1))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"episode_{episode_idx:03d}.mp4")
    with imageio.get_writer(path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
    cprint(f"[z_tsne_eval] episode {episode_idx} video -> {path}", "magenta")
    return path


def collect(args):
    """Replicate the clean real-eval chain and record per-step obs+z.

    Episode 终止语义与 VRL3 fullpc 采集一致(默认):
      --success-goal-threshold 10  +  --extra-steps-after-success 15
      -> 累计 goal_achieved>=10 记 success_substep, 之后再执行 15 个子步即停,
         避免成功后超长静止尾帧; <=0 关闭截断(=跑到底的旧行为).
    This is an eval-style action: the user runs it (the agent only static-checks).
    """
    import random  # noqa: F401
    import time
    from termcolor import cprint

    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    device = _pick_device(args.device)
    print(f"[z_tsne_eval] load ckpt: {args.ckpt}  (device={device})")
    model, cfg = load_model(args.ckpt, device)
    obs_dim = int(model.obs_encoder.output_shape())
    print(f"[z_tsne_eval] obs_encoder output dim = {obs_dim} "
          f"(pn_max fixed = {PN_MAX_DIM})")

    # ---- fixed-quota observation (optional, --core-n) ---------------------------
    # The env wrapper must build the *same* observation the policy was trained on.
    core_n = (int(args.core_n) if getattr(args, "core_n", None) not in (None, "", "auto")
              else 0)
    if core_n:
        print(f"[z_tsne_eval] fixed quota: core_n={core_n} (from --core-n)")
    else:
        print("[z_tsne_eval] historical 512-row observation (core_n=0)")

    task = str(getattr(cfg.task, "task_name", "hammer"))
    er_cfg = getattr(cfg.task, "env_runner", None)
    n_obs_steps = int(getattr(cfg, "n_obs_steps", 2))
    n_action_steps = int(getattr(cfg, "n_action_steps", 8))
    max_steps = int(getattr(er_cfg, "max_steps", 200)) if er_cfg else 200
    use_point_crop = bool(getattr(er_cfg, "use_point_crop", True)) \
        if er_cfg else True
    catgen = bool(getattr(args, "catgen", False))
    posgen = bool(getattr(args, "posgen", False))
    dynsphere = bool(getattr(args, "dynsphere", False))
    dynhand = bool(getattr(args, "dynhand", False))
    save_video = bool(getattr(args, "save_video", False))
    video_dir = getattr(args, "video_dir", None) or args.out
    episodes = int(args.episodes)
    seed = int(args.seed)
    suite = ("catgen" if catgen else
             ("posgen" if posgen else ("dynsphere" if dynsphere else
                                       ("dynhand" if dynhand else None))))

    from diffusion_policy_3d.env import AdroitEnv  # noqa: E402
    from diffusion_policy_3d.gym_util.mjpc_diffusion_wrapper import (  # noqa: E402
        MujocoPointcloudWrapperAdroit)
    from diffusion_policy_3d.gym_util.multistep_wrapper import (  # noqa: E402
        MultiStepWrapper)
    if save_video:
        from diffusion_policy_3d.gym_util.video_recording_wrapper import (  # noqa: E402
            SimpleVideoRecordingWrapper)

    print(f"[z_tsne_eval] build env chain: task={task} "
          f"n_obs_steps={n_obs_steps} n_action_steps={n_action_steps} "
          f"max_steps={max_steps} use_point_crop={use_point_crop}"
          f" save_video={save_video}"
          + (f" [{suite} suite: per-episode scene swap]" if suite else ""))

    def make_env():
        # rebuilt per episode when the scene asset changes (catgen/posgen)
        inner = MujocoPointcloudWrapperAdroit(
            env=AdroitEnv(env_name=task, use_point_cloud=True),
            env_name="adroit_" + task, use_point_crop=use_point_crop,
            core_n=core_n)
        if save_video:
            # 与 eval_policy.sh(DP3 runner) 同款: 仅录制, 不改 obs
            inner = SimpleVideoRecordingWrapper(inner, steps_per_render=1)
        return MultiStepWrapper(
            inner,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_episode_steps=max_steps,
            reward_agg_method="sum",
        )

    env = make_env()

    # catgen/posgen: deterministic per-episode scene from the eval seed
    # (private scene dir cleanup is registered inside scene_runtime)
    ep_types = [None] * episodes
    ep_pos = [None] * episodes
    ep_refresh = [None] * episodes
    ep_radius = [None] * episodes
    ep_hand = [None] * episodes
    if suite:
        import sample as _bench_sample
        import suite as _SU
        _suite_obj = _SU.Suite(suite, task, seed, episodes)   # 本工具仅 adroit
        if suite == "catgen":
            _, ep_types = _bench_sample.sample_run(seed, episodes)
            ep_types = list(_suite_obj.ep_types)          # 同 seed -> 逐值一致
            print(f"[z_tsne_eval] catgen type sequence (seed={seed}): {ep_types}")
        elif suite == "posgen":                 # posgen: cube at random green cell
            ep_pos = [tuple(p) for p in _suite_obj.ep_pos]
            ep_types = ["cube"] * episodes
            print(f"[z_tsne_eval] posgen positions (seed={seed}, "
                  f"{len(_suite_obj.cells)} candidate cells): "
                  f"{[(round(x, 3), round(y, 3)) for x, y in ep_pos[:5]]} ...")
        elif suite == "dynsphere":              # 瞬现(可悬空)
            import dynamic as _dynlib
            ep_types = ["sphere"] * episodes
            print(f"[z_tsne_eval] dynsphere: kinematic sphere, episode-level SIZE, "
                  f"teleported every decision step (floating allowed, zero-contact) "
                  f"(seed={seed}, {len(_suite_obj.cells)} green cells, "
                  f"R_SET={_dynlib.R_SET})")
        else:                                   # dynhand: 逐子步平滑乒乓
            import dynamic_hand as _dh
            ep_types = ["inspire"] * episodes
            print(f"[z_tsne_eval] dynhand: kinematic inspire LEFT hand (FIXED open "
                  f"pose), smooth ping-pong inside the "
                  f"green cells every substep (zero-contact; seed={seed}, "
                  f"{len(_suite_obj.cells)} green cells, speed={_dh.SPEED} m/s, "
                  f"vis_frac_min={_dh.VIS_FRAC_MIN:.2f})")


    # deterministic eval, same as eval.py::eval()
    _seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    rec = []          # per-frame dicts
    success_rates = []
    threshold = float(args.success_goal_threshold)
    extra = int(args.extra_steps_after_success)
    cadence = not bool(args.every_step)
    print(f"[z_tsne_eval] success-stop: threshold={threshold:g} "
          f"extra_steps_after_success={extra} | record {'decision-cadence frames' if cadence else 'every substep frame'}")
    ep_meta = []
    for epi in range(episodes):
        ep_type = ep_types[epi]
        dyn = None
        if suite:
            _inf = _suite_obj.begin(epi, rebuild_env=lambda _p=None: make_env())
            env, dyn = _inf["env"], _inf["dyn"]
            if not QUIET:
                print(f"[z_tsne_eval] episode {epi:03d} {_inf['desc']}")
            if dyn is not None and suite == "dynsphere":
                ep_radius[epi] = float(dyn.r)
                if not QUIET:
                    print(f"[z_tsne_eval] episode {epi:03d} dynsphere start "
                          f"pos=({dyn.pos[0]:+.3f},{dyn.pos[1]:+.3f},z={dyn.pos[2]:.3f}) "
                          f"episode_radius={dyn.r:.2f} "
                          f"cells={len(dyn.cells_r)}/{len(dyn.cells)}")
            elif dyn is not None:                # dynhand
                ep_hand[epi] = dyn.spec()
                if not QUIET:
                    print(f"[z_tsne_eval] episode {epi:03d} dynhand start "
                          f"yaw={np.degrees(dyn.yaw):+.1f}deg z={dyn.z:.3f} "
                          f"A=({dyn.A[0]:+.3f},{dyn.A[1]:+.3f}) "
                          f"B=({dyn.B[0]:+.3f},{dyn.B[1]:+.3f}) path={dyn.L:.3f}m "
                          f"cells={len(dyn.cells_g)}(full={dyn.cells_full})/"
                          f"{len(dyn.cells)}")

        s = seed + epi
        _seed_all(s)
        try:
            env.seed(s)
        except Exception as e:  # noqa: BLE001
            cprint(f"[WARN] env.seed failed: {e}", "yellow")

        obs = env.reset()
        if dyn is not None:
            # 防御性: reset 之后再写一次 t=0 位形(首帧渲染在 reset 内部完成, 故
            # start_episode 的 reset 前写入仍是必需的; 这里兜底并暴露任何改写)
            _pe = dyn.reapply()
            print(f"[z_tsne_eval] episode {epi:03d} {suite} post-reset reapply "
                  f"pose_err={_pe:.2e}")
        if hasattr(model, "reset"):
            model.reset()

        executed = 0                      # env substeps completed
        act_chunk = None                  # (n_action_steps, Da) current action chunk
        j = 0                             # next row index inside act_chunk
        n_goal = 0.0                      # cumulative goal count (per substep)
        success_substep = -1              # executed count at first threshold crossing
        done = False
        last_info = None
        ep_rec = []

        def _frame(obs, t, act_part):
            np_obs = {k: np.asarray(v) for k, v in obs.items()}
            pc_cur = np_obs["point_cloud"][-1].astype(np.float32)     # (512,6)
            st_cur = np_obs["agent_pos"][-1].astype(np.float32)       # (24,)
            z_pn, z_ob = encode_batch(model, pc_cur[None], st_cur[None], device)
            return {"zp": z_pn[0], "zo": z_ob[0], "pc": pc_cur,
                    "st": st_cur, "t": int(t),
                    "act": np.asarray(act_part, dtype=np.float32)}

        while True:
            # --- post-success tail stop (VRL3: 循环顶部判断, 到上限即 break,
            #     不再记录/执行到达上限的那一帧) ---
            if extra > 0 and success_substep >= 0 and executed - success_substep >= extra:
                break

            # --- decision boundary: predict a fresh action chunk (every n_action_steps) ---
            if act_chunk is None or j >= act_chunk.shape[0]:
                if executed % n_action_steps != 0:
                    # previous chunk was cut short (done / tail stop): nothing left to run
                    break
                np_obs = {k: np.asarray(v) for k, v in obs.items()}
                with torch.no_grad():
                    obs_dict_input = {
                        "point_cloud": torch.from_numpy(
                            np_obs["point_cloud"]).unsqueeze(0).to(device),
                        "agent_pos": torch.from_numpy(
                            np_obs["agent_pos"]).unsqueeze(0).to(device),
                    }
                    action_dict = model.predict_action(obs_dict_input)
                act_chunk = action_dict["action"].detach().to("cpu").numpy().squeeze(0)
                j = 0
                if dyn is not None:
                    dyn.plan()      # 本决策步瞬现到新绿区位置(高度随机, 可悬空)
                if cadence:
                    ep_rec.append(_frame(obs, executed, act_chunk))

            # --- optional per-substep frame (state before executing row j) ---
            if not cadence:
                ep_rec.append(_frame(obs, executed, act_chunk[j:j + 1]))

            # --- 动态干扰物位形: dynhand 逐子步写 mocap_pos/quat + qpos;
            #     dynsphere 为"每决策步瞬现", tick 为空操作 ---
            if dyn is not None:
                dyn.tick(executed)
            # --- execute ONE env substep (env.step renders the cloud every substep) ---
            obs, reward, done, info = env.step(act_chunk[j:j + 1])

            done = bool(np.all(done))
            j += 1
            executed = len(env.get_rewards())
            # per-substep goal flag = newest frame of the returned n_obs_steps window
            ga = 0.0
            if "goal_achieved" in info and np.asarray(info["goal_achieved"]).size:
                ga = float(np.asarray(info["goal_achieved"]).ravel()[-1])
            n_goal += ga
            if success_substep < 0 and n_goal >= threshold:
                success_substep = int(executed)
            last_info = info
            if done:
                break

        if dyn is not None:
            dyn.finish()                 # 两种模式都是空操作(仅为 API 对称保留)
            ep_refresh[epi] = dyn.history()
            if suite == "dynhand":
                print(f"[z_tsne_eval] episode {epi:03d} dynhand ticks="
                      f"{len(dyn.records)} z={dyn.z:.3f} "
                      f"path={dyn.L:.3f}m pose_err={dyn.pose_err():.1e}")
            else:
                print(f"[z_tsne_eval] episode {epi:03d} dynsphere placements="
                      f"{len(dyn.records)} episode_radius={dyn.r:.2f}")


        total_t = max(1, executed)
        sr = float(np.mean(np.asarray(last_info["goal_achieved"], dtype=float))) \
            if last_info is not None and "goal_achieved" in last_info else 0.0
        success_rates.append(sr)
        tail_kept = int(executed - success_substep) if success_substep >= 0 else None
        epm = {"episode": int(epi), "type": ep_type,
               "pos": ([float(v) for v in ep_pos[epi]]
                       if ep_pos[epi] is not None else None),
               "radius": ep_radius[epi],
               "substeps": int(executed),
               "n_goal": float(n_goal),
               "success_substep": int(success_substep),
               "tail_kept": tail_kept, "success_rate": float(sr)}
        if ep_hand[epi] is not None:            # dynhand: episode 级手势/路径/高度
            epm["hand"] = ep_hand[epi]
        ep_meta.append(epm)

        drop_it = bool(args.drop_failed) and n_goal < threshold
        for r in ep_rec:
            r["ep"] = int(epi)
            r["T"] = int(executed)
            r["phase"] = float(r["t"]) / float(total_t)
            r["success"] = sr
            r["ep_type"] = ep_type if ep_type is not None else ""
            if not drop_it:
                rec.append(r)
        print(f"[z_tsne_eval] episode {epi:03d}: kept {len(ep_rec)} frames "
              f"(cadence={cadence}, dropped={drop_it}), T={total_t}, "
              f"n_goal={n_goal:.0f}, success@{success_substep}, "
              f"tail_kept={tail_kept}, success_rate={sr:.3f}")
        if save_video:
            # 必须在下一次 env.reset() 之前落盘(reset 会清空录制缓冲)
            _save_video(env.env.get_video(), video_dir, epi, fps=10)

    if suite:
        print(f"[z_tsne_eval] {suite}: private scene dir removed "
              f"(third_party untouched)")

    if not rec:
        raise SystemExit("[z_tsne_eval] no frames collected")

    z_pn = np.stack([r["zp"] for r in rec]).astype(np.float32)
    z_ob = np.stack([r["zo"] for r in rec]).astype(np.float32)
    state = np.stack([r["st"] for r in rec]).astype(np.float32)
    pc = np.stack([r["pc"] for r in rec]).astype(np.float32)
    ep = np.asarray([r["ep"] for r in rec], dtype=np.int64)
    phase = np.asarray([r["phase"] for r in rec], dtype=np.float32)
    tarr = np.asarray([r["t"] for r in rec], dtype=np.int64)
    act = np.stack([r["act"] for r in rec]).astype(np.float32)
    succ = np.asarray([r["success"] for r in rec], dtype=np.float32)
    ep_type_arr = np.asarray([r.get("ep_type", "") for r in rec], dtype="<U8")

    base = os.path.join(args.out, "eval_rollout_z_embeddings")
    np.savez_compressed(
        base + ".npz",
        z_pnmax=z_pn, z_obsenc=z_ob, state=state, pc=pc,
        phase=phase, episode=ep, substep_t=tarr, action=act, success=succ,
        episode_type=ep_type_arr)
    meta = {
        "script": "test/z_tsne_eval.py collect",
        "ckpt": args.ckpt,
        "task": task,
        "device": str(device),
        "n_episodes": int(episodes), "seed": seed,
        "max_steps": int(max_steps),
        "n_obs_steps": int(n_obs_steps),
        "n_action_steps": int(n_action_steps),
        "n_frames": int(len(rec)),
        "obs_dim": int(obs_dim),
        "pn_max_dim": int(PN_MAX_DIM),
        "success_goal_threshold": float(threshold),
        "extra_steps_after_success": int(extra),
        "record_cadence": bool(cadence),
        "episodes_detail": ep_meta,
        "success_rates": [float(s) for s in success_rates],
        "mean_success_rate": float(np.mean(success_rates)) \
            if success_rates else 0.0,
        "elapsed_s": round(time.time() - t0, 1),
    }
    if suite:
        meta["suite"] = suite
        meta["episode_types"] = [m.get("type") for m in ep_meta]
        # 与 DP3 的 dr3d_suite_manifest.json 对齐: dict 形式, key = episode_XXX
        meta["episode_positions"] = {
            f"episode_{m['episode']:03d}": m.get("pos") for m in ep_meta}
        if suite == "dynsphere":
            # episode 级半径(整段 episode 不变)
            meta["episode_radius"] = {
                f"episode_{i:03d}": r for i, r in enumerate(ep_radius)}
            # 每决策步放置记录: [[k, x, y, h, r], ...] (h>=r 不穿桌, 球完整可见)
            meta["episode_refresh"] = {
                f"episode_{i:03d}": h for i, h in enumerate(ep_refresh) if h}
        if suite == "dynhand":
            # episode 级路径/高度/yaw 规格 + 逐子步位形日志
            meta["episode_hand_specs"] = {
                f"episode_{i:03d}": h for i, h in enumerate(ep_hand) if h}
            meta["episode_refresh"] = {
                f"episode_{i:03d}": h for i, h in enumerate(ep_refresh) if h}

        if suite == "catgen":
            per_type = {}
            for m in ep_meta:
                t = m.get("type")
                if t is None:
                    continue
                d = per_type.setdefault(t, {"n": 0, "sum_sr": 0.0})
                d["n"] += 1
                d["sum_sr"] += float(m["success_rate"])
            meta["per_type_success"] = {
                t: {"n": d["n"], "mean_success_rate": d["sum_sr"] / max(1, d["n"])}
                for t, d in per_type.items()}
    with open(os.path.join(args.out, "eval_rollout_collect_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[z_tsne_eval] mean_success_rate={meta['mean_success_rate']:.3f} "
          f"({len(rec)} frames)  saved -> {base}.npz")
    if not args.no_plot:
        run_tsne_from_npz(base + ".npz", args.out, args.perplexity,
                          args.tsne_seed, meta)
    return meta


def run_tsne_from_npz(npz_path, out_dir, perplexity, seed, meta=None):
    """Load saved embeddings (schema shared with the z-TSNE collector) and run
    the TSNE compare figure + rho heatmap + summary json."""
    import time
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    d = np.load(str(npz_path))
    missing = [k for k in NPZ_KEYS if k not in d]
    if missing:
        raise SystemExit(f"[z_tsne_eval] npz {npz_path} missing keys {missing}")
    z_pnmax = np.asarray(d["z_pnmax"])
    z_obsenc = np.asarray(d["z_obsenc"])
    state = np.asarray(d["state"])
    phase = np.asarray(d["phase"])
    episode = np.asarray(d["episode"])
    n = int(len(z_pnmax))
    print(f"[z_tsne_eval] tsne on {n} frames: pn_max={z_pnmax.shape[1]}d "
          f"obs_enc={z_obsenc.shape[1]}d  (npz={npz_path})")
    if n < 4:
        raise SystemExit(f"need >=4 frames for TSNE, got {n}")
    z_dict = {"pn_max": z_pnmax, "obs_enc": z_obsenc}
    fields, spca = prepare_fields(state, phase, episode)
    emb = {}
    for src in SOURCE_NAMES:
        emb[f"{src}_z"] = z_dict[src]
        emb[f"{src}_tsne"] = tsne2d(z_dict[src], perplexity, seed)
    base = os.path.join(out_dir, "eval_rollout_z_tsne")
    plot_compare(
        emb, fields, base + "_compare.png",
        "Real-eval rollout z-TSNE (clean scene, DP3 0112) | "
        "colored by dexterous-hand state")
    met = build_metrics(z_dict, np.asarray(state, dtype=np.float64),
                        phase, spca)
    M = met["M"]
    plot_rho(M, met["rows"], met["cols"], base + "_rho.png")
    extra = dict(meta or {})
    if "success" in d:
        extra["mean_success_rate"] = float(np.mean(np.asarray(d["success"])))
    summary = {
        "script": "test/z_tsne_eval.py tsne",
        "npz": str(npz_path),
        "n_frames": n,
        "z_dims": {"pn_max": int(z_pnmax.shape[1]),
                    "obs_enc": int(z_obsenc.shape[1])},
        "tsne": {"perplexity": int(perplexity), "seed": int(seed),
                 "init": "pca", "n_iter": 1000},
        "rho": {"rows": met["rows"], "cols": met["cols"],
                "M": [[float(v) for v in row] for row in np.asarray(M)]},
        "top_state_dims_by_zPC1": met["top_state_dims"],
        "meta": extra,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(base + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    for src in SOURCE_NAMES:
        c = SOURCE_NAMES.index(src) * 2
        print(f"[z_tsne_eval] rho(statePC1, {src}-zPC1) = {M[0, c]:+.3f}")
    print(f"[z_tsne_eval] saved -> {base}_compare.png / _rho.png / "
          f"_summary.json")
    return summary


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("collect", help="rollout (clean eval) + record")
    pc.add_argument("--ckpt", default=DEFAULT_CKPT)
    pc.add_argument("--device", default=None)
    pc.add_argument("--core-n", default=None,
                    help="env 的点云真实行先 FPS 到 N 再补零(固定名额); "
                         "缺省/0 = 历史 512 行路径(例如 DP3 基线)")
    pc.add_argument("--seed", type=int, default=0)
    pc.add_argument("--episodes", type=int, default=20)
    pc.add_argument("--success-goal-threshold", type=float, default=10.0,
                    help="cumulative per-substep goal_achieved to mark success "
                         "(VRL3 fullpc 采集同款默认 10)")
    pc.add_argument("--extra-steps-after-success", type=int, default=15,
                    help="成功后仅再多跑 N 个子步即停(镜像 VRL3, 默认 15; "
                         "<=0 = 关闭截断跑到底)")
    pc.add_argument("--every-step", action="store_true",
                    help="逐子步存帧(默认只存策略决策点帧, 与旧 schema 兼容)")
    pc.add_argument("--drop-failed", action="store_true",
                    help="丢弃 n_goal<threshold 的失败 episode 帧(默认保留)")
    pc.add_argument("--catgen", action="store_true",
                    help="dr3d category-generalization suite: per-episode random "
                         "distractor type (cube/can/bottle/cereal/mug) at the fixed "
                         "P0 position; type sequence = RandomState(seed).randint, "
                         "scene swapped + env rebuilt per episode")
    pc.add_argument("--posgen", action="store_true",
                    help="dr3d position-generalization suite: 'cube' at a random "
                         "GREEN (placeable) table cell each episode; candidates from "
                         "dr3d/bench/green_cells.py, positions = "
                         "RandomState(seed+7919); scene swapped + env rebuilt per "
                         "episode")
    pc.add_argument("--dynsphere", action="store_true",
                    help="dr3d dynamic-distractor suite: a KINEMATIC (mocap) sphere "
                         "whose radius is drawn ONCE PER EPISODE from "
                         "R_SET={0.06..0.02} and which is TELEPORTED (no sliding, "
                         "possibly FLOATING) to a new GREEN (placeable) cell at "
                         "EVERY decision step (n_action_steps substeps). It has "
                         "contype=conaffinity=0 (zero physical interaction): only "
                         "the point cloud / rendered image changes. Constraints: "
                         "z>=r (never sinks through the table) and the sphere stays "
                         "fully visible to the top camera, so far cells are "
                         "rejected for large radii. Samples derive from "
                         "mix(seed, episode, k) -> fully reproducible; scene "
                         "swapped + env rebuilt per episode")
    pc.add_argument("--dynhand", action="store_true",
                    help="dr3d dynamic-distractor suite #4: a KINEMATIC (mocap) inspire "
                         "LEFT hand (12 finger hinges incl. a 4-joint thumb) held in a "
                         "FIXED open pose; the hand PING-PONGS smoothly along a "
                         "segment between two GREEN (placeable) cells (phase = pure "
                         "function of the substep index, so it is bit-reproducible). "
                         "Height z is drawn once per episode within [Z_MIN, z_cap] where "
                         "z_cap keeps >=70%% of the hand surface inside the depth image "
                         "('mostly in frame'); base orientation = palm down / fingers +x "
                         "plus a per-episode random yaw. All geoms are "
                         "contype=conaffinity=0 (zero physical interaction): only the "
                         "point cloud / rendered image changes. Adds 11 qpos/v and keeps "
                         "the 24-dim agent_pos untouched. Scene swapped + env rebuilt per "
                         "episode")
    pc.add_argument("--save-video", action="store_true",

                    help="record one mp4 per episode: <out>/episode_XXX.mp4 (fps=10), "
                         "same specs as eval_policy.sh's video chain")
    pc.add_argument("--video-dir", default=None,
                    help="video output dir (default: <out>, i.e. <run_dir>/media/<mode>)")
    pc.add_argument("--out", default=DEFAULT_OUT)
    pc.add_argument("--perplexity", type=int, default=30)
    pc.add_argument("--tsne-seed", type=int, default=42)
    pc.add_argument("--no-plot", action="store_true")
    pc.set_defaults(func=lambda a: collect(a))

    pt = sub.add_parser("tsne", help="re-plot from a saved npz")
    pt.add_argument("--npz", required=True)
    pt.add_argument("--out", default=DEFAULT_OUT)
    pt.add_argument("--perplexity", type=int, default=30)
    pt.add_argument("--seed", type=int, default=42)
    pt.set_defaults(func=lambda a: run_tsne_from_npz(
        a.npz, a.out, a.perplexity, a.seed))

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
