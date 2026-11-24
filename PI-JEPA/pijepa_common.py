#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared PI-JEPA training library: two-view data, losses, cfg, ckpt, rollout.

Consumers: ``PI-JEPA/train_pijepa.py`` (v5 trainer) and
``test/verify_pijepa_pipeline.py``.  See ``PI-JEPA/pijepa_policy.py``.

Data (``CleanPollDataset``): two views of the *first* ``n_obs_steps`` frames
(that is all the encoder consumes)::

    clean : data/point_cloud[f]                              (FPS(raw, N))
    poll  : sample_512(concat(raw[f], forge_voxel_rows(...)))  (voxel pollution clutter)

Per step (single backward, one shared noised action / timestep)::

    s   = StateMLP(agent_pos)                                  # once, both views
    z_c = Pool(BottleNeck(MLP(pc_clean), s)) -> UNet(final_proj(z_c) | s)
    z_p = Pool(BottleNeck(MLP(pc_poll),  s)) -> UNet(final_proj(z_p) | s)
    loss = l2_clean_weight * L2_clean + l2_poll_weight * L2_poll
           + lambda_jepa_clean * SmoothL1(z_c, sg(t))
           + lambda_jepa_poll  * SmoothL1(z_p, sg(t))
    t = Pool(MLP(pc_clean).detach())          # pre-bottleneck clean feature

``--lambda-jepa-clean`` / ``--lambda-jepa-poll`` both default to 1 and only accept
0 or 1; 0 switches that view's SmoothL1 term off.

The bottleneck's last layer is zero-initialised, so the network is
DP3-identical at step 0.

Deploy = the same shared branch (``PIJEPAEncoder.forward``) on the env cloud, run
from the EMA copy: ``obs -> mlp -> bottleneck -> softmax pool -> final_proj|state
-> UNet``.

In-loop eval (``--rollout-every``) also drives the best-checkpoint logic
(``TopKCheckpointManager``, monitor ``cfg.checkpoint.topk.monitor_key`` =
``test_mean_score``): whenever the score improves ``checkpoints/best.ckpt`` is
overwritten in place (tmp + rename), and ``checkpoints/best.json`` records
{monitor, mode, score, epoch, file}.  Best ckpts skip the optimizer state (half the
size; ``eval.py`` ignores it anyway).

Everything lives in ``PI-JEPA/`` on purpose: repo core (train.py / DP3) is
untouched.  The checkpoint payload matches ``train.py`` (``cfg`` +
``state_dicts.{model,ema_model,optimizer}`` + ``pickles.{epoch,global_step}``) so
``eval.py`` can load it.
"""
import argparse
import copy
import gc
import inspect
import json
import math
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F
from einops import reduce as _reduce

_REPO = pathlib.Path(__file__).resolve().parents[1]          # /root/3DP
_PKG = pathlib.Path(__file__).resolve().parent               # /root/3DP/PI-JEPA
for _p in (str(_REPO / "3D-Diffusion-Policy"), str(_REPO), str(_PKG)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dill  # noqa: E402
import hydra  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from diffusion_policy_3d.common.pc_sample import sample_512  # noqa: E402
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer  # noqa: E402
from diffusion_policy_3d.common.sampler import (  # noqa: E402
    SequenceSampler, downsample_mask)
from diffusion_policy_3d.jepa.forge import forge_voxel_rows  # noqa: E402

CFG_DIR = str(_REPO / "3D-Diffusion-Policy" / "diffusion_policy_3d" / "config")



def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", required=True, help="hydra task config name")
    p.add_argument("--zarr", default=None,
                   help="fullpc zarr (needs data/raw_point_cloud); default "
                        "/hy-tmp/3DP/data/<task>_expert.zarr")
    p.add_argument("--out", default=None,
                   help="run dir; default /hy-tmp/data/outputs/<exp>_seed<seed>")
    p.add_argument("--exp-name", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=None, help="default: cfg.optimizer.lr")
    p.add_argument("--max-steps", type=int, default=0,
                   help=">0 stops after N optimizer steps (smoke test)")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every-epoch", action="store_true")
    p.add_argument("--no-ckpt", action="store_true")
    p.add_argument("--resume", action="store_true")
    # losses: SmoothL1 view-consistency weights (1 = on, 0 = off; no other values)
    p.add_argument("--lambda-jepa-clean", type=float, default=1.0,
                   help="weight of SmoothL1(z_clean, sg(t)); 1 = on, 0 = off")
    p.add_argument("--lambda-jepa-poll", type=float, default=1.0,
                   help="weight of SmoothL1(z_poll, sg(t)); 1 = on, 0 = off")
    p.add_argument("--l2-clean-weight", type=float, default=1.0)
    p.add_argument("--l2-poll-weight", type=float, default=1.0)
    # bottleneck
    p.add_argument("--bn-hidden", default="128,64,128",
                   help="PIJEPA-BottleNeck widths, comma separated "
                        "(320->h1->h2->h3->256)")
    # pollution (voxel-blob definition)
    p.add_argument("--poll-voxel-min", type=int, default=10)
    p.add_argument("--poll-voxel-max", type=int, default=50)
    p.add_argument("--poll-mode", choices=("bbox", "occupied"), default="bbox")
    p.add_argument("--poll-seed", type=int, default=0)
    p.add_argument("--num-points", type=int, default=0, help="0 = zarr N")
    p.add_argument("--fps-device", choices=("auto", "cpu", "cuda"), default="cpu",
                   help="FPS device for the polluted view; default 'cpu' keeps the "
                        "DataLoader workers parallel (num_workers default 8); 'auto' "
                        "follows the eval-time point_cloud_sampling() device, which "
                        "forces num_workers=0 (CUDA is unsafe in forked workers)")
    p.add_argument("--workspace", choices=("auto", "hammer", "explicit"), default="auto")
    # encoder / policy overrides
    p.add_argument("--pool-type", default="softmax",
                   help="softmax (default) | mean | max")
    p.add_argument("--pool-temp", type=float, default=1.0)
    p.add_argument("--pool-score-hidden", type=int, default=0)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--n-obs-steps", type=int, default=None)
    p.add_argument("--max-train-episodes", type=int, default=None)
    p.add_argument("--rollout-every", type=int, default=200,
                   help="in-loop eval every N epochs on the EMA policy (0 = off); "
                        "DP3 baseline uses 200")
    # best checkpoint (same idea as train.py's TopKCheckpointManager)
    p.add_argument("--save-best", choices=("auto", "1", "0"), default="auto",
                   help="save checkpoints/best.ckpt when the in-loop eval improves "
                        "(auto = on iff --rollout-every > 0)")
    p.add_argument("--best-monitor", default=None,
                   help="metric to optimise (default cfg.checkpoint.topk.monitor_key, "
                        "i.e. test_mean_score; others: SR_test_L3, SR_test_L5, "
                        "mean_success_rates, mean_traj_rewards)")
    p.add_argument("--best-k", type=int, default=None,
                   help="keep top-k eval ckpts (default cfg.checkpoint.topk.k = 1)")
    p.add_argument("--best-mode", choices=("max", "min"), default=None,
                   help="default cfg.checkpoint.topk.mode (max)")
    p.add_argument("--best-name", default="best.ckpt",
                   help="stable alias refreshed on every improvement")
    p.add_argument("--runtime-override", action="append", default=[], metavar="K=V",
                   help="extra hydra override, repeatable")
    return p


def parse_hidden(text):
    vals = [int(x) for x in str(text).replace("x", ",").replace("-", ",").split(",")
            if str(x).strip()]
    if not vals or any(v <= 0 for v in vals):
        raise SystemExit("[pijepa_common] --bn-hidden needs positive ints, got {!r}".format(
            text))
    return tuple(vals)


# --------------------------------------------------------------------------- data
class CleanPollDataset(torch.utils.data.Dataset):
    """clean (stored FPS cloud) + Phase1-polluted (fullpc + forged clutter) pairs.

    ``clean`` is exactly what DP3 trains on (``data/point_cloud``); ``poll`` is the
    pollution composition ``sample_512(concat(raw_point_cloud[f], blob))``.
    Pollution is only generated for the first ``n_obs_steps`` frames of a sample
    (that is all the encoder consumes) -> 2 FPS per sample instead of ``horizon``.

    Only ``action`` needs the full horizon (the DP3 mask supervises all of it), so
    the ``SequenceSampler`` is built with ``keys=['action']`` and the obs frames are
    read directly: per-sample point-cloud I/O drops from ``horizon`` to
    ``n_obs_steps`` frames.
    """

    def __init__(self, zarr_path, horizon=16, pad_before=1, pad_after=7, seed=42,
                 max_train_episodes=None,
                 n_obs_steps=2, num_points=1024, voxel_min=10, voxel_max=50,
                 poll_mode="bbox", poll_seed=0, fps_device="cpu", ws=None,
                 occ_cells=None):
        super().__init__()
        import zarr
        root = zarr.open(str(zarr_path), mode="r")
        if "data" not in root.group_keys():
            raise ValueError("{} is not a dataset (no 'data' group)".format(zarr_path))
        d = root["data"]
        for key in ("raw_point_cloud", "raw_point_cloud_len"):
            if key not in d:
                raise KeyError("{} has no data/{} -> need a fullpc zarr".format(
                    zarr_path, key))
        self.zarr_path = str(zarr_path)
        self.n_frames = int(d["point_cloud"].shape[0])
        self.num_points = int(num_points or d["point_cloud"].shape[1])
        self.n_obs_steps = int(n_obs_steps)
        self.voxel_min, self.voxel_max = int(voxel_min), int(voxel_max)
        self.poll_mode, self.poll_seed = str(poll_mode), int(poll_seed)
        self.fps_device = str(fps_device)
        self.ws, self.occ_cells = ws, occ_cells
        self.raws = [np.asarray(d["raw_point_cloud"][i], np.float32).reshape(-1, 6)
                     for i in range(self.n_frames)]
        self.replay_buffer = ReplayBuffer.copy_from_path(
            self.zarr_path, keys=["state", "action", "point_cloud"])
        self.point_cloud = self.replay_buffer["point_cloud"]
        self.state = self.replay_buffer["state"]
        # no validation split: every episode is used for training
        train_mask = downsample_mask(
            mask=np.ones(self.replay_buffer.n_episodes, dtype=bool),
            max_n=max_train_episodes, seed=seed)
        self.train_mask = train_mask
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, sequence_length=int(horizon),
            pad_before=int(pad_before), pad_after=int(pad_after),
            keys=["action"],
            episode_mask=train_mask)
        self.seed = int(seed)
        self.base_seed = int(seed) + int(poll_seed)
        self.epoch = 0
        self._rng_key = None
        self.rng = np.random.default_rng(self.base_seed)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self._rng_key = None            # force per-worker re-seed on next item

    def _rng(self):
        info = torch.utils.data.get_worker_info()
        wid = 0 if info is None else int(info.id)
        key = (self.epoch, wid)
        if self._rng_key != key:
            self._rng_key = key
            self.rng = np.random.default_rng(
                self.base_seed + 7919 * wid + 104729 * self.epoch)
        return self.rng

    def _frame_indices(self, idx):
        """Global frame ids of the first ``n_obs_steps`` steps (pads map to the
        first valid frame, mirroring SequenceSampler's duplicate padding)."""
        bs, _be, ss, _se = self.sampler.indices[idx]
        return [int(bs) if t < ss else int(bs + (t - ss)) for t in range(self.n_obs_steps)]

    def get_normalizer(self, mode="limits", **kwargs):
        from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"][..., :],
            "point_cloud": self.replay_buffer["point_cloud"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        idx = int(idx)
        act = np.asarray(self.sampler.sample_sequence(idx)["action"], np.float32)
        frames = self._frame_indices(idx)
        pc = np.stack([self.point_cloud[f] for f in frames], axis=0).astype(np.float32)
        st = np.stack([self.state[f] for f in frames], axis=0).astype(np.float32)
        rng = self._rng()
        polls = []
        for f in frames:
            blob = forge_voxel_rows(rng, p_min=self.voxel_min, p_max=self.voxel_max,
                                    ws=self.ws, mode=self.poll_mode,
                                    occ_cells=self.occ_cells)
            polls.append(sample_512(np.concatenate([self.raws[f], blob], axis=0),
                                    num_points=self.num_points, device=self.fps_device))
        poll = np.stack(polls, axis=0).astype(np.float32)        # (To, N, 6)
        return {
            "obs": {"point_cloud": torch.from_numpy(pc),
                    "agent_pos": torch.from_numpy(st)},
            "action": torch.from_numpy(act),
            "obs_poll": torch.from_numpy(poll),
        }


# ------------------------------------------------------------------------ losses
def make_targets(policy, action):
    """Sample one noised action / timestep / cond_mask shared by both views."""
    nactions = policy.normalizer["action"].normalize(action)
    cond_mask = policy.mask_generator(nactions.shape)
    noise = torch.randn_like(nactions)
    ts = torch.randint(0, policy.noise_scheduler.config.num_train_timesteps,
                       (nactions.shape[0],), device=nactions.device).long()
    noisy = policy.noise_scheduler.add_noise(nactions, noise, ts)
    noisy[cond_mask] = nactions[cond_mask]
    ptype = str(policy.noise_scheduler.config.prediction_type)
    if ptype == "sample":
        target = nactions          # == MSE(pred_action, gt_action)
    elif ptype == "epsilon":
        target = noise
    else:
        raise NotImplementedError("prediction_type: {}".format(ptype))
    return noisy, ts, target, cond_mask


def diffusion_loss(policy, noisy, ts, target, cond_mask, cond128):
    """DP3 diffusion loss for one view (``cond128``: (M,128) encoder output)."""
    global_cond = cond128.reshape(noisy.shape[0], -1)
    pred = policy.model(sample=noisy, timestep=ts, local_cond=None,
                        global_cond=global_cond)
    loss = F.mse_loss(pred, target, reduction="none")
    loss = loss * (~cond_mask).type(loss.dtype)
    return _reduce(loss, "b ... -> b (...)", "mean").mean()


def eval_fps_device():
    """Device used by the eval-time point-cloud pipeline (single source of truth)."""
    try:
        import inspect
        from diffusion_policy_3d.gym_util.mjpc_wrapper import point_cloud_sampling
        src = inspect.getsource(point_cloud_sampling)
        if ".cuda()" in src:
            return "cuda" if torch.cuda.is_available() else "cpu"
        dev = inspect.signature(point_cloud_sampling).parameters["device"].default
        return str(dev)
    except Exception:                                       # noqa: BLE001
        return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_fps_device(requested):
    """'auto' -> the eval-time FPS device, so train pollution matches deployment."""
    return eval_fps_device() if str(requested) == "auto" else str(requested)


def compose_cfg(a):
    try:
        OmegaConf.register_new_resolver("eval", eval, replace=True)
    except ValueError:
        pass
    overrides = [
        "task=" + str(a.task),
        "policy.pointcloud_encoder_cfg.pool_type=" + str(a.pool_type),
        "policy.pointcloud_encoder_cfg.pool_temperature=" + str(a.pool_temp),
        "policy.pointcloud_encoder_cfg.pool_score_hidden=" + str(a.pool_score_hidden),
    ]
    if a.zarr:
        overrides.append("task.dataset.zarr_path=" + str(a.zarr))
    if a.horizon:
        overrides.append("horizon=" + str(int(a.horizon)))
    if a.n_obs_steps:
        overrides.append("n_obs_steps=" + str(int(a.n_obs_steps)))
    overrides.extend(list(a.runtime_override or []))
    with initialize_config_dir(config_dir=CFG_DIR, version_base=None):
        return compose(config_name="dp3.yaml", overrides=overrides)


def load_ckpt(path):
    try:
        return torch.load(str(path), pickle_module=dill, map_location="cpu",
                          weights_only=False, mmap=True)
    except Exception:                                       # noqa: BLE001
        with open(str(path), "rb") as f:
            return torch.load(f, pickle_module=dill, map_location="cpu",
                              weights_only=False)


def save_ckpt(path, cfg, policy, ema_policy, optimizer, epoch, global_step, out_dir,
              lr_scheduler=None, with_optimizer=True, extra_pickles=None):
    state_dicts = {"model": policy.state_dict(),
                   "ema_model": ema_policy.state_dict()}
    if with_optimizer:
        state_dicts["optimizer"] = optimizer.state_dict()
    payload = {
        "cfg": cfg,
        "state_dicts": state_dicts,
        "pickles": {"epoch": dill.dumps(int(epoch)),
                    "global_step": dill.dumps(int(global_step)),
                    "_output_dir": dill.dumps(str(out_dir))},
    }
    if lr_scheduler is not None:
        payload["pickles"]["lr_scheduler"] = dill.dumps(lr_scheduler.state_dict())
    for k, v in (extra_pickles or {}).items():
        payload["pickles"][k] = dill.dumps(v)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path.open("wb"), pickle_module=dill)
    return str(path)


def maybe_save_best(topk, threshold, mode, score, cfg, policy, ema_policy, optimizer,
                    epoch, global_step, out_dir, alias, best_json, best_monitor,
                    best_name="best.ckpt"):
    """Overwrite ``checkpoints/<best_name>`` once ``score`` beats ``threshold``.

    ``topk`` only gates this off at ``k=0``; the file itself is a single stable
    ``best.ckpt`` (no ``epoch=*-score.ckpt`` copies, no symlink).  Written to
    ``<name>.tmp`` and renamed, so a crash cannot truncate the previous best.  No
    optimizer state (half the size; ``eval.py`` skips it anyway).

    Returns ``(score, path)`` when saved, else ``None``.
    """
    if score is None or not math.isfinite(float(score)):
        return None
    if topk is not None and int(getattr(topk, "k", 1)) <= 0:
        return None
    better = (score > threshold) if mode == "max" else (score < threshold)
    if not better:
        return None
    path = pathlib.Path(alias)
    tmp = path.with_name(path.name + ".tmp")
    save_ckpt(tmp, cfg, policy, ema_policy, optimizer, epoch, global_step, out_dir,
              lr_scheduler=None, with_optimizer=False,
              extra_pickles={"best_score": float(score),
                             "best_monitor": best_monitor})
    tmp.replace(path)
    best_json.write_text(json.dumps({"monitor": best_monitor, "mode": mode,
                                     "score": float(score), "epoch": int(epoch),
                                     "file": path.name}, indent=2))
    return float(score), str(path)


def _memlog():
    """dr3d/bench/memlog.py (mem_str / soft_close / gc_cuda); lazy, optional."""
    _bench = _REPO / "3D-Diffusion-Policy" / "diffusion_policy_3d" / "dr3d" / "bench"
    if str(_bench) not in sys.path:
        sys.path.insert(0, str(_bench))
    import memlog as _m
    return _m


def _vram_str():
    """Caching-allocator footprint; '' when CUDA is unavailable."""
    try:
        if not torch.cuda.is_available():
            return ""
        return " vram_reserved={:.2f}G".format(torch.cuda.memory_reserved() / 1e9)
    except Exception:                                  # noqa: BLE001
        return ""


def run_rollout(cfg, out_dir, policy, seed, num_points):
    """In-loop eval of the EMA policy on the shared deploy branch.

    The eval env owns a MuJoCo model + MjrContext (GPU buffers on the process-wide
    EGL context), so it is closed and dropped in ``finally``.  Without that each
    rollout left a live env behind, and the run died with SIGABRT on the second
    rollout (epoch 400 of a 200-epoch rollout cadence).
    """
    cfg = copy.deepcopy(cfg)
    OmegaConf.set_struct(cfg, False)
    # Only runners whose __init__ takes num_points (metaworld) accept the training
    # N; adroit/dexart fix N inside their gym wrapper (ENV_POINT_CLOUD_CONFIG).
    _rcls = hydra.utils.get_class(str(cfg.task.env_runner._target_))
    if "num_points" in inspect.signature(_rcls.__init__).parameters:
        cfg.task.env_runner.num_points = int(num_points)  # match training/view N
    else:
        print("[rollout] {} has no num_points kwarg -> wrapper-fixed N stays in "
              "effect".format(_rcls.__name__))
    ml = _memlog()
    print("[rollout] env build: {}{}".format(ml.mem_str(), _vram_str()))
    runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=str(out_dir))
    runner.save_episode_video = False
    if hasattr(runner, "seed"):
        runner.seed = int(seed)
    policy.eval()
    try:
        log = runner.run(policy)
    finally:
        policy.train()                       # never resume training in eval mode
        ml.soft_close(getattr(runner, "env", None))
        del runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[rollout] env released: {}{}".format(ml.mem_str(), _vram_str()))
    return {k: float(v) for k, v in log.items() if isinstance(v, (int, float))}


def _to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    return {k: _to_device(v, device) for k, v in x.items()}


def train_step(policy, ema, optimizer, normalizer, batch, to, device=None,
               lr_sched=None, l2_clean_weight=1.0, l2_poll_weight=1.0,
               lambda_jepa_clean=1.0, lambda_jepa_poll=1.0):
    """One optimizer step: both views through the SAME weights, one backward.

    ``loss = l2_clean_weight * L2_clean + l2_poll_weight * L2_poll`` plus, for
    every enabled view (``lambda_jepa_* > 0``), ``lambda * SmoothL1(z, sg(t))``
    where ``t`` is the pre-bottleneck pooled clean feature.  ``batch`` is one
    ``CleanPollDataset`` item batch: clouds hold the first ``to`` frames,
    ``action`` the full horizon.
    """
    batch = _to_device(batch, device or next(policy.parameters()).device)
    clean = batch["obs"]["point_cloud"]         # (B, To, N, 6)
    state = batch["obs"]["agent_pos"]           # (B, To, S)
    poll = batch["obs_poll"]                    # (B, To, N, 6)
    action = batch["action"]                    # (B, T, Da)
    flat = clean.shape[0] * to
    pc_clean = clean.reshape(flat, *clean.shape[2:])
    st = state.reshape(flat, -1)
    pc_poll = poll.reshape(flat, *poll.shape[2:])

    nobs = normalizer.normalize({"point_cloud": pc_clean, "agent_pos": st})
    nobs_poll = normalizer.normalize({"point_cloud": pc_poll, "agent_pos": st})

    state_feat = policy.encode_state(nobs)                   # once, shared by both
    z_c, pc_c, _ = policy.encode_obs(nobs, state_feat)
    z_p, pc_p, _ = policy.encode_obs(nobs_poll, state_feat)
    cond_c = torch.cat([pc_c, state_feat], dim=-1)           # (M,128)
    cond_p = torch.cat([pc_p, state_feat], dim=-1)

    noisy, ts, target, cond_mask = make_targets(policy, action)
    loss_clean = diffusion_loss(policy, noisy, ts, target, cond_mask, cond_c)
    loss_poll = diffusion_loss(policy, noisy, ts, target, cond_mask, cond_p)
    loss = l2_clean_weight * loss_clean + l2_poll_weight * loss_poll

    l1_clean = l1_poll = None
    if lambda_jepa_clean > 0.0 or lambda_jepa_poll > 0.0:
        tg = policy.encode_target(nobs, state_feat)          # detached (no grad)
        if lambda_jepa_clean > 0.0:
            l1_clean = F.smooth_l1_loss(z_c, tg)
            loss = loss + lambda_jepa_clean * l1_clean
        if lambda_jepa_poll > 0.0:
            l1_poll = F.smooth_l1_loss(z_p, tg)
            loss = loss + lambda_jepa_poll * l1_poll

    optimizer.zero_grad(set_to_none=True)
    loss.backward()      # both views share the weights: L2_c + L2_p train the trunk
    optimizer.step()
    if lr_sched is not None:
        lr_sched.step()  # per batch, matching 3D-Diffusion-Policy/train.py
    ema.step(policy)
    return (float(loss), float(loss_clean), float(loss_poll),
            None if l1_clean is None else float(l1_clean),
            None if l1_poll is None else float(l1_poll))
