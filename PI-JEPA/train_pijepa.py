#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PI-JEPA v5 (main method): per-point bottleneck + temporal dynamics predictor (TDT).

Data: clean/poll pairs over the observation window plus the F real clean frames
that follow it (frames past episode end are clamped and masked out of the loss).
Train step: L2 on clean/poll + weighted SmoothL1 on the (To+F) TDT sequence against
stop-grad pooled pre-bottleneck features; future frame k >= To is weighted
``gamma**(k-To+1)``, and the aux weights ramp 0 -> 1 over ``--l1-warmup-epochs``.
Deploy: the bottleneck branch only, the TDT is train-only. See the class
docstrings in this file for the exact loss formulas.
"""
import copy
import json
import os
import pathlib
import sys

import tqdm

import numpy as np
import torch
import torch.nn.functional as F

_REPO = pathlib.Path(__file__).resolve().parents[1]          # /root/3DP
_PKG = pathlib.Path(__file__).resolve().parent               # /root/3DP/PI-JEPA
for _p in (str(_REPO / "3D-Diffusion-Policy"), str(_REPO), str(_PKG)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dill  # noqa: E402
import hydra  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import pijepa_common as common  # noqa: E402
from pijepa_policy import DEFAULT_TDT_CFG  # noqa: E402
from diffusion_policy_3d.common.checkpoint_util import TopKCheckpointManager  # noqa: E402
from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler  # noqa: E402
from diffusion_policy_3d.voxel.workspace import resolve_workspace  # noqa: E402


def build_parser():
    """Shared parser + the TDT options (same flags, same defaults elsewhere)."""
    p = common.build_parser()
    p.description = __doc__.splitlines()[0]
    g = p.add_argument_group("v5 / TDT")
    g.add_argument("--future-steps", type=int, default=0,
                   help="F future frames supervised by the TDT (0 = cfg.n_action_steps)")
    g.add_argument("--gamma", type=float, default=0.5,
                   help="per-step decay of the future SmoothL1 weight: "
                        "gamma**(k-To+1)")
    g.add_argument("--tdt-layers", type=int, default=DEFAULT_TDT_CFG["layers"])
    g.add_argument("--tdt-heads", type=int, default=DEFAULT_TDT_CFG["heads"])
    g.add_argument("--tdt-ff", type=int, default=DEFAULT_TDT_CFG["dim_ff"])
    g.add_argument("--tdt-dropout", type=float, default=DEFAULT_TDT_CFG["dropout"])
    g.add_argument("--tdt-pool", choices=("task", "mean"), default="task",
                   help="TDT input pool: 'task' = the softmax/attention-pooled feature "
                        "the UNet also uses (identical at step 0, zero-init scores); "
                        "'mean' = the old plain-mean input")
    g.add_argument("--l1-mode", choices=("sum", "mean"), default="sum",
                   help="'sum' = doc formula (sum_k w_k m_k * loss_k); 'mean' = same "
                        "normalised by the active weight mass")
    g.add_argument("--l1-warmup-epochs", type=int, default=200,
                   help="L1 warmstart: JEPA aux weights 0 -> 1 over the first N epochs "
                        "(1 = off, full weight from epoch 1; default 200)")
    # pijepa_common defaults --l2-poll-weight to 1.0; v5 starts from a weaker L2p.
    p.set_defaults(l2_poll_weight=0.3)
    return p


# --------------------------------------------------------------------------- data
class CleanFuturePollDataset(common.CleanPollDataset):
    """Clean/poll pairs + the F real clean frames after the observation window.

    The future frames are read straight from ``data/point_cloud`` (no FPS, no
    pollution), so the sampling cost stays at the base level.  Frames past the end of the
    episode are clamped to its last frame and flagged invalid.
    """

    def __init__(self, *args, future_steps=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.future_steps = int(future_steps)
        if self.future_steps < 1:
            raise ValueError("future_steps must be >= 1, got {}".format(future_steps))
        seq_len = int(self.sampler.sequence_length)
        if self.n_obs_steps + self.future_steps > seq_len:
            raise ValueError(
                "To+F = {}+{} exceeds the sampler sequence_length {} (horizon); use a "
                "smaller --future-steps".format(self.n_obs_steps, self.future_steps,
                                                seq_len))

    def _future_indices(self, idx):
        """(bs,be,ss,se) -> (frame ids [F], valid mask [F]).

        ``[bs, be)`` is the episode-valid part of the window, so ``bs+(t-ss) >= be``
        means the episode already ended: clamp to ``be-1`` and mark invalid.
        """
        bs, be, ss, _se = (int(x) for x in self.sampler.indices[int(idx)])
        last = be - 1
        frames, valid = [], []
        for t in range(self.n_obs_steps, self.n_obs_steps + self.future_steps):
            g = bs + (t - ss)
            frames.append(min(g, last))
            valid.append(g < be)
        return frames, np.asarray(valid, dtype=np.bool_)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        frames, valid = self._future_indices(idx)
        fut = np.stack([self.point_cloud[f] for f in frames], axis=0).astype(np.float32)
        item["obs_future"] = torch.from_numpy(fut)              # (F,N,6)
        item["fut_valid"] = torch.from_numpy(valid)             # (F,) bool
        return item


# ------------------------------------------------------------------------ losses
def seq_l1_weights(to, future_steps, gamma, device=None):
    """(To+F,) frame weights: 1.0 for k < To, ``gamma**(k-To+1)`` afterwards."""
    to, fut = int(to), int(future_steps)
    k = torch.arange(to + fut, dtype=torch.float32)
    w = torch.where(k < to, torch.ones_like(k),
                    torch.pow(float(gamma), k - (to - 1)))
    return w.to(device) if device is not None else w


def seq_smooth_l1(pred, target, weight, valid=None, mode="sum"):
    """Weighted per-frame SmoothL1 between ``(B,L,D)`` sequences.

    ``weight`` is ``(L,)`` (the seq_l1_weights vector).  ``valid`` is the ``(B,F)``
    episode-end mask of the *trailing* future frames: it is left-padded with ones for
    the observed frames (``L-F`` of them), so ``valid`` may also be omitted when the
    slice covers the future only.  ``mode='sum'`` follows the doc (sum over frames of
    the per-frame mean); ``'mean'`` divides by the active weight mass, which keeps the
    scale independent of L and of the episode end.
    """
    per = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1)     # (B,L)
    w = weight.unsqueeze(0).to(per.dtype)
    if valid is not None:
        n_obs = int(per.shape[1]) - int(valid.shape[1])
        if n_obs < 0:
            raise ValueError("valid has more frames ({}) than the sequence ({})".format(
                valid.shape[1], per.shape[1]))
        mask = valid.to(per.dtype)
        if n_obs:
            mask = torch.cat([per.new_ones(per.shape[0], n_obs), mask], dim=1)
        w = w * mask
    num = (per * w).sum(dim=1)                                             # (B,)
    if mode == "mean":
        return (num / w.sum(dim=1).clamp_min(1e-6)).mean()
    return num.mean()


def l1_warmup_scale(epoch, warmup_epochs):
    """0-based ``epoch`` -> L1 multiplier: 0 at epoch 1, 1 at epoch ``warmup_epochs``."""
    n = int(warmup_epochs)
    return 1.0 if n <= 1 else min(1.0, float(epoch) / float(n - 1))


def train_step_v5(policy, ema, optimizer, normalizer, batch, to, future_steps,
                  device=None, lr_sched=None, l2_clean_weight=1.0,
                  l2_poll_weight=1.0, lambda_jepa_clean=1.0, lambda_jepa_poll=1.0,
                  gamma=0.5, l1_mode="sum"):
    """One optimizer step: clean + poll views and the (To+F) sequence losses.

    ``batch`` is one ``CleanFuturePollDataset`` item batch: clean/poll/future clouds
    hold the first ``To`` / the next ``F`` frames, ``action`` the full horizon.
    Returns ``(loss, L2_clean, L2_poll, L1_clean, L1_poll, extras)``.
    """
    to, fut = int(to), int(future_steps)
    batch = common._to_device(batch, device or next(policy.parameters()).device)
    clean = batch["obs"]["point_cloud"]         # (B, To, N, 6)
    state = batch["obs"]["agent_pos"]           # (B, To, S)
    poll = batch["obs_poll"]                    # (B, To, N, 6)
    fut_pc = batch["obs_future"]                # (B, F, N, 6)
    valid = batch["fut_valid"]                  # (B, F) bool
    action = batch["action"]                    # (B, T, Da)

    b = int(clean.shape[0])
    m = b * to
    st = state.reshape(m, -1)
    nobs = normalizer.normalize({"point_cloud": clean.reshape(m, *clean.shape[2:]),
                                 "agent_pos": st})
    nobs_poll = normalizer.normalize({"point_cloud": poll.reshape(m, *poll.shape[2:]),
                                      "agent_pos": st})
    nobs_fut = normalizer.normalize(
        {"point_cloud": fut_pc.reshape(b * fut, *fut_pc.shape[2:])})

    out = policy.encode_views(nobs, nobs_poll, nobs_fut, to, fut)

    noisy, ts, target, cond_mask = common.make_targets(policy, action)
    loss_clean = common.diffusion_loss(policy, noisy, ts, target, cond_mask, out["cond_clean"])
    loss_poll = common.diffusion_loss(policy, noisy, ts, target, cond_mask, out["cond_poll"])
    loss = l2_clean_weight * loss_clean + l2_poll_weight * loss_poll

    l1_clean = l1_poll = l1c_fut = l1p_fut = None
    if lambda_jepa_clean > 0.0 or lambda_jepa_poll > 0.0:
        w = seq_l1_weights(to, fut, gamma, clean.device)
        seq_t = out["seq_target"]
        if lambda_jepa_clean > 0.0:
            pred = out["seq_clean"]
            l1_clean = seq_smooth_l1(pred, seq_t, w, valid, l1_mode)
            l1c_fut = seq_smooth_l1(pred[:, to:], seq_t[:, to:], w[to:], valid, l1_mode)
            loss = loss + lambda_jepa_clean * l1_clean
        if lambda_jepa_poll > 0.0:
            pred = out["seq_poll"]
            l1_poll = seq_smooth_l1(pred, seq_t, w, valid, l1_mode)
            l1p_fut = seq_smooth_l1(pred[:, to:], seq_t[:, to:], w[to:], valid, l1_mode)
            loss = loss + lambda_jepa_poll * l1_poll

    optimizer.zero_grad(set_to_none=True)
    loss.backward()      # both views + TDT share the trunk: one backward
    optimizer.step()
    if lr_sched is not None:
        lr_sched.step()  # per batch, matching 3D-Diffusion-Policy/train.py
    ema.step(policy)

    def _f(x):
        return None if x is None else float(x)

    extras = {"L1c_fut": _f(l1c_fut), "L1p_fut": _f(l1p_fut),
              "valid_ratio": float(valid.to(torch.float32).mean())}
    return (float(loss), float(loss_clean), float(loss_poll),
            _f(l1_clean), _f(l1_poll), extras)


# -------------------------------------------------------------------------- main
def tdt_cfg_from_args(a, future_steps):
    """The CLI/env knobs that shape the TDT (must match between train and eval)."""
    return {"future_steps": int(future_steps), "layers": int(a.tdt_layers),
            "heads": int(a.tdt_heads), "dim_ff": int(a.tdt_ff),
            "dropout": float(a.tdt_dropout)}


def build_policy(cfg, bottleneck_cfg, tdt_cfg, tdt_pool="task"):
    """Instantiate ``PIJEPAPolicyV5`` from the hydra policy node (same kwargs as DP3)."""
    pcfg = OmegaConf.create(OmegaConf.to_container(cfg.policy, resolve=True))
    OmegaConf.set_struct(pcfg, False)
    pcfg._target_ = "pijepa_policy.PIJEPAPolicyV5"
    pcfg.bottleneck_cfg = OmegaConf.create({"hidden": list(bottleneck_cfg)})
    pcfg.tdt_cfg = OmegaConf.create({k: v for k, v in dict(tdt_cfg).items()})
    pcfg.tdt_pool = str(tdt_pool)
    return hydra.utils.instantiate(pcfg)


def main(argv=None):
    a = build_parser().parse_args(argv)
    lam_c, lam_p = float(a.lambda_jepa_clean), float(a.lambda_jepa_poll)
    for _name, _val in (("--lambda-jepa-clean", lam_c), ("--lambda-jepa-poll", lam_p)):
        if _val not in (0.0, 1.0):
            raise SystemExit("[pijepa] {} must be 1 (on) or 0 (off), got {}".format(
                _name, _val))
    stage = {(0.0, 0.0): "action loss only",
             (1.0, 0.0): "L1_clean seq only",
             (0.0, 1.0): "L1_poll seq only",
             (1.0, 1.0): "L1_clean + L1_poll seq"}[(lam_c, lam_p)]
    if float(a.l2_clean_weight) < 0 or float(a.l2_poll_weight) < 0:
        raise SystemExit("[pijepa] --l2-*-weight must be >= 0")
    if not (0.0 < float(a.gamma) <= 1.0):
        raise SystemExit("[pijepa] --gamma must be in (0,1], got {}".format(a.gamma))
    bn_hidden = common.parse_hidden(a.bn_hidden)
    zarr = a.zarr or "/hy-tmp/3DP/data/{}_expert.zarr".format(a.task)
    if not os.path.isdir(zarr):
        raise SystemExit("[pijepa] zarr not found: {}".format(zarr))
    cfg = common.compose_cfg(a)
    device = torch.device(a.device if (torch.cuda.is_available() or a.device == "cpu")
                          else "cpu")
    torch.manual_seed(int(a.seed))
    np.random.seed(int(a.seed))

    horizon = int(cfg.horizon)
    to = int(cfg.n_obs_steps)
    fut = int(a.future_steps) if int(a.future_steps) > 0 else int(cfg.n_action_steps)
    if to + fut > horizon:
        raise SystemExit("[pijepa] To+F = {}+{} exceeds horizon {}; lower "
                         "--future-steps".format(to, fut, horizon))
    tdt_cfg = tdt_cfg_from_args(a, fut)
    import zarr as _zarr
    _d = _zarr.open(zarr, mode="r")["data"]
    n_points = int(_d["point_cloud"].shape[1])
    if a.num_points and int(a.num_points) != n_points:
        raise SystemExit("[pijepa] --num-points {} != zarr N {} (pollution must "
                         "match the clean cloud)".format(a.num_points, n_points))

    fps_device = common.resolve_fps_device(a.fps_device)
    num_workers = int(a.num_workers)
    if fps_device != "cpu" and num_workers > 0:
        print("[pijepa] FPS device '{}' == eval-time device; CUDA inside forked "
              "DataLoader workers is unsafe -> forcing num_workers=0 (use "
              "FPS_DEVICE=cpu for parallel workers, at the cost of a device that "
              "may differ from eval)".format(fps_device))
        num_workers = 0
    print("[pijepa] fps_device={} (requested={}, auto=eval-aligned)".format(
        fps_device, a.fps_device))

    ws = resolve_workspace(task=a.task, zarr=zarr, mode=str(a.workspace))
    print("[pijepa] {}".format(ws.describe()))
    occ_cells = None
    if str(a.poll_mode) == "occupied":
        from diffusion_policy_3d.jepa.forge import occupied_cells_from_zarr
        occ_cells, _dims = occupied_cells_from_zarr(zarr, ws)
        print("[pijepa] forge mode=occupied: {} occupied cells".format(len(occ_cells)))

    ds_kwargs = dict(zarr_path=zarr, horizon=horizon, pad_before=to - 1,
                     pad_after=int(cfg.n_action_steps) - 1, seed=int(a.seed),
                     max_train_episodes=(a.max_train_episodes
                                         if a.max_train_episodes is not None
                                         else cfg.task.dataset.get("max_train_episodes",
                                                                   None)),
                     n_obs_steps=to, num_points=n_points,
                     voxel_min=int(a.poll_voxel_min), voxel_max=int(a.poll_voxel_max),
                     poll_mode=str(a.poll_mode), poll_seed=int(a.poll_seed),
                     fps_device=fps_device, ws=ws, occ_cells=occ_cells,
                     future_steps=fut)
    train_ds = CleanFuturePollDataset(**ds_kwargs)
    print("[pijepa] windows train={} (episodes={}, no val split) N={} To={} F={} "
          "horizon={}".format(len(train_ds), train_ds.replay_buffer.n_episodes, n_points, to,
                              fut, horizon))

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=int(a.batch), shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=False)

    policy = build_policy(cfg, bn_hidden, tdt_cfg, a.tdt_pool)
    normalizer = train_ds.get_normalizer()
    policy.set_normalizer(normalizer)
    policy.to(device)                 # normalizer stats live inside the policy module
    normalizer.to(device)             # the training loop uses this object directly
    ema_policy = copy.deepcopy(policy)        # already on `device`
    ema_policy.set_normalizer(normalizer)
    ema = hydra.utils.instantiate(cfg.ema, model=ema_policy)
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=policy.parameters())
    if a.lr:
        for group in optimizer.param_groups:
            group["lr"] = float(a.lr)
    lr_sched = get_scheduler(cfg.training.lr_scheduler, optimizer=optimizer,
                             num_warmup_steps=int(cfg.training.lr_warmup_steps),
                             num_training_steps=max(1, len(train_loader)) * int(a.epochs))

    # default follows the launcher: <task>-pijepa-<addition_info>_seed<seed>
    out_dir = a.out or str(pathlib.Path("/hy-tmp/data/outputs") / "{}_seed{}".format(
        a.exp_name or "{}-pijepa".format(a.task), a.seed))
    ckpt_path = pathlib.Path(out_dir) / "checkpoints" / "latest.ckpt"
    # --- best checkpoint (in-loop eval), mirroring train.py's TopKCheckpointManager
    save_best = {"auto": bool(int(a.rollout_every)), "1": True, "0": False}[str(a.save_best)]
    best_monitor = str(a.best_monitor or cfg.checkpoint.topk.monitor_key)
    best_mode = str(a.best_mode or cfg.checkpoint.topk.mode)
    best_k = int(a.best_k if a.best_k is not None else cfg.checkpoint.topk.k)
    best_alias = ckpt_path.parent / str(a.best_name)
    best_json = ckpt_path.parent / "best.json"
    topk = None
    best_score = float("-inf") if best_mode == "max" else float("inf")
    if save_best:
        topk = TopKCheckpointManager(
            save_dir=str(ckpt_path.parent), monitor_key=best_monitor, mode=best_mode,
            k=best_k)
        if best_json.is_file():                     # resume must not overwrite a better one
            try:
                rec = json.loads(best_json.read_text())
                if rec.get("monitor") == best_monitor:
                    best_score = float(rec["score"])
            except Exception:                       # noqa: BLE001
                pass
    if int(a.rollout_every) == 0 and str(a.save_best) == "1":
        print("[pijepa] WARNING: --save-best 1 needs an in-loop eval, but "
              "--rollout-every 0 -> no best ckpt will be written")
    start_epoch, global_step = 0, 0
    if a.resume and ckpt_path.is_file():
        payload = common.load_ckpt(ckpt_path)
        policy.load_state_dict(payload["state_dicts"]["model"])
        ema_policy.load_state_dict(payload["state_dicts"]["ema_model"])
        optimizer.load_state_dict(payload["state_dicts"]["optimizer"])
        # the normalizer params are rebuilt from CPU/mmap tensors -> re-place them
        policy.to(device)
        ema_policy.to(device)
        normalizer.load_state_dict(policy.normalizer.state_dict())
        normalizer.to(device)
        for st in optimizer.state.values():
            for k, val in list(st.items()):
                if torch.is_tensor(val):
                    st[k] = val.to(device)
        start_epoch = int(dill.loads(payload["pickles"]["epoch"]))
        global_step = int(dill.loads(payload["pickles"]["global_step"]))
        print("[pijepa] resumed {} (epoch={}, step={})".format(
            ckpt_path, start_epoch, global_step))
        sched_blob = payload["pickles"].get("lr_scheduler")
        if sched_blob is not None:
            lr_sched.load_state_dict(dill.loads(sched_blob))
            sched_src = "restored"
        else:
            for _ in range(start_epoch * len(train_loader)):
                lr_sched.step()
            sched_src = "fast-forwarded"
        print("[pijepa] lr_scheduler {} to step {} (lr={:.3e})".format(
            sched_src, lr_sched.last_epoch, lr_sched.get_last_lr()[0]))
    elif a.resume:
        print("[pijepa] --resume given but {} missing -> fresh run".format(ckpt_path))

    OmegaConf.set_struct(cfg, False)   # allow the provenance block below
    cfg.pijepa_v5 = OmegaConf.create({
        "loss_combination": "l2_clean_weight*L2_clean + l2_poll_weight*L2_poll "
                            "+ lambda_jepa_clean*L1_clean_seq "
                            "+ lambda_jepa_poll*L1_poll_seq",
        "lambda_jepa_clean": lam_c,
        "lambda_jepa_poll": lam_p,
        "stage": stage,
        "l2_clean_weight": float(a.l2_clean_weight),
        "l2_poll_weight": float(a.l2_poll_weight),
        "l1_mode": str(a.l1_mode),
        "l1_warmup_epochs": int(a.l1_warmup_epochs),
        "gamma": float(a.gamma),
        "future_steps": int(fut),
        "seq_weights": "1.0 for k<To, gamma**(k-To+1) after; future frames masked at "
                       "episode end",
        "bottleneck_cfg": {"hidden": list(bn_hidden)},
        "bottleneck_init": "last layer zero-init -> identity at step 0",
        "tdt_cfg": dict(tdt_cfg),
        "tdt_pool": str(a.tdt_pool),
        "pooling": "tdt_pool=task: softmax_n -> TDT input + UNet + target; "
                   "tdt_pool=mean: mean_n -> TDT input (legacy)",
        "encoder": "DP3 extractor.mlp + PIJEPA-BottleNeck + frame pools + final_proj",
        "state_encoder": "DP3 state_mlp (Linear(S,64)+ReLU+Linear(64,64))",
        "deploy": "mlp -> bottleneck -> softmax pool -> final_proj|state -> UNet "
                  "(TDT is train-only)",
        "poll_voxel_min": int(a.poll_voxel_min),
        "poll_voxel_max": int(a.poll_voxel_max), "poll_mode": str(a.poll_mode),
        "poll_seed": int(a.poll_seed), "fps_device": str(fps_device),
        "fps_device_requested": str(a.fps_device),
        "workspace": ws.to_dict(), "zarr": zarr, "pool_type": str(a.pool_type),
        "num_points": int(n_points), "n_obs_steps": int(to),
    })
    metrics_path = pathlib.Path(out_dir) / "pijepa_v5_metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    print("[pijepa] task={} seed={} epochs={} batch={} steps/epoch={}".format(
        a.task, a.seed, a.epochs, a.batch, len(train_loader)))
    print("[pijepa] loss = {}*L2_clean + {}*L2_poll + {}*L1_clean_seq + "
          "{}*L1_poll_seq | {} | pool={} T={} h={} l1_mode={}".format(
              a.l2_clean_weight, a.l2_poll_weight, lam_c, lam_p, stage,
              a.pool_type, a.pool_temp, a.pool_score_hidden, a.l1_mode))
    print("[pijepa] seq = To({})+F({}) frames, gamma={} -> weights {}".format(
        to, fut, a.gamma,
        [round(float(x), 4) for x in seq_l1_weights(to, fut, a.gamma)]))
    print("[pijepa] bottleneck 320->{}->256 (zero-init) | TDT {} | input pool={}"
          .format("->".join(str(h) for h in bn_hidden), tdt_cfg, a.tdt_pool))
    print("[pijepa] deploy: mlp->bn->softmax pool->proj|state->UNet (no TDT; "
          "eval must pass PIJEPA_V5=1 with the same TDT_* values)")
    print("[pijepa] zarr={}".format(zarr))
    print("[pijepa] out(write)={} device={} amp=no".format(out_dir, device))
    if save_best:
        print("[pijepa] best ckpt: monitor={} mode={} -> checkpoints/{} "
              "(overwritten when the score improves, no optimizer)".format(
                  best_monitor, best_mode, a.best_name))
    else:
        print("[pijepa] best ckpt: disabled")

    w_c = float(a.l2_clean_weight)
    w_p = float(a.l2_poll_weight)
    for epoch in range(start_epoch, int(a.epochs)):
        lam_w = l1_warmup_scale(epoch, a.l1_warmup_epochs)
        lam_c_e, lam_p_e = lam_c * lam_w, lam_p * lam_w
        train_ds.set_epoch(epoch)
        policy.train()
        ep_c = ep_p = ep_l1c = ep_l1p = 0.0
        ep_futc = ep_futp = ep_valid = 0.0
        n_batches = 0
        tepoch = tqdm.tqdm(total=len(train_loader),
                           desc="Training epoch {}/{}".format(epoch + 1, a.epochs),
                           leave=False,
                           mininterval=float(cfg.training.tqdm_interval_sec))
        for batch in train_loader:
            loss, l2c, l2p, l1c, l1p, extra = train_step_v5(
                policy, ema, optimizer, normalizer, batch, to, fut, device, lr_sched,
                w_c, w_p, lam_c_e, lam_p_e, float(a.gamma), str(a.l1_mode))
            global_step += 1
            n_batches += 1
            ep_c += l2c
            ep_p += l2p
            ep_valid += extra["valid_ratio"]
            post = {"loss": "{:.4f}".format(loss),
                    "L2c": "{:.4f}".format(l2c), "L2p": "{:.4f}".format(l2p)}
            if l1c is not None:
                ep_l1c += l1c
                ep_futc += extra["L1c_fut"] or 0.0
                post["L1c"] = "{:.4f}".format(l1c)
            if l1p is not None:
                ep_l1p += l1p
                ep_futp += extra["L1p_fut"] or 0.0
                post["L1p"] = "{:.4f}".format(l1p)
            tepoch.set_postfix(**post, refresh=False)
            tepoch.update(1)
            if a.log_every and global_step % int(a.log_every) == 0:
                rec = {"step": global_step, "epoch": epoch, "loss": loss,
                       "L2c": l2c, "L2p": l2p, "lr": lr_sched.get_last_lr()[0],
                       "valid_ratio": extra["valid_ratio"]}
                if l1c is not None:
                    rec["L1c"] = l1c
                    rec["L1c_fut"] = extra["L1c_fut"]
                if l1p is not None:
                    rec["L1p"] = l1p
                    rec["L1p_fut"] = extra["L1p_fut"]
                with metrics_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
            if a.max_steps and global_step >= int(a.max_steps):
                break
        tepoch.close()
        nb = max(1, n_batches)
        msg = "[pijepa] epoch {}/{} done: L2c={:.4f} L2p={:.4f}".format(
            epoch + 1, a.epochs, ep_c / nb, ep_p / nb)
        if lam_c_e > 0.0:
            msg += " L1c={:.4f} (fut {:.4f})".format(ep_l1c / nb, ep_futc / nb)
        if lam_p_e > 0.0:
            msg += " L1p={:.4f} (fut {:.4f})".format(ep_l1p / nb, ep_futp / nb)
        if lam_w < 1.0: msg += " l1w={:.3f}".format(lam_w)
        msg += " valid={:.3f}".format(ep_valid / nb)
        print(msg)

        if not a.no_ckpt and (a.save_every_epoch or (epoch + 1) % 50 == 0
                              or epoch + 1 == int(a.epochs)):
            p = common.save_ckpt(ckpt_path, cfg, policy, ema_policy, optimizer, epoch + 1,
                             global_step, out_dir, lr_scheduler=lr_sched)
            print("[pijepa] saved {}".format(p))
        if a.rollout_every and (epoch + 1) % int(a.rollout_every) == 0:
            try:
                log = common.run_rollout(cfg, out_dir, ema_policy, a.seed, n_points)
                print("[pijepa] rollout epoch {}/{}: {}".format(
                    epoch + 1, a.epochs, log))
            except Exception as exc:                       # noqa: BLE001
                log = {"error": "{}: {}".format(type(exc).__name__, exc)}
                print("[pijepa] rollout epoch {}/{} FAILED: {}".format(
                    epoch + 1, a.epochs, log["error"]))
            rec = {"epoch": epoch, "rollout": log}
            score = log.get(best_monitor) if save_best else None
            saved = None
            if save_best and score is not None:
                saved = common.maybe_save_best(topk, best_score, best_mode, score, cfg,
                                           policy, ema_policy, optimizer, epoch + 1,
                                           global_step, out_dir, best_alias, best_json,
                                           best_monitor, a.best_name)
            if saved is not None:
                best_score, best_file = saved
                print("[pijepa] best ckpt saved {} ({:.4f} @ epoch {})".format(
                    best_file, best_score, epoch + 1))
                rec["best"] = {"score": best_score,
                               "file": pathlib.Path(best_file).name}
            with metrics_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        if a.max_steps and global_step >= int(a.max_steps):
            print("[pijepa] --max-steps {} reached, stopping".format(a.max_steps))
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())





