#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PI-JEPA policy: DP3 encoder + per-point bottleneck (+ train-only TDT).

Two views of the same frame (only the first ``n_obs_steps`` frames, which is all
the encoder consumes)::

    clean : data/point_cloud[f]                              (FPS(raw, N))
    poll  : sample_512(concat(raw[f], forge_voxel_rows(...)))  (voxel pollution clutter)

Both views go through the SAME weights (no EMA target encoder, no extra head)::

    f  = PointNetMLP(pc)          # (M,N,256), DP3 extractor.mlp unchanged
    s  = StateMLP(agent_pos)      # (M,64), DP3 state_mlp unchanged, computed once
    f' = BottleNeck(f, s)         # (M,N,256), residual delta, zero-init => identity
    z  = SoftmaxPool(f')          # (M,256), DP3 _pool (pool_type='softmax')
    a  = UNet(final_projection(z) | s)

Loss (see ``pijepa_common.py``; each SmoothL1 weight is 1 or 0)::

    loss = l2_clean_weight * L2_clean + l2_poll_weight * L2_poll
           + lambda_jepa_clean * SmoothL1(z_c, sg(t))
           + lambda_jepa_poll  * SmoothL1(z_p, sg(t))

Because the bottleneck's last layer is zero-initialised the encoder output is
bit-identical to the DP3 baseline at step 0 (clean A/B), and the state encoder is
kept exactly as in DP3 (Linear(S,64)+ReLU+Linear(64,64), i.e. DP3's ``state_mlp``).

Deploy: one shared branch applied to whatever cloud the env provides::

    obs -> mlp -> bottleneck -> softmax pool -> final_projection | state -> UNet

``PIJEPAEncoder.forward`` is that function, so training and evaluation cannot
drift apart.  The EMA copy carries the bottleneck with it (no predictor sync).

Checkpoint compatibility: the module tree only adds ``obs_encoder.bottleneck.*``;
UNet / ``obs_feature_dim`` (128) / ``global_cond_dim`` are unchanged, so a DP3
state dict loads with ``strict=False`` (missing = bottleneck only).
"""
import pathlib
import sys

import torch
import torch.nn as nn

_REPO = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(_REPO / "3D-Diffusion-Policy"), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from diffusion_policy_3d.model.vision.pointnet_extractor import DP3Encoder  # noqa: E402
from diffusion_policy_3d.policy.dp3 import DP3  # noqa: E402

__all__ = ["PIJEPABottleNeck", "PIJEPAEncoder", "PIJEPAPolicy",
           "DEFAULT_BOTTLENECK_CFG", "PIJEPATemporalDynamics", "PIJEPAEncoderV5",
           "PIJEPAPolicyV5", "DEFAULT_TDT_CFG"]

# PIJEPA-BottleNeck widths (320 -> 128 -> 64 -> 128 -> 256)
DEFAULT_BOTTLENECK_CFG = {"hidden": (128, 64, 128)}

# TDT: 2 layers, 4 heads, d_model = pc_dim = 256
DEFAULT_TDT_CFG = {"future_steps": 8, "layers": 2, "heads": 4,
                   "dim_ff": 1024, "dropout": 0.0}


def _as_plain_dict(cfg):
    """DictConfig / dict / None -> plain python dict."""
    if not cfg:
        return {}
    if hasattr(cfg, "items"):
        return {str(k): v for k, v in cfg.items()}
    return dict(cfg)


def _last_linear_dim(module):
    """output width of the last nn.Linear inside ``module``."""
    for m in reversed(list(module.modules())):
        if isinstance(m, nn.Linear):
            return int(m.out_features)
    raise ValueError("module has no nn.Linear layer")


class PIJEPABottleNeck(nn.Module):
    """Per-point residual bottleneck conditioned on the state feature.

    ``[f_i(256) || s(64)] -> Linear(320,128)+GELU -> 128 -> 64 -> 128 -> 256`` and
    ``f_i' = f_i + delta``.  The last layer is zero-initialised, so the module is
    the identity at initialisation (which makes the whole encoder DP3-identical at
    step 0 while still giving the denoiser full gradient from step 1).
    """

    def __init__(self, pc_dim=256, state_dim=64, hidden=(128, 64, 128)):
        super().__init__()
        self.pc_dim = int(pc_dim)
        self.state_dim = int(state_dim)
        self.hidden = tuple(int(h) for h in (hidden or ()))
        layers = []
        prev = self.pc_dim + self.state_dim
        for width in self.hidden:
            layers += [nn.Linear(prev, width), nn.GELU()]
            prev = width
        layers.append(nn.Linear(prev, self.pc_dim))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, pc_feat, state_feat):
        if pc_feat.dim() != 3:
            raise ValueError("pc_feat must be (M,N,{})", self.pc_dim)
        if state_feat.dim() != 2:
            raise ValueError("state_feat must be (M,{})".format(self.state_dim))
        state_exp = state_feat.unsqueeze(1).expand(-1, pc_feat.shape[1], -1)
        x = torch.cat([pc_feat, state_exp], dim=-1)      # (M, N, 320)
        return pc_feat + self.net(x)                     # residual: identity at init



class PIJEPAEncoder(DP3Encoder):
    """DP3Encoder + PIJEPA-BottleNeck between the point MLP and the pooling.

    ``extractor`` and ``state_mlp`` are taken over from an existing ``DP3Encoder``
    (no rebuild, no weight loss), so the state-dict keys stay
    ``obs_encoder.extractor.*`` / ``obs_encoder.state_mlp.*``; only
    ``obs_encoder.bottleneck.*`` is new.
    """

    def __init__(self, base, bottleneck_cfg=None):
        nn.Module.__init__(self)                 # skip DP3Encoder.__init__
        if not isinstance(base, DP3Encoder):
            raise TypeError("PIJEPAEncoder needs a DP3Encoder, got {}".format(type(base)))
        if base.use_imagined_robot:
            raise NotImplementedError("PIJEPAEncoder does not support imagination input")
        # --- shared trunk (identical to DP3) ----------------------------------
        self.imagination_key = base.imagination_key
        self.state_key = base.state_key
        self.point_cloud_key = base.point_cloud_key
        self.rgb_image_key = base.rgb_image_key
        self.n_output_channels = base.n_output_channels
        self.use_imagined_robot = base.use_imagined_robot
        self.point_cloud_shape = base.point_cloud_shape
        self.state_shape = base.state_shape
        self.imagination_shape = base.imagination_shape
        self.use_pc_color = base.use_pc_color
        self.pointnet_type = base.pointnet_type
        self.extractor = base.extractor
        self.state_mlp = base.state_mlp
        # --- new per-point bottleneck -----------------------------------------
        cfg = dict(DEFAULT_BOTTLENECK_CFG)
        cfg.update(_as_plain_dict(bottleneck_cfg))
        self.bottleneck = PIJEPABottleNeck(pc_dim=int(self.extractor.pc_dim),
                                           state_dim=_last_linear_dim(self.state_mlp),
                                           **cfg)

    # -- shared trunk ---------------------------------------------------------
    def state_feature(self, observations):
        return self.state_mlp(observations[self.state_key])

    def point_feats(self, observations):
        """(M,N,256) per-point features before the bottleneck."""
        points = observations[self.point_cloud_key]
        if not self.use_pc_color:
            points = points[..., :3]
        return self.extractor.mlp(points)

    def bottleneck_feats(self, observations, state_feat=None):
        """(M,N,256) per-point features after the bottleneck."""
        s = self.state_feature(observations) if state_feat is None else state_feat
        return self.bottleneck(self.point_feats(observations), s)

    def pooled(self, observations, state_feat=None, use_bottleneck=True):
        """(M,256) softmax-pooled feature (``use_bottleneck=False`` -> pre-bottleneck)."""
        s = self.state_feature(observations) if state_feat is None else state_feat
        feats = self.bottleneck_feats(observations, s) if use_bottleneck \
            else self.point_feats(observations)
        return self.extractor._pool(feats)

    # -- deploy path ----------------------------------------------------------
    def forward(self, observations, state_feat=None):
        """DP3 encoder output (M,128) of the shared, bottleneck-augmented branch."""
        s = self.state_feature(observations) if state_feat is None else state_feat
        z = self.pooled(observations, s, use_bottleneck=True)
        return torch.cat([self.extractor.final_projection(z), s], dim=-1)



class PIJEPAPolicy(DP3):
    """DP3 whose ``obs_encoder`` carries the PI-JEPA bottleneck.

    ``predict_action`` / ``compute_loss`` / normalizer are inherited unchanged, so
    ``eval.py`` evaluates the shared deploy branch with no extra hooks.
    """

    def __init__(self, *args, bottleneck_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_encoder = PIJEPAEncoder(self.obs_encoder, bottleneck_cfg)
        n_bn = sum(p.numel() for p in self.obs_encoder.bottleneck.parameters())
        print("[PIJEPAPolicy] bottleneck {} params, hidden={}".format(
            n_bn, self.obs_encoder.bottleneck.hidden))

    # -- accessors ------------------------------------------------------------
    @property
    def bottleneck(self):
        return self.obs_encoder.bottleneck

    @property
    def bottleneck_cfg(self):
        return {"hidden": list(self.obs_encoder.bottleneck.hidden)}

    # -- encoder paths --------------------------------------------------------
    def encode_state(self, nobs):
        """(M,S) -> (M,64) state feature (computed once, shared by both views)."""
        return self.obs_encoder.state_feature(nobs)

    def encode_obs(self, nobs, state_feat=None):
        """-> (z (M,256), z_pc (M,64), state_feat (M,64)) of the shared branch."""
        s = self.encode_state(nobs) if state_feat is None else state_feat
        z = self.obs_encoder.pooled(nobs, s, use_bottleneck=True)
        return z, self.obs_encoder.extractor.final_projection(z), s

    def encode_target(self, nobs, state_feat=None):
        """(M,256) pre-bottleneck pooled clean feature, detached (SmoothL1 target)."""
        s = self.encode_state(nobs) if state_feat is None else state_feat
        with torch.no_grad():
            z = self.obs_encoder.pooled(nobs, s, use_bottleneck=False)
        return z.detach()


# ------------------------------------------------------------------- v5 (TDT)
# PI-JEPA v5 = the classes above + a frame-level temporal dynamics predictor.
# Train-only: PIJEPAEncoderV5.forward is inherited, so deploy == PIJEPAPolicy.

class PIJEPATemporalDynamics(nn.Module):
    """Frame-level temporal dynamics predictor (train only).

    ``(B, To, D) -> (B, To+F, D)``: the observed frame features are concatenated with
    ``F`` learned future queries, a frame-index positional embedding is added and a
    bidirectional Transformer encoder mixes them (no causal mask, so the current
    frames and the future queries attend to each other).  ``temporal_pe`` indexes
    frames, not points, so it carries no point-order assumption.
    """

    def __init__(self, d_model=256, n_obs=2, future_steps=8, layers=2, heads=4,
                 dim_ff=1024, dropout=0.0):
        super().__init__()
        self.d_model = int(d_model)
        self.n_obs = int(n_obs)
        self.future_steps = int(future_steps)
        self.layers = int(layers)
        self.heads = int(heads)
        self.dim_ff = int(dim_ff)
        self.dropout = float(dropout)
        if self.n_obs < 1 or self.future_steps < 1:
            raise ValueError("TDT needs n_obs >= 1 and future_steps >= 1, got {} / {}"
                             .format(self.n_obs, self.future_steps))
        if self.d_model % self.heads:
            raise ValueError("d_model {} must be divisible by heads {}".format(
                self.d_model, self.heads))
        self.temporal_pe = nn.Embedding(self.n_obs + self.future_steps, self.d_model)
        self.future_query = nn.Parameter(torch.randn(self.future_steps, self.d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.heads, dim_feedforward=self.dim_ff,
            dropout=self.dropout, activation="gelu", batch_first=True)
        # enable_nested_tensor=False: no nested fast path, keeps the plain (B,L,D) API
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.layers,
                                             enable_nested_tensor=False)
        self.head = nn.Linear(self.d_model, self.d_model)

    def cfg(self):
        return {"future_steps": self.future_steps, "layers": self.layers,
                "heads": self.heads, "dim_ff": self.dim_ff, "dropout": self.dropout}

    def forward(self, z):
        """``z`` (B, n_obs, D) -> (B, n_obs + F, D)."""
        if z.dim() != 3:
            raise ValueError("TDT input must be (B,{},D), got {}".format(
                self.n_obs, tuple(z.shape)))
        if int(z.shape[1]) != self.n_obs:
            raise ValueError("TDT was built for n_obs={} but got {} frames; rebuild the "
                             "policy with the training n_obs_steps".format(
                                 self.n_obs, int(z.shape[1])))
        if int(z.shape[-1]) != self.d_model:
            raise ValueError("TDT expects d_model={}, got {}".format(
                self.d_model, int(z.shape[-1])))
        q = self.future_query.unsqueeze(0).expand(z.shape[0], -1, -1)   # (B,F,D)
        seq = torch.cat([z, q], dim=1)                                  # (B,To+F,D)
        seq = seq + self.temporal_pe.weight.unsqueeze(0)
        return self.head(self.encoder(seq))                             # (B,To+F,D)


class PIJEPAEncoderV5(PIJEPAEncoder):
    """``PIJEPAEncoder`` + frame-level pooling + the TDT.

    ``mlp`` / ``bottleneck`` / ``final_projection`` / ``forward`` are inherited from
    ``PIJEPAEncoder``, so the deploy path (and the state-dict keys
    ``obs_encoder.{extractor,bottleneck}.*``) are unchanged; only
    ``obs_encoder.tdt.*`` is new.
    """

    def __init__(self, base, bottleneck_cfg=None, tdt_cfg=None, n_obs=2,
                 tdt_pool="task"):
        super().__init__(base, bottleneck_cfg)
        if str(tdt_pool) not in ("task", "mean"):
            raise ValueError("tdt_pool must be 'task' (softmax/attention pool, the "
                             "same feature the UNet sees) or 'mean', got {}"
                             .format(tdt_pool))
        # plain attribute (no parameter) -> checkpoints stay interchangeable
        self.tdt_pool = str(tdt_pool)
        cfg = _as_plain_dict(DEFAULT_TDT_CFG)
        cfg.update(_as_plain_dict(tdt_cfg))
        self.tdt = PIJEPATemporalDynamics(d_model=int(self.extractor.pc_dim),
                                          n_obs=int(n_obs), **cfg)

    # -- accessors ------------------------------------------------------------
    @property
    def tdt_cfg(self):
        return self.tdt.cfg()

    # -- frame-level pooling --------------------------------------------------
    @staticmethod
    def as_frames(feats, batch, frames):
        """(M,N,C) with M = B*frames -> (B,frames,N,C)."""
        batch, frames = int(batch), int(frames)
        if feats.shape[0] != batch * frames:
            raise ValueError("expected {}*{}={} rows, got {}".format(
                batch, frames, batch * frames, feats.shape[0]))
        return feats.reshape(batch, frames, *feats.shape[1:])

    def frame_feats(self, observations, state_feat=None, use_bottleneck=True):
        """(M,N,256) per-point features, optionally after the bottleneck."""
        feats = self.point_feats(observations)
        if not use_bottleneck:
            return feats
        s = self.state_feature(observations) if state_feat is None else state_feat
        return self.bottleneck(feats, s)

    def pool_mean(self, feats4d):
        """(B,To,N,256) -> (B,To,256): plain mean over the points (``tdt_pool='mean'``)."""
        return feats4d.mean(dim=2)

    def pool_task(self, feats4d, state4d=None):
        """(B,To,N,256) -> (B,To,256): DP3 ``_pool`` per frame (task feature/target)."""
        b, f, n, c = feats4d.shape
        flat = feats4d.reshape(b * f, n, c)
        state = None if state4d is None else state4d.reshape(b * f, -1)
        return self.extractor._pool(flat, state).reshape(b, f, c)


class PIJEPAPolicyV5(PIJEPAPolicy):
    """Two-view policy + TDT: clean/poll/future features in one call (train helper).

    ``predict_action`` / ``compute_loss`` / ``obs_encoder.forward`` are inherited, so
    deployment is bit-identical to ``PIJEPAPolicy`` (and to DP3 at step 0, where the
    bottleneck is the identity and the zero-init softmax scores give mean pooling).
    """

    def __init__(self, *args, bottleneck_cfg=None, tdt_cfg=None, tdt_pool="task",
                 **kwargs):
        DP3.__init__(self, *args, **kwargs)          # skip PIJEPAPolicy's wrap
        self.obs_encoder = PIJEPAEncoderV5(self.obs_encoder, bottleneck_cfg,
                                           tdt_cfg=tdt_cfg,
                                           n_obs=int(self.n_obs_steps),
                                           tdt_pool=tdt_pool)
        n_bn = sum(p.numel() for p in self.obs_encoder.bottleneck.parameters())
        n_tdt = sum(p.numel() for p in self.obs_encoder.tdt.parameters())
        print("[PIJEPAPolicyV5] bottleneck {} params (hidden={}), TDT {} params "
              "({}, input pool={})"
              .format(n_bn, self.obs_encoder.bottleneck.hidden, n_tdt,
                      self.obs_encoder.tdt_cfg, self.obs_encoder.tdt_pool))

    # -- accessors ------------------------------------------------------------
    @property
    def tdt(self):
        return self.obs_encoder.tdt

    @property
    def tdt_cfg(self):
        return self.obs_encoder.tdt_cfg

    @property
    def tdt_pool(self):
        return self.obs_encoder.tdt_pool

    # -- training paths -------------------------------------------------------
    def frame_z(self, nobs, state_feat, to):
        """-> (z_seq (B,To,256) TDT input, z_task (B,To,256), feats4d (B,To,N,256))."""
        to = int(to)
        m = nobs[self.obs_encoder.point_cloud_key].shape[0]
        if m % to:
            raise ValueError("M={} is not a multiple of To={}".format(m, to))
        b = m // to
        feats = self.obs_encoder.frame_feats(nobs, state_feat, use_bottleneck=True)
        feats4d = PIJEPAEncoderV5.as_frames(feats, b, to)
        state4d = state_feat.reshape(b, to, -1)
        z_task = self.obs_encoder.pool_task(feats4d, state4d)
        if self.obs_encoder.tdt_pool == "task":
            z_seq = z_task                  # softmax/attention pool == the UNet feature
        else:
            z_seq = self.obs_encoder.pool_mean(feats4d)
        return z_seq, z_task, feats4d

    def seq_pred(self, z_seq):
        """(B,To,256) -> (B,To+F,256) TDT output."""
        return self.tdt(z_seq)

    def task_cond(self, z_task, state_feat):
        """(B,To,256)+(M,64) -> (M,128) UNet global cond (same as the deploy branch)."""
        m = state_feat.shape[0]
        pc_feat = self.obs_encoder.extractor.final_projection(z_task.reshape(m, -1))
        return torch.cat([pc_feat, state_feat], dim=-1)

    def encode_seq_target(self, nobs, nobs_fut, to, future_steps):
        """(B,To+F,256) stop-grad target from the PRE-bottleneck clean features.

        Both sides pool with ``pool_task`` (softmax over points): with
        ``tdt_pool='task'`` the TDT input is the same task feature the UNet sees and
        the target is its pre-bottleneck counterpart, so the sequence loss compares
        like with like.  ``tdt_pool='mean'`` restores the old mean-pooled TDT input.
        """
        to, fut = int(to), int(future_steps)
        with torch.no_grad():
            f_t = self.obs_encoder.point_feats(nobs)        # (M,N,256)
            f_f = self.obs_encoder.point_feats(nobs_fut)    # (B*F,N,256)
            b = f_t.shape[0] // to
            z_t = self.obs_encoder.pool_task(PIJEPAEncoderV5.as_frames(f_t, b, to))
            z_f = self.obs_encoder.pool_task(PIJEPAEncoderV5.as_frames(f_f, b, fut))
        return torch.cat([z_t, z_f], dim=1).detach()

    def encode_views(self, nobs, nobs_poll, nobs_fut, to, future_steps,
                     state_feat=None):
        """Training path in one call -> every tensor the v5 loss needs."""
        s = self.encode_state(nobs) if state_feat is None else state_feat
        z_seq_c, z_task_c, _ = self.frame_z(nobs, s, to)
        z_seq_p, z_task_p, _ = self.frame_z(nobs_poll, s, to)
        return {
            "state_feat": s,
            "z_seq_clean": z_seq_c, "z_task_clean": z_task_c,
            "z_seq_poll": z_seq_p, "z_task_poll": z_task_p,
            "seq_clean": self.seq_pred(z_seq_c),
            "seq_poll": self.seq_pred(z_seq_p),
            "seq_target": self.encode_seq_target(nobs, nobs_fut, to, future_steps),
            "cond_clean": self.task_cond(z_task_c, s),
            "cond_poll": self.task_cond(z_task_p, s),
        }
