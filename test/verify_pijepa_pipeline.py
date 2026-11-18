#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Verify the PI-JEPA pipeline wiring (CPU only, no sim / no training).

    python test/verify_pijepa_pipeline.py [--num-points 512] [--batch 3]

Checks
  C1 shape        : point feats (M,N,256), z (M,256), z_pc (M,64), cond (M,128)
  C2 zero-init    : bottleneck is the identity at init, the encoder output is
                    bit-identical to DP3 and a DP3 state dict loads with
                    strict=False (missing keys == obs_encoder.bottleneck.*)
  C3 residual     : a non-zero last layer changes both bottleneck and encoder output
  C4 grad routing : L2_clean / L2_poll both reach mlp + bottleneck + pool score +
                    final_proj + state_mlp + UNet (shared weights); the SmoothL1
                    terms additionally reach the bottleneck only (target detached)
  C5 spec dims    : 320 -> 128 -> 64 -> 128 -> 256, GELU, zero-init last layer
  C6 train step   : loss == w_c*L2c + w_p*L2p (+ lambda*(L1c+L1p)), params/EMA update
  C7 deploy       : obs_encoder(obs) == manual mlp->bn->pool->proj|state, and
                    predict_action runs on the shared branch
"""
import argparse
import copy
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

_REPO = pathlib.Path(__file__).resolve().parents[1]
_DP3_DIR = _REPO / "3D-Diffusion-Policy"
for _p in (str(_DP3_DIR), str(_REPO), str(_REPO / "PI-JEPA")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pijepa_policy import PIJEPAPolicy  # noqa: E402
from diffusion_policy_3d.policy.dp3 import DP3  # noqa: E402


def _pc_cfg():
    return OmegaConf.create({
        "in_channels": 3, "out_channels": 64, "use_layernorm": True,
        "final_norm": "layernorm", "normal_channel": False,
        "pool_type": "softmax", "pool_temperature": 1.0, "pool_score_hidden": 0,
    })


def _scheduler():
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    return DDIMScheduler(num_train_timesteps=100, beta_start=1e-4, beta_end=0.02,
                         beta_schedule="squaredcos_cap_v2", clip_sample=True,
                         set_alpha_to_one=True, steps_offset=0,
                         prediction_type="sample")


def _policy_kwargs(n_points, state_dim, action_dim):
    # shape_meta must be a DictConfig (dict_apply() recurses into plain dicts and
    # would call x['shape'] on the leaf lists)
    shape_meta = OmegaConf.create({
        "obs": {"point_cloud": {"shape": [n_points, 3], "type": "point_cloud"},
                "agent_pos": {"shape": [state_dim], "type": "low_dim"}},
        "action": {"shape": [action_dim]},
    })
    return dict(
        shape_meta=shape_meta,
        noise_scheduler=_scheduler(),
        horizon=16, n_action_steps=8, n_obs_steps=2, num_inference_steps=10,
        obs_as_global_cond=True, diffusion_step_embed_dim=64,
        down_dims=[64, 128], kernel_size=5, n_groups=8, condition_type="film",
        use_down_condition=True, use_mid_condition=True, use_up_condition=True,
        encoder_output_dim=64, crop_shape=[80, 80],
        use_pc_color=False, pointnet_type="pointnet",
        pointcloud_encoder_cfg=_pc_cfg(),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--num-points", type=int, default=512)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--state-dim", type=int, default=9)
    ap.add_argument("--action-dim", type=int, default=4)
    a = ap.parse_args(argv)

    torch.manual_seed(0)
    N, M, S = int(a.num_points), int(a.batch) * 2, int(a.state_dim)   # M = B * To
    dp3 = DP3(**_policy_kwargs(N, S, a.action_dim)).eval()
    pij = PIJEPAPolicy(**_policy_kwargs(N, S, a.action_dim),
                       bottleneck_cfg={"hidden": [128, 64, 128]}).eval()

    obs = {"point_cloud": torch.randn(M, N, 3), "agent_pos": torch.randn(M, S)}
    return _checks(a, dp3, pij, obs, N, M, S)


def _checks(a, dp3, pij, obs, N, M, S):
    # C1 shapes ---------------------------------------------------------------
    f = pij.obs_encoder.point_feats(obs)
    z, z_pc, s = pij.encode_obs(obs)
    assert tuple(f.shape) == (M, N, 256), f.shape
    assert tuple(z.shape) == (M, 256), z.shape
    assert tuple(z_pc.shape) == (M, 64), z_pc.shape
    assert tuple(s.shape) == (M, 64), s.shape
    cond = torch.cat([z_pc, s], dim=-1)
    assert tuple(cond.shape) == (M, 128), cond.shape
    assert tuple(pij.obs_encoder(obs).shape) == (M, 128)
    print("C1 PASS shape: feats{} z{} z_pc{} cond{}".format(
        tuple(f.shape), tuple(z.shape), tuple(z_pc.shape), tuple(cond.shape)))

    # C2 zero-init identity + DP3 parity -------------------------------------
    missing, unexpected = pij.load_state_dict(dp3.state_dict(), strict=False)
    assert not unexpected, unexpected
    assert missing and all(k.startswith("obs_encoder.bottleneck.") for k in missing), \
        missing
    with torch.no_grad():
        bn_out = pij.bottleneck(f, s)
        assert torch.equal(bn_out, f), (bn_out - f).abs().max()
        a_ref = dp3.obs_encoder(obs)
        a_new = pij.obs_encoder(obs)
        assert torch.equal(a_ref, a_new), (a_ref - a_new).abs().max()
    print("C2 PASS zero-init: bottleneck==identity, encoder==DP3 bit-identical, "
          "{} extra keys".format(len(missing)))

    # C3 residual actually changes things ------------------------------------
    with torch.no_grad():
        pij.bottleneck.net[-1].weight.data.normal_(std=0.01)
        bn_out2 = pij.bottleneck(f, s)
        assert not torch.allclose(bn_out2, f)
        a_new2 = pij.obs_encoder(obs)
        assert not torch.allclose(a_new2, a_ref)
    print("C3 PASS residual: non-zero last layer changes bottleneck + encoder output")

    # ---- shared setup for the loss-level checks (CPU) ----------------------
    from pijepa_policy import PIJEPABottleNeck
    from pijepa_common import (diffusion_loss, make_targets, train_step)
    from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler
    from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
    from diffusion_policy_3d.model.diffusion.ema_model import EMAModel

    T, Da, B, To = 16, int(a.action_dim), 2, 2
    stats = {"action": np.random.randn(8, T, Da).astype(np.float32),
             "agent_pos": np.random.randn(8, S).astype(np.float32),
             "point_cloud": np.random.randn(8, N, 6).astype(np.float32)}
    normalizer = LinearNormalizer()
    normalizer.fit(data=stats, last_n_dims=1, mode="limits")
    pij.set_normalizer(normalizer)
    ema_policy = copy.deepcopy(pij)
    ema_policy.set_normalizer(normalizer)
    ema = EMAModel(model=ema_policy, update_after_step=0, inv_gamma=1.0, power=0.75,
                   min_value=0.0, max_value=0.9999)
    optimizer = torch.optim.AdamW(pij.parameters(), lr=1e-3, betas=(0.95, 0.999))
    lr_sched = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=0,
                             num_training_steps=1000)
    action = torch.randn(B, T, Da)
    nobs_c = normalizer.normalize({"point_cloud": torch.randn(B * To, N, 6),
                                   "agent_pos": torch.randn(B * To, S)})
    nobs_p = normalizer.normalize({"point_cloud": torch.randn(B * To, N, 6),
                                   "agent_pos": torch.randn(B * To, S)})
    noisy, ts, target, mask = make_targets(pij, action)

    def _grad_names():
        return {n for n, p in pij.named_parameters() if p.grad is not None}

    def _touched(names):
        pre = ("obs_encoder.extractor.mlp", "obs_encoder.extractor.score_mlp",
               "obs_encoder.extractor.final_projection", "obs_encoder.bottleneck",
               "obs_encoder.state_mlp", "model.")
        return [p for p in pre if any(n.startswith(p) for n in names)]

    # C4 grad routing (shared weights, both views) ----------------------------
    all_pre = ("obs_encoder.extractor.mlp", "obs_encoder.extractor.score_mlp",
               "obs_encoder.extractor.final_projection", "obs_encoder.bottleneck",
               "obs_encoder.state_mlp", "model.")
    for tag, nobs_v in (("clean", nobs_c), ("poll", nobs_p)):
        pij.zero_grad(set_to_none=True)
        state_feat = pij.encode_state(nobs_v)
        _, z_pc_v, _ = pij.encode_obs(nobs_v, state_feat)
        cond_v = torch.cat([z_pc_v, state_feat], dim=-1)
        diffusion_loss(pij, noisy, ts, target, mask, cond_v).backward()
        hit = _touched(_grad_names())
        assert len(hit) == len(all_pre), "L2_{} missed {}".format(
            tag, sorted(set(all_pre) - set(hit)))
    # SmoothL1 touches the bottleneck as well, with a detached target
    pij.zero_grad(set_to_none=True)
    state_feat = pij.encode_state(nobs_c)
    z_c, _, _ = pij.encode_obs(nobs_c, state_feat)
    tg = pij.encode_target(nobs_c, state_feat)
    assert not tg.requires_grad
    torch.nn.functional.smooth_l1_loss(z_c, tg).backward()
    hit = _touched(_grad_names())
    assert "obs_encoder.bottleneck" in hit and "model." not in hit
    print("C4 PASS grad routing: L2_clean/L2_poll reach {} groups (shared weights); "
          "SmoothL1 -> bottleneck only".format(len(all_pre)))

    # C5 bottleneck spec ------------------------------------------------------
    bn = pij.bottleneck
    assert bn.pc_dim == 256 and bn.state_dim == 64
    linears = [(m.in_features, m.out_features) for m in bn.net if isinstance(m, nn.Linear)]
    assert linears == [(320, 128), (128, 64), (64, 128), (128, 256)], linears
    assert all(isinstance(bn.net[i], nn.GELU) for i in (1, 3, 5))
    fresh = PIJEPABottleNeck(pc_dim=256, state_dim=64, hidden=(128, 64, 128))
    assert bool((fresh.net[-1].weight == 0).all())
    assert bool((fresh.net[-1].bias == 0).all())
    with torch.no_grad():
        f5 = torch.randn(M, N, 256)
        s5 = torch.randn(M, 64)
        y5 = fresh(f5, s5)
        assert torch.equal(y5, f5)                     # identity at init
        assert tuple(y5.shape) == (M, N, 256)
    print("C5 PASS spec: 320->128->64->128->256 (GELU), zero-init last layer, "
          "per-point residual")

    # C6 one real training step (CPU) ----------------------------------------
    w_c, w_p = 1.0, 0.5
    before_bn = {k: v.clone() for k, v in bn.state_dict().items()}
    before_unet = next(pij.model.parameters()).detach().clone()
    batch = {"obs": {"point_cloud": torch.randn(B, To, N, 6),
                     "agent_pos": torch.randn(B, To, S)},
             "obs_poll": torch.randn(B, To, N, 6),
             "action": action}
    dev = torch.device("cpu")
    loss, l2c, l2p, l1c, l1p = train_step(pij, ema, optimizer, normalizer, batch,
                                         To, dev, lr_sched, w_c, w_p, 0.0, 0.0)
    assert l1c is None and l1p is None
    assert np.isfinite([loss, l2c, l2p]).all()
    assert abs(loss - (w_c * l2c + w_p * l2p)) < 1e-4, (loss, l2c, l2p)
    moved = [k for k, v in bn.state_dict().items() if not torch.equal(v, before_bn[k])]
    assert moved, "bottleneck did not update"
    assert not torch.equal(next(pij.model.parameters()), before_unet)
    loss2, l2c2, l2p2, l1c2, l1p2 = train_step(pij, ema, optimizer, normalizer, batch,
                                              To, dev, lr_sched, w_c, w_p, 1.0, 1.0)
    assert l1c2 is not None and l1p2 is not None and l1c2 > 0
    assert abs(loss2 - (w_c * l2c2 + w_p * l2p2 + l1c2 + l1p2)) < 1e-3
    assert torch.equal(ema_policy.bottleneck.net[-1].weight,
                       pij.bottleneck.net[-1].weight)      # decay 0 at step 0/1
    assert lr_sched.get_last_lr()[0] > 0.0
    # one-sided terms: the disabled view's SmoothL1 must not be computed at all
    _, _, _, l1c_a, l1p_a = train_step(pij, ema, optimizer, normalizer, batch,
                                       To, dev, lr_sched, w_c, w_p, 1.0, 0.0)
    assert l1c_a is not None and l1p_a is None
    _, _, _, l1c_b, l1p_b = train_step(pij, ema, optimizer, normalizer, batch,
                                       To, dev, lr_sched, w_c, w_p, 0.0, 1.0)
    assert l1c_b is None and l1p_b is not None
    print("C6 PASS train step: loss={:.4f} (L2c={:.4f} L2p={:.4f}) | L1 loss={:.4f} "
          "(L1c={:.4f} L1p={:.4f}) | L1c only={:.4f} | L1p only={:.4f}, "
          "ema synced".format(loss, l2c, l2p, loss2, l1c2, l1p2, l1c_a, l1p_b))

    # C7 deploy == the shared training branch --------------------------------
    with torch.no_grad():
        s_d = pij.obs_encoder.state_feature(obs)
        manual = torch.cat([pij.obs_encoder.extractor.final_projection(
            pij.obs_encoder.pooled(obs, s_d, use_bottleneck=True)), s_d], dim=-1)
        fwd = pij.obs_encoder(obs)
    assert torch.equal(manual, fwd), (manual - fwd).abs().max()
    obs_dict = {"point_cloud": torch.randn(1, 2, N, 3),
                "agent_pos": torch.randn(1, 2, S)}
    with torch.no_grad():
        out = pij.predict_action(obs_dict)
    assert tuple(out["action"].shape) == (1, 8, int(a.action_dim)), out["action"].shape
    assert bool(torch.isfinite(out["action"]).all())
    print("C7 PASS deploy: encoder forward == manual mlp->bn->pool->proj|state; "
          "predict_action {} finite".format(tuple(out["action"].shape)))

    # C8 best-checkpoint bookkeeping (single overwritten best.ckpt) --------------
    import json as _json
    import shutil
    import tempfile
    import dill
    from pijepa_common import maybe_save_best
    from diffusion_policy_3d.common.checkpoint_util import TopKCheckpointManager

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="pijepa_best_"))
    ck_dir = tmp / "checkpoints"
    ck_dir.mkdir(parents=True)
    topk = TopKCheckpointManager(
        save_dir=str(ck_dir), monitor_key="test_mean_score", mode="max", k=1)
    alias = ck_dir / "best.ckpt"
    bjson = ck_dir / "best.json"
    common = dict(cfg=OmegaConf.create({"dummy": 1}), policy=pij,
                  ema_policy=ema_policy, optimizer=optimizer, global_step=7,
                  out_dir=str(tmp), alias=alias, best_json=bjson,
                  best_monitor="test_mean_score")
    thr = float("-inf")
    r1 = maybe_save_best(topk, thr, "max", 0.40, epoch=10, **common)
    assert r1 is not None, "first score must be saved"
    p1 = pathlib.Path(r1[1])
    assert p1 == alias and p1.is_file() and not alias.is_symlink(), (p1, alias)
    assert maybe_save_best(topk, r1[0], "max", 0.30, epoch=20, **common) is None, \
        "a worse score must not overwrite"
    r2 = maybe_save_best(topk, r1[0], "max", 0.55, epoch=30, **common)
    assert r2 is not None and r2[1] == str(p1), "best.ckpt must be overwritten in place"
    assert sorted(x.name for x in ck_dir.iterdir()) == ["best.ckpt", "best.json"], \
        [x.name for x in ck_dir.iterdir()]
    rec = _json.loads(bjson.read_text())
    assert rec["score"] == 0.55 and rec["epoch"] == 30
    assert rec["file"] == "best.ckpt" and rec["monitor"] == "test_mean_score"
    ck = torch.load(str(alias), pickle_module=dill, map_location="cpu",
                    weights_only=False)
    assert "optimizer" not in ck["state_dicts"], "best ckpt must skip the optimizer"
    assert abs(float(dill.loads(ck["pickles"]["best_score"])) - 0.55) < 1e-9
    shutil.rmtree(tmp)
    print("C8 PASS best ckpt: overwrite-in-place best.ckpt + best.json, "
          "no optimizer state")

    print("ALL CHECKS PASSED (N={}, M={})".format(N, M))
    return 0


if __name__ == "__main__":
    sys.exit(main())



