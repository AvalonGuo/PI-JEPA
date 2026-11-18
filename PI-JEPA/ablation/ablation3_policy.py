#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Ablation 3 (w/o Refiner): PI-JEPA v5 with the PIJEPA-BottleNeck removed.

Data, losses, TDT and checkpoint format are the v5 ones; the only change is that the
per-point refiner is not built at all::

    clean, poll : pc -> extractor.mlp -> softmax pool -> final_proj|state -> UNet
    TDT input   : pool_task(mlp(clean|poll))
    TDT target  : pool_task(mlp(obs frames | future frames))      (stop-grad)
    loss        : l2_clean*L2_clean + l2_poll*L2_poll + lambda_jepa_* * SmoothL1

``obs_encoder.bottleneck`` is ``None``: no ``obs_encoder.bottleneck.*`` keys and no
refiner parameters.  Loading an ablation3 checkpoint into v5 fails (torch reports the
missing bottleneck keys), and the reverse direction is caught by the explicit
``load_state_dict`` override below, because torch>=2.4 does *not* report the keys of a
``None``-registered submodule -- they would be dropped in silence.

``encode_seq_target`` is inherited from v5: with no refiner the "pre-bottleneck"
features it pools are the model features themselves (still detached), so the only
asymmetry left between the prediction and the target is the TDT itself.
"""
import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parents[2]
for _p in (str(_REPO / "3D-Diffusion-Policy"), str(_REPO), str(_REPO / "PI-JEPA")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from diffusion_policy_3d.policy.dp3 import DP3  # noqa: E402
from pijepa_policy import PIJEPAEncoderV5, PIJEPAPolicyV5  # noqa: E402

__all__ = ["PIJEPAEncoderV5NoRefiner", "PIJEPAPolicyV5NoRefiner"]


class PIJEPAEncoderV5NoRefiner(PIJEPAEncoderV5):
    """``PIJEPAEncoderV5`` without the bottleneck: ``mlp -> pool -> final_proj``."""

    def __init__(self, base, bottleneck_cfg=None, tdt_cfg=None, n_obs=2,
                 tdt_pool="task"):
        # bottleneck_cfg is accepted (same kwargs as v5) and ignored
        super().__init__(base, bottleneck_cfg, tdt_cfg=tdt_cfg, n_obs=n_obs,
                         tdt_pool=tdt_pool)
        self.bottleneck = None                  # 0 params, no state-dict keys

    def bottleneck_feats(self, observations, state_feat=None):
        """(M,N,256) with no refiner: the point-MLP features themselves."""
        return self.point_feats(observations)

    def frame_feats(self, observations, state_feat=None, use_bottleneck=True):
        """(M,N,256) per frame; ``use_bottleneck`` is a no-op in this ablation."""
        return self.point_feats(observations)


class PIJEPAPolicyV5NoRefiner(PIJEPAPolicyV5):
    """``PIJEPAPolicyV5`` whose obs encoder has ``bottleneck is None`` (ablation 3)."""

    def __init__(self, *args, bottleneck_cfg=None, tdt_cfg=None, tdt_pool="task",
                 **kwargs):
        DP3.__init__(self, *args, **kwargs)          # skip PIJEPAPolicy's wrap
        self.obs_encoder = PIJEPAEncoderV5NoRefiner(
            self.obs_encoder, bottleneck_cfg, tdt_cfg=tdt_cfg,
            n_obs=int(self.n_obs_steps), tdt_pool=tdt_pool)
        n_tdt = sum(p.numel() for p in self.obs_encoder.tdt.parameters())
        print("[PIJEPAPolicyV5NoRefiner] refiner=none (0 params), TDT {} params "
              "({}, input pool={})".format(n_tdt, self.obs_encoder.tdt_cfg,
                                           self.obs_encoder.tdt_pool))

    # -- accessors: the inherited ones assume a non-None bottleneck -------------
    @property
    def bottleneck(self):
        return None

    @property
    def bottleneck_cfg(self):
        return {"hidden": [], "type": "none"}

    # -- guard: torch drops the keys of a None-registered submodule in silence ---
    def load_state_dict(self, state_dict, **kwargs):
        """Reject a v5 / ablation1 / ablation2 checkpoint instead of dropping its
        ``obs_encoder.bottleneck.*`` weights (torch>=2.4 does not report them as
        unexpected when the submodule is registered as ``None``)."""
        extra = [k for k in state_dict if k.startswith("obs_encoder.bottleneck.")]
        if extra:
            raise RuntimeError(
                "{}: this checkpoint still has the refiner ({} obs_encoder."
                "bottleneck.* key(s), e.g. {}) -> not a w/o-Refiner run; evaluate it "
                "with scripts/eval_pijepa.sh or the ablation1/ablation2 wrapper"
                .format(type(self).__name__, len(extra), extra[0]))
        return super().load_state_dict(state_dict, **kwargs)
