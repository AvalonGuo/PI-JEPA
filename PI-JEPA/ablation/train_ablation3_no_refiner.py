#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Ablation 3 trainer: ``PI-JEPA/train_pijepa.py`` with the Refiner removed.

Data, losses, TDT, warmup, EMA, the loop and the checkpoint format are the upstream v5
ones -- only two hooks are swapped, so nothing else can drift:

  * ``train_pijepa.build_policy`` -> ``ablation3_policy.PIJEPAPolicyV5NoRefiner``
    (``--bn-hidden`` is accepted and ignored: the refiner is not built)
  * ``pijepa_common.save_ckpt``   -> rewrites the ``cfg.pijepa_v5`` provenance block
    (``bottleneck_cfg = {hidden: [], type: none}``); ``latest.ckpt`` and ``best.ckpt``
    both go through this writer (``maybe_save_best`` calls the module-level name).

It also drops ``ablation3_no_refiner.json`` into the run dir, because the launcher's
``ablation.json`` is a one-line claim marker that still records ``bn_hidden``.

Run it through the launcher:

    bash PI-JEPA/ablation/train_ablation3_no_refiner.sh <task> <seed> <gpu> ...
"""
import json
import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parents[2]
for _p in (str(_REPO / "3D-Diffusion-Policy"), str(_REPO), str(_REPO / "PI-JEPA"),
           str(_REPO / "PI-JEPA" / "ablation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hydra  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import pijepa_common as common  # noqa: E402
import train_pijepa as trainer  # noqa: E402

TARGET = "ablation3_policy.PIJEPAPolicyV5NoRefiner"
SIDECAR = "ablation3_no_refiner.json"


def build_policy(cfg, bottleneck_cfg, tdt_cfg, tdt_pool="task"):
    """``train_pijepa.build_policy`` with the refiner-free policy class."""
    pcfg = OmegaConf.create(OmegaConf.to_container(cfg.policy, resolve=True))
    OmegaConf.set_struct(pcfg, False)
    pcfg._target_ = TARGET
    pcfg.bottleneck_cfg = OmegaConf.create({"hidden": []})   # ignored by the class
    pcfg.tdt_cfg = OmegaConf.create({k: v for k, v in dict(tdt_cfg).items()})
    pcfg.tdt_pool = str(tdt_pool)
    return hydra.utils.instantiate(pcfg)


_save_ckpt = common.save_ckpt            # captured before the patch below


def save_ckpt(path, cfg, policy, ema_policy, optimizer, epoch, global_step, out_dir,
              **kwargs):
    """Rewrite ``cfg.pijepa_v5`` in place (``main`` already cleared the struct)."""
    try:
        v5 = cfg.pijepa_v5
        OmegaConf.set_struct(v5, False)
        v5.bottleneck_cfg = {"hidden": [], "type": "none"}
        v5.bottleneck_init = "none (ablation3: refiner removed)"
        v5.encoder = "DP3 extractor.mlp + frame pools + final_proj (no refiner)"
        v5.deploy = ("mlp -> softmax pool -> final_proj|state -> UNet "
                     "(no refiner; TDT is train-only)")
    except Exception as exc:                       # noqa: BLE001
        print("[ablation3] WARN: could not rewrite cfg.pijepa_v5 ({})".format(exc))
    return _save_ckpt(path, cfg, policy, ema_policy, optimizer, epoch, global_step,
                      out_dir, **kwargs)


def _drop_sidecar(argv):
    """Write the ablation-3 provenance next to the checkpoints (best effort)."""
    out = ""
    for i, arg in enumerate(argv):
        if arg == "--out" and i + 1 < len(argv):
            out = argv[i + 1]
        elif arg.startswith("--out="):
            out = arg.split("=", 1)[1]
    if not out:
        return
    payload = {
        "ablation": "3",
        "name": "w/o Refiner",
        "refiner": "none",
        "policy_target": TARGET,
        "encoder": "extractor.mlp -> softmax pool -> final_proj|state -> UNet",
        "tdt": "kept (train-only); target = pool_task(mlp(obs|future)), stop-grad",
        "loss": "v5 defaults: the refiner is the only change",
        "note": "obs_encoder.bottleneck is None -> no bottleneck.* keys in the ckpt",
    }
    try:
        path = pathlib.Path(out)
        path.mkdir(parents=True, exist_ok=True)
        (path / SIDECAR).write_text(json.dumps(payload, indent=2) + "\n")
        print("[ablation3] provenance -> {}".format(path / SIDECAR))
    except Exception as exc:                       # noqa: BLE001
        print("[ablation3] WARN: could not write {} ({})".format(SIDECAR, exc))


def main(argv=None):
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    trainer.build_policy = build_policy
    common.save_ckpt = save_ckpt               # also reached from maybe_save_best()
    print("[ablation3] refiner=none | v5 loop via train_pijepa.py with build_policy + "
          "save_ckpt patched (PI-JEPA/ablation/train_ablation3_no_refiner.py)")
    _drop_sidecar(argv)
    return trainer.main(argv)


if __name__ == "__main__":
    sys.exit(main())
