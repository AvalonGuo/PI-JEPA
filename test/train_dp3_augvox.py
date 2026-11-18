#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DP3 \u7aef\u5230\u7aef\u8bad\u7ec3 + \u5728\u7ebf\u4f53\u7d20\u6c61\u67d3\u589e\u5f3a(voxel -> renderer -> \u70b9\u4e91)
================================================================================
\u76ee\u7684: \u8bc1\u660e\u201cvoxel \u968f\u673a\u91c7\u6837(10-25\u70b9) -> synth_cell_points(renderer\u56de\u70b9\u4e91)
\u5c3e\u9644 -> sample_512\u201d\u662f\u4e00\u79cd\u53ef\u8ba1\u7b97\u7684\u6570\u636e\u589e\u5f3a: \u5728 DP3 BC \u8bad\u7ec3\u65f6\u6309\u6982\u7387 p \u5bf9\u6bcf\u5e27\u73b0\u573a\u6ce8\u5165\u4f53\u7d20\u70b9\u7fa4,
\u8ba9 DP3 \u81ea\u5df1\u5b66\u4f1a\u6297 OOD(\u4e0e\u201c\u51bb\u7ed3 DP3 + \u4fee\u6b63\u5668 g\u201d\u5bf9\u6bd4\u3002

\u57fa\u672c\u8bbe\u8ba1(\u4e0d\u4f7f\u7528 train_policy.sh)
  - \u6570\u636e: clean fullpc 331\u5e27 zarr(state/action/point_cloud/img) AdroitDataset \u7ee7\u627f
  - \u589e\u5f3a: __getitem__ \u65f6\u5bf9\u7a97\u53e3\u6bcf\u5e27\u6309 poll_frac \u73b0\u573a forge_voxel_rows -> sample_512
  - \u8bad\u7ec3: \u590d\u523b\u5b98\u65b9 workspace \u5185\u5faa(DP3.compute_loss/EMA/cosine/checkpoint \u683c\u5f0f\u517c\u5bb9)
  - ckpt: <out>/checkpoints/latest.ckpt payload{cfg,state_dicts{ema_model,model}} -> eval_rollout \u53ef\u76f4\u63a5\u52a0\u8f7d

\u7528\u6cd5(\u7531\u7528\u6237\u8fd0\u884c; \u672c\u811a\u672c\u4e0d\u53d1\u5e03/\u6536\u96c6\u6570\u636e)
  python test/train_dp3_augvox.py --poll-frac 0.5 --epochs 600 --batch 32 \\
      --out /hy-tmp/3DP/z_tsne_out/dp3_augvox_b
  python test/train_dp3_augvox.py --poll-frac 0.0 --epochs 600 --batch 32 \\
      --out /hy-tmp/3DP/z_tsne_out/dp3_augvox_a   # \u5bf9\u7167\u7ec4(A)
"""
import argparse
import copy
import json
import os
import pathlib
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn  # noqa: F401
from omegaconf import OmegaConf

_REPO = pathlib.Path(__file__).resolve().parents[1]       # /root/3DP
_DP3_DIR = _REPO / "3D-Diffusion-Policy"
for _p in (str(_DP3_DIR), str(_REPO), str(pathlib.Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from torch.utils.data import DataLoader  # noqa: E402
import hydra  # noqa: E402
import dill  # noqa: E402

from diffusion_policy_3d.dataset.adroit_dataset import AdroitDataset  # noqa: E402
from diffusion_policy_3d.common.pytorch_util import dict_apply, optimizer_to  # noqa: E402
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel  # noqa: E402
from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler  # noqa: E402
from diffusion_policy_3d.common.pc_sample import sample_512  # noqa: E402
from diffusion_policy_3d.jepa.forge import (  # noqa: E402
    forge_voxel_rows, occupied_cells_from_zarr)
from diffusion_policy_3d.voxel.workspace import (  # noqa: E402
    FORGE_MODES, MODES, resolve_workspace)

DEFAULT_TASK = "adroit_hammer"
DEFAULT_ZARR = "/hy-tmp/3DP/data/adroit_hammer_expert_fullpc.zarr"
DP3_CONFIG_DIR = str(_DP3_DIR / "diffusion_policy_3d" / "config")


OmegaConf.register_new_resolver("eval", eval, replace=True)


class AugVoxAdroitDataset(AdroitDataset):
    """AdroitDataset + \u5728\u7ebf\u4f53\u7d20\u6c61\u67d3\u589e\u5f3a.

    \u6bcf\u6b21 __getitem__: \u5bf9\u7a97\u53e3\u5185\u6bcf\u5e27\u6309 poll_frac \u51b3\u5b9a\u662f\u5426\u628a\u8be5\u5e27\u7684\n    point_cloud(512,6) \u6362\u6210\u201c\u975e\u96f6\u884c + forge_voxel_rows + sample_512\u201d \u7684\u6c61\u67d3\u7248\u3002
    \u6b63\u5219\u5316\u5668\u4ecd\u7528 clean \u5b58\u50a8\u70b9\u4e91\u7edf\u8ba1(\u7ee7\u627f get_normalizer), \u4e0e eval \u4e00\u81f4\u3002
    """

    def __init__(self, poll_frac=0.5, voxel_min=10, voxel_max=25,
                 poll_seed=0, ws=None, forge_mode="bbox", occ_cells=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.poll_frac = float(poll_frac)
        self.voxel_min = int(voxel_min)
        self.voxel_max = int(voxel_max)
        self.ws = ws
        self.forge_mode = str(forge_mode)
        self.occ_cells = occ_cells
        self.rng = np.random.default_rng(int(poll_seed))

    def _pollute_cloud(self, pc):
        base = pc[np.any(pc[:, :3] != 0, axis=1)]
        if base.shape[0] == 0:
            return pc
        extra = forge_voxel_rows(self.rng, p_min=self.voxel_min,
                                 p_max=self.voxel_max, ws=self.ws,
                                 mode=self.forge_mode, occ_cells=self.occ_cells)
        poll = sample_512(np.concatenate([base, extra], axis=0),
                          num_points=512)
        return np.asarray(poll, dtype=np.float32)

    def __getitem__(self, idx):
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        pc = data["obs"]["point_cloud"].astype(np.float32)   # (T,512,6)
        for t in range(pc.shape[0]):
            if float(self.poll_frac) > 0 and self.rng.random() < float(self.poll_frac):
                pc[t] = self._pollute_cloud(pc[t])
        data["obs"]["point_cloud"] = pc
        return dict_apply(data, torch.from_numpy)


def _compose_cfg(args):
    overrides = [
        f"task={args.task}",
        f"training.seed={int(args.seed)}",
        f"training.num_epochs={int(args.epochs)}",
        f"training.use_ema={str(bool(args.use_ema)).lower()}",
        f"dataloader.batch_size={int(args.batch)}",
        f"dataloader.num_workers={int(args.workers)}",
        "dataloader.persistent_workers=False",
        "dataloader.shuffle=True",
    ]
    if args.device:
        overrides.append(f"training.device={args.device}")
    # hydra.initialize_config_dir 接受绝对 config 目录(与调用文件相对路径无关)
    with hydra.initialize_config_dir(
            config_dir=str(pathlib.Path(_DP3_DIR) / "diffusion_policy_3d" / "config"),
            version_base=None):
        cfg = hydra.compose(config_name="dp3", overrides=overrides)
    return cfg


def _save_ckpt(cfg, out_dir, model, ema_model, use_ema):
    def _cpu(sd):
        return {k: v.detach().cpu() for k, v in sd.items()}
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    payload = {
        "cfg": cfg,
        "state_dicts": {
            "model": _cpu(model.state_dict()),
            "ema_model": _cpu(ema_model.state_dict()) if ema_model is not None else {},
        },
        "pickles": {},
    }
    path = os.path.join(ckpt_dir, "latest.ckpt")
    torch.save(payload, path, pickle_module=dill)
    return path


def train(args):
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    cfg = _compose_cfg(args)
    device = torch.device(cfg.training.device)
    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print(f"[dp3aug] dataset zarr={args.zarr}  horizon={int(cfg.horizon)}")
    ws = resolve_workspace(task=str(args.task), zarr=str(args.zarr),
                           mode=str(args.workspace))
    print(f"[dp3aug] {ws.describe()}")
    occ_cells = None
    if str(args.forge_mode) == "occupied":
        occ_cells, _dims = occupied_cells_from_zarr(args.zarr, ws)
        print(f"[dp3aug] forge mode=occupied: {len(occ_cells)} occupied cells")
    dataset = AugVoxAdroitDataset(
        zarr_path=args.zarr,
        horizon=int(cfg.horizon),
        pad_before=int(cfg.n_obs_steps) - 1,
        pad_after=int(cfg.n_action_steps) - 1,
        seed=seed,
        val_ratio=0.0,
        max_train_episodes=None,
        task_name=str(cfg.task.task_name),
        poll_frac=float(args.poll_frac),
        voxel_min=int(args.voxel_min),
        voxel_max=int(args.voxel_max),
        poll_seed=int(args.poll_seed),
        ws=ws,
        forge_mode=str(args.forge_mode),
        occ_cells=occ_cells,
    )
    loader = DataLoader(dataset, **cfg.dataloader)
    print(f"[dp3aug] dataset windows={len(dataset)}  batches/epoch={len(loader)}")

    normalizer = dataset.get_normalizer()
    model = hydra.utils.instantiate(cfg.policy)
    model.set_normalizer(normalizer)
    ema_model = None
    if bool(cfg.training.use_ema):
        try:
            ema_model = copy.deepcopy(model)
        except Exception:  # noqa: BLE001
            ema_model = hydra.utils.instantiate(cfg.policy)
        ema_model.set_normalizer(normalizer)
    model.to(device)
    if ema_model is not None:
        ema_model.to(device)

    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    optimizer_to(optimizer, device)
    num_steps = (len(loader) * int(cfg.training.num_epochs)) \
        // max(1, int(cfg.training.gradient_accumulate_every))
    scheduler = get_scheduler(
        str(cfg.training.lr_scheduler),
        optimizer=optimizer,
        num_warmup_steps=int(cfg.training.lr_warmup_steps),
        num_training_steps=num_steps,
        last_epoch=-1)
    ema = None
    if bool(cfg.training.use_ema):
        ema = hydra.utils.instantiate(cfg.ema, model=ema_model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[dp3aug] DP3 params={n_params/1e6:.1f}M  device={device}  "
          f"poll_frac={args.poll_frac} epochs={int(cfg.training.num_epochs)}")

    global_step = 0
    accum = max(1, int(cfg.training.gradient_accumulate_every))
    ckpt_every = int(getattr(args, "ckpt_every", 0)) or max(1, int(cfg.training.num_epochs) // 10)
    model.train()
    for epoch in range(int(cfg.training.num_epochs)):
        losses = []
        for batch in loader:
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            raw_loss, loss_dict = model.compute_loss(batch)
            (raw_loss / accum).backward()
            if global_step % accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            if ema is not None:
                ema.step(model)
            losses.append(float(raw_loss))
            global_step += 1
            if global_step % 50 == 0:
                print(f"[dp3aug] epoch {epoch} step {global_step} "
                      f"loss={float(raw_loss):.4f} lr={scheduler.get_last_lr()[0]:.2e}")
        avg = float(np.mean(losses)) if losses else float("nan")
        print(f"[dp3aug] epoch {epoch}: avg_loss={avg:.4f}")
        if (epoch + 1) % ckpt_every == 0 or epoch == int(cfg.training.num_epochs) - 1:
            ckpt = _save_ckpt(cfg, args.out, model, ema_model, bool(cfg.training.use_ema))
            print(f"[dp3aug] checkpoint saved -> {ckpt} (step {global_step})")

    summary = {
        "script": "test/train_dp3_augvox.py",
        "task": str(args.task),
        "zarr": args.zarr,
        "out": args.out,
        "workspace": ws.to_dict(),
        "forge_mode": str(args.forge_mode),
        "ckpt": os.path.join(args.out, "checkpoints", "latest.ckpt"),
        "hyper": {"poll_frac": float(args.poll_frac), "voxel_min": int(args.voxel_min),
                  "voxel_max": int(args.voxel_max), "poll_seed": int(args.poll_seed),
                  "epochs": int(cfg.training.num_epochs), "batch": int(args.batch),
                  "seed": int(seed), "use_ema": bool(cfg.training.use_ema)},
        "last_avg_loss": float(avg),
        "global_step": int(global_step),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "train_dp3_augvox_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[dp3aug] done in {summary['elapsed_s']}s -> "
          f"{summary['ckpt']}")
    return summary


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--zarr", default=DEFAULT_ZARR)
    p.add_argument("--task", default=DEFAULT_TASK,
                   help="hydra task config name under diffusion_policy_3d/config/task/")
    p.add_argument("--workspace", choices=MODES, default="auto",
                   help="forger workspace: auto = table / derive from --zarr; "
                        "hammer = historical constants; explicit = lo/hi args")
    p.add_argument("--forge-mode", choices=FORGE_MODES, default="bbox",
                   help="bbox = uniform anchor in the box (historical); "
                        "occupied = anchor among cells occupied by real geometry")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-ema", action="store_true", default=True,
                   help="\u4f7f\u7528 EMA(\u4e0e\u5b98\u65b9\u4e00\u81f4; ckpt \u5b58 ema_model key)")
    p.add_argument("--poll-frac", type=float, default=0.5,
                   help="\u6bcf\u5e27\u88ab\u5728\u7ebf\u4f53\u7d20\u6c61\u67d3\u7684\u6982\u7387; 0=\u7eaf clean(\u5bf9\u7167 A)")
    p.add_argument("--voxel-min", type=int, default=10)
    p.add_argument("--voxel-max", type=int, default=25)
    p.add_argument("--poll-seed", type=int, default=0)
    p.add_argument("--ckpt-every", type=int, default=0,
                   help="\u6bcf N epoch \u4fdd\u5b58\u4e00\u6b21; 0=\u81ea\u52a8(epochs//10)")
    a = p.parse_args()
    train(a)


if __name__ == "__main__":
    main()
