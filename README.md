# PI-JEPA

**Breaking the Distractor Hijack: Perturbation-Invariant JEPA for Robust 3D Manipulation**

<p align="center">
  <a href="https://www.alphaxiv.org/abs/2611.pi-jepa-robust-3d-manipulation"><img src="tools/assets/paper.svg" alt="alphaXiv"></a>
  <a href="https://www.modelscope.cn/models/a1339296390/PI-JEPA"><img src="tools/assets/modelscope.svg" alt="🤖 Model: ModelScope"></a>
  <a href="QUICK_START.md"><img src="tools/assets/quickstart.svg" alt="Quick Start: Guide"></a>
  <a href="LICENSE"><img src="tools/assets/license.svg" alt="License: MIT"></a>
</p>

---

## Abstract

PI-JEPA (Perturbation-Invariant Joint-Embedding Predictive Architecture) is a 3D visuomotor
policy for point-cloud manipulation that stays robust when the scene is cluttered with
distractors. Instead of relying on the DP3 max-pooled PointNet feature -- which a handful of
distractor points can hijack -- it learns a perturbation-invariant scene representation by
jointly embedding a clean view and a voxel-cluttered view of the same observation and
predicting future representations, while a lightweight point-level refiner suppresses clutter
before pooling. The refined features drive a diffusion policy, and the full pipeline is
trained end-to-end with the joint-embedding and diffusion action losses. Across category,
position and dynamics distractor suites it improves the average success rate by more than 37
points over DP3, while preserving clean-scene performance.


## Released checkpoints

All runs use the tag `pjv5`, seed `0`, and ship `checkpoints/best.ckpt` (~1.9 GB) either:

| Task | Path inside the model repo |
|---|---|
| `adroit_door` | `adroit_door-pijepa-pjv5_seed0/checkpoints/best.ckpt` |
| `adroit_hammer` | `adroit_hammer-pijepa-pjv5_seed0/checkpoints/best.ckpt` |
| `adroit_pen` | `adroit_pen-pijepa-pjv5_seed0/checkpoints/best.ckpt` |
| `metaworld_assembly` | `metaworld_assembly-pijepa-pjv5_seed0/checkpoints/best.ckpt` |
| `metaworld_disassemble` | `metaworld_disassemble-pijepa-pjv5_seed0/checkpoints/best.ckpt` |
| `metaworld_stick-push` | `metaworld_stick-push-pijepa-pjv5_seed0/checkpoints/best.ckpt` |

```bash
TASK=adroit_door
REMOTE="${TASK}-pijepa-pjv5_seed0/checkpoints/best.ckpt"
uvx --python 3.11 --from modelscope-hub ms-hub download a1339296390/PI-JEPA "$REMOTE" \
    --repo-type model --local-dir "$REPO/outputs" --continue-on-error
bash scripts/eval_pijepa.sh "$TASK" pjv5 0 0        # evaluates best.ckpt by default
```

See [QUICK_START.md](QUICK_START.md#13-download-a-pretrained-checkpoint) for the
`modelscope-hub` details (it needs Python >= 3.10, hence `uvx`).

## Installation

- Recommended: [QUICK_START.md](QUICK_START.md) (uv, Python 3.8, CUDA 12.1).
- Legacy conda recipe from upstream DP3: [INSTALL.md](INSTALL.md).
- In-depth errors: [ERROR_CATCH.md](ERROR_CATCH.md).

## Quick start

```bash
# 1. demonstrations (written to 3D-Diffusion-Policy/data/)
bash scripts/gen_demonstration_adroit.sh hammer
bash scripts/gen_demonstration_metaworld.sh assembly

# 2. train PI-JEPA: <task> <addition_info> <seed> <gpu> [epochs=1000] [zarr_path]
bash scripts/train_pijepa.sh adroit_hammer pjv5 0 0 1000

# 3. evaluate (best.ckpt by default; CKPT=latest for the last epoch)
bash scripts/eval_pijepa.sh adroit_hammer pjv5 0 0
```

## Evaluation protocol

`eval_pijepa.sh <task> <addition_info> <seed> <gpu> [save_video=false] [episodes=50]
[suite=false] [extra_steps=0] [run_tag] [ckpt]`

- Distractor suites: `false` (clean) | `catgen` | `posgen` | `dynsphere` | `dynhand`.
- `extra_steps=0` runs to the environment horizon (the protocol used in the paper);
  `15` reproduces the trainer's in-loop rollout.

```bash
for SUITE in false catgen posgen dynsphere dynhand; do
    bash scripts/eval_pijepa.sh adroit_hammer pjv5 0 0 false 50 "$SUITE" 0
done
```

## Results

Across the five distractor suites (category, position and dynamics variations), PI-JEPA
improves the average success rate by more than **37 points** over DP3, while fully preserving
clean-scene performance. Per-task and per-suite numbers are reported in the
[paper](https://www.alphaxiv.org/abs/2611.pi-jepa-robust-3d-manipulation).

## Citation

```bibtex
@misc{2611.pi-jepa-robust-3d-manipulation,
  title     = {Breaking the Distractor Hijack: Perturbation-Invariant JEPA for Robust 3D Manipulation},
  author    = {Yudi Guo},
  year      = {2026},
  publisher = {alphaXiv},
  url       = {https://www.alphaxiv.org/abs/2611.pi-jepa-robust-3d-manipulation}
}
```

## Acknowledgements

The codebase is built on
[3D Diffusion Policy](https://github.com/YanjieZe/3D-Diffusion-Policy). We thank the authors
for their open-source contributions.

## License

MIT -- see [LICENSE](LICENSE). This repository is derived from
[3D Diffusion Policy](https://github.com/YanjieZe/3D-Diffusion-Policy); the original
copyright notice is retained.
