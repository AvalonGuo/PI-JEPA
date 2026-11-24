# QUICK Start

*Verified on Ubuntu + NVIDIA RTX 4060 Ti, CUDA 12.1, Python 3.8. This is the recommended
installation path. In-depth errors are collected in [ERROR_CATCH.md](ERROR_CATCH.md).*

## 0. Requirements

An NVIDIA GPU (verified on an RTX 4060 Ti) with a CUDA 12.1 driver, and a Linux machine
with the usual C/C++ build toolchain.

## 1. Clone the repository

```bash
git clone https://github.com/AvalonGuo/PI-JEPA.git
cd PI-JEPA
export REPO=$(pwd)
```

## 2. Create the Python 3.8 environment with uv

```bash
pip install uv
uv venv --python 3.8 .venv
source .venv/bin/activate
```

## 3. Install PyTorch (2.4.1, cu121 -- the build this repository was verified with)

```bash
uv pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

For another CUDA version install the matching torch build; nothing else pins a version.

## 4. Install DP3 (editable)

```bash
uv pip install -e 3D-Diffusion-Policy
```

## 5. Install MuJoCo 2.1.0

```bash
mkdir -p ~/.mujoco && cd ~/.mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco210.tar.gz
tar -xvzf mujoco210.tar.gz && rm -f mujoco210.tar.gz
```

Append the following to `~/.bashrc`, then `source ~/.bashrc` and open a new shell:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export MUJOCO_GL=egl
```

## 6. Compiler dependencies + mujoco-py

```bash
sudo apt-get install -y libglew-dev libgl1-mesa-dev libglu1-mesa-dev \
    libosmesa6-dev libglfw3-dev patchelf
uv pip install setuptools==59.5.0 Cython==0.29.35 patchelf==0.17.2.0 wheel ninja
cd "$REPO/third_party/mujoco-py-2.1.2.14"
uv pip install -e .
```

## 7. Install the simulator environments

```bash
cd "$REPO/third_party"
cd gym-0.21.0       && uv pip install -e . && cd ..
cd Metaworld        && uv pip install -e . && cd ..
cd rrl-dependencies && uv pip install -e mj_envs/. && uv pip install -e mjrl/. && cd ..
```

Adroit expert policies (required for the Adroit tasks):
[OneDrive](https://1drv.ms/u/s!Ag5QsBIFtRnTlFWqYWtS2wMMPKNX?e=dw8hsS) -> unzip and put the
`ckpts` folder under `third_party/VRL3/`.

## 8. Install pytorch3d_simplified (non-editable, no build isolation)

```bash
cd "$REPO/third_party/pytorch3d_simplified"
uv pip install . --no-build-isolation
```

`setup.py` compiles a CUDA extension through `torch.utils.cpp_extension`; an isolated
build environment would not see torch, so `--no-build-isolation` is required.

## 9. Install the remaining Python packages

```bash
uv pip install zarr==2.12.0 wandb ipdb gpustat dm_control omegaconf \
    hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 diffusers==0.11.1 \
    numba==0.56.4 moviepy imageio av matplotlib termcolor \
    yacs==0.1.8 huggingface_hub==0.25.2 cloudpickle natsort open3d==0.19.0
```

## 10. Generate demonstrations

```bash
# Adroit (task: hammer | door | pen)
bash scripts/gen_demonstration_adroit.sh hammer

# MetaWorld (task: assembly | disassemble | stick-push )
bash scripts/gen_demonstration_metaworld.sh assembly
```

Arguments: `<task> [save_video=0] [num_episodes=10] [seed=42] [gpu=0] [output_dir]
[pc_resolution] [num_points]`; `output_dir` (6th) defaults to `3D-Diffusion-Policy/data`.
For Adroit, `pc_resolution` must stay 84 (the depth map feeds the 84x84 back-projection).


## 11. Quick run -- DP3 baseline

```bash
bash scripts/train_policy.sh dp3 adroit_hammer 0112 0 0 1000
bash scripts/eval_policy.sh dp3 adroit_hammer 0112 0 0 false 50
```

## 12. Quick run -- PI-JEPA (main method)

```bash
bash scripts/train_pijepa.sh adroit_hammer pjv5 0 0 1000
bash scripts/eval_pijepa.sh adroit_hammer pjv5 0 0
# out-of-distribution clutter suites: false | catgen | posgen | dynsphere | dynhand
bash scripts/eval_pijepa.sh adroit_hammer pjv5 0 0 false 50 catgen 0
```

## 13. Download a pretrained checkpoint

The trained PI-JEPA checkpoints are published on ModelScope as `a1339296390/PI-JEPA`.
Each run is one file, `<task>-pijepa-pjv5_seed0/checkpoints/best.ckpt` (~1.9 GB); the
released tasks are `adroit_door`, `adroit_hammer`, `adroit_pen`, `metaworld_assembly`,
`metaworld_disassemble` and `metaworld_stick-push`.

`modelscope-hub` requires Python >= 3.10, so call it through `uvx` instead of the project
`.venv` (3.8):

```bash
TASK=adroit_door
REMOTE="${TASK}-pijepa-pjv5_seed0/checkpoints/best.ckpt"
uvx --python 3.11 --from modelscope-hub ms-hub download a1339296390/PI-JEPA "$REMOTE" \
    --repo-type model --local-dir "$REPO/outputs" --continue-on-error
```

`--local-dir` keeps the remote layout, so the result is
`outputs/<task>-pijepa-pjv5_seed0/checkpoints/best.ckpt` -- the exact run dir the launchers
use. Evaluate it directly (only `best.ckpt` is published, so keep the default checkpoint):

```bash
bash scripts/eval_pijepa.sh adroit_door pjv5 0 0
```

Re-running the download resumes an interrupted file; the `run.json missing` warning from
`eval_pijepa.sh` is expected for a downloaded run dir.

## 14. Troubleshooting

- OpenGL errors -> `unset LD_PRELOAD`
- `distutils has no attribute 'version'` -> `uv pip install setuptools==59.5.0`
- `CUDA no kernel image` or `cannot import name '_C'` -> reinstall step 8
- Training cannot find the dataset -> generate the demos first (step 10), or pass
  `ZARR=<path>` / the extra zarr argument
- `No module named 'natsort'` -> `uv pip install natsort`
- A simulator fails to import -> `3D-Diffusion-Policy/diffusion_policy_3d/env/__init__.py`
  already guards each env with its own try/except, so one broken env does not block the others;
  see [ERROR_CATCH.md](ERROR_CATCH.md) for the rest
- `ninja` missing while building a CUDA extension -> install it (`uv pip install ninja` /
  `sudo apt-get install ninja-build`)

## 15. See also

- [INSTALL.md](INSTALL.md) -- legacy conda recipe from upstream DP3
- [ERROR_CATCH.md](ERROR_CATCH.md) -- in-depth installation / runtime errors
- [Adroit_Eval.md](Adroit_Eval.md) / [Metaworld_Eval.md](Metaworld_Eval.md) -- per-task
  train/eval commands and the 5-suite evaluation loops
- `scripts/train_pijepa.sh`, `scripts/eval_pijepa.sh` -- authoritative argument / env lists
