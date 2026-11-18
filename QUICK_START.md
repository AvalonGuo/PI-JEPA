# 快速开始 RTX4060
## 0. 安装 uv
```bash
pip install uv
echo 'export UV_CACHE_DIR=/hy-tmp/uv-cache' >> ~/.bashrc
source ~/.bashrc
```

## 1. 创建并激活 Python 3.8 虚拟环境
```bash
uv venv --python 3.8 .venv
source .venv/bin/activate
```

## 2. 安装 PyTorch
```bash
uv pip install torch torchvision torchaudio
```

## 3. 安装 dp3
```bash
cd 3D-Diffusion-Policy && uv pip install -e . && cd ..
```

## 4. 安装 MuJoCo 并写环境变量
```bash
mkdir -p ~/.mujoco && cd ~/.mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco210.tar.gz --no-check-certificate
tar -xvzf mujoco210.tar.gz && rm -f mujoco210.tar.gz
```
把以下内容追加到 `~/.bashrc`，执行 `source ~/.bashrc` 后开新终端：
```bash
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco210/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64' >> ~/.bashrc
echo 'export MUJOCO_GL=egl' >> ~/.bashrc
```

## 5. 安装编译依赖 + mujoco-py
```bash
sudo apt-get install -y libglew-dev libgl1-mesa-dev libglu1-mesa-dev libosmesa6-dev libglfw3-dev patchelf
```
再装 Python 编译依赖并安装 mujoco-py：
```bash
cd ../3DP
uv pip install setuptools==59.5.0 Cython==0.29.35 patchelf==0.17.2.0 wheel ninja
cd third_party/mujoco-py-2.1.2.14 && uv pip install -e . 
```

## 6. 安装仿真环境（先修正 gym 的 opencv 版本写法）
```bash
cd ..
cd dexart-release && uv pip install -e . && cd ..
cd gym-0.21.0 && uv pip install -e . && cd ..
cd Metaworld && uv pip install -e . && cd ..
cd rrl-dependencies && uv pip install -e mj_envs/. && uv pip install -e mjrl/. && cd ..
```

下载并放置数据资产（可选，复现 adroit 任务需要）：
- DexArt assets（Google Drive）→ 解压到 `third_party/dexart-release/assets`
- Adroit RL experts ckpts（OneDrive）→ 解压后把 `ckpts` 放到 `third_party/VRL3/`

## 7. 安装 pytorch3d（简化版，必须非 editable + --no-build-isolation）
```bash
cd ./pytorch3d_simplified && uv pip install . --no-build-isolation && cd ..
```

## 8. 安装其余依赖
```bash
uv pip install zarr==2.12.0 wandb ipdb gpustat dm_control omegaconf hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 diffusers==0.11.1 numba==0.56.4 moviepy imageio av matplotlib termcolor yacs==0.1.8 huggingface_hub==0.25.2 cloudpickle natsort
```

## 9. 快速复现（生成演示 → 训练）
```bash
source .venv/bin/activate

wandb login   # 查看训练视频需要，可跳过

bash scripts/gen_demonstration_adroit.sh hammer      # 生成演示数据
bash scripts/train_policy.sh dp3 adroit_hammer 0112 0 0   # 训练 DP3
bash scripts/eval_policy.sh dp3 adroit_hammer 0112 0 0    # 评估 DP3（加载最新 checkpoint 并在仿真中 rollout）
```
> 脚本内部用裸 `python`，所以务必确保 `source .venv/bin/activate` 已生效。
> 训练与评估的输出均保存在数据盘 `/hy-tmp/data/outputs/${任务}-${算法}-${备忘}_seed${seed}/`，权重在其中的 `checkpoints/` 目录（`latest.ckpt` 为最新权重）。
> 训练可用第 6 个可选参数指定 epoch 数，例如 `bash scripts/train_policy.sh dp3 adroit_hammer 0112 0 0 200` 表示训练 200 个 epoch（默认 3000, seed=0, cuda_id=0）。
> 评估脚本会从上述 `checkpoints/latest.ckpt` 加载权重，需与训练命令的 3~5 个参数保持一致（`0112 0 0`）。

## 常见报错速查
- OpenGL 报错 → `unset LD_PRELOAD`
- `distutils has no attribute 'version'` → `uv pip install setuptools==59.5.0`
- CUDA no kernel image / `cannot import name '_C'` → 重跑第 7 步命令
- 训练数据路径报错 → 确认 ckpts 已放到 `third_party/VRL3/ckpts/`
- `import diffusion_policy_3d` / `import dexart` 失败 → 检查并补建顶层 `__init__.py`，重装对应包（见第 3、6 步注意）
- `No module named 'natsort'` → `uv pip install natsort`
- 训练报 `... has no attribute 'xxx_runner'` 且是 dexart/metaworld 导入失败导致 → 将 `3D-Diffusion-Policy/diffusion_policy_3d/env/__init__.py` 改成 try/except 独立导入，缺失环境不影响其它环境

## 10. 安装其它依赖
```
cd /root/3DP
uv pip install --python .venv/bin/python timm
uv pip install --python .venv/bin/python spconv-cu121        # cu121 版，配 torch 2.4.1
uv pip install --python .venv/bin/python torch-scatter       # 若 pip 无 py3.8/torch2.4 预编译wheel需源码编译

```