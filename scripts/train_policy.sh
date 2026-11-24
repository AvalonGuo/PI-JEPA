# Examples:
# bash scripts/train_policy.sh dp3 adroit_hammer 0322 0 0
# bash scripts/train_policy.sh dp3 dexart_laptop 0322 0 0
# bash scripts/train_policy.sh simple_dp3 adroit_hammer 0322 0 0
# bash scripts/train_policy.sh dp3 metaworld_basketball 0602 0 0



DEBUG=False
save_ckpt=True

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed=${4}
repo="$(cd "$(dirname "$0")/.." && pwd)"
exp_name=${task_name}-${alg_name}-${addition_info}
# run dir stays inside the repository: <repo>/outputs/<task>-<alg>-<info>_seed<seed>
# (RUN_DIR= overrides it, e.g. to evaluate a run that still lives elsewhere)
run_dir="${RUN_DIR:-${repo}/outputs/${exp_name}_seed${seed}}"


# gpu_id=$(bash scripts/find_gpu.sh)
gpu_id=${5}
# optional 6th argument: number of training epochs (default 3000, same as dp3.yaml)
epochs=${6:-3000}
# optional 7th argument: dataset zarr path override (absolute path)
zarr_path=${7:-}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


wandb_mode=${WANDB_MODE:-offline}

if [ $DEBUG = True ]; then
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
fi

cd "${repo}/3D-Diffusion-Policy"

# optional dataset zarr override, appended only when provided
if [ -n "$zarr_path" ]; then
    ZARR_OVERRIDE="task.dataset.zarr_path=${zarr_path}"
else
    ZARR_OVERRIDE=""
fi

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}
python train.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.num_epochs=${epochs} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            ${ZARR_OVERRIDE}



                                