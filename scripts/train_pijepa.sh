#!/usr/bin/env bash
# PI-JEPA (v5, main method): two-view DP3 + a train-only frame-level temporal
# dynamics predictor on the F real frames that follow the observation window.
#   model: PI-JEPA/pijepa_policy.py | trainer: PI-JEPA/train_pijepa.py
# Deploy is the bottleneck branch, so the eval call MUST pass PIJEPA=1 PIJEPA_V5=1 and the same
# FUTURE_STEPS / TDT_LAYERS / TDT_HEADS / TDT_FF (TDT_POOL is train-only).
# Loss/data details: PI-JEPA/train_pijepa.py docstring + PI-JEPA/pijepa_common.py.
#
# Usage: bash scripts/train_pijepa.sh <task> <addition_info> <seed> <gpu> [epochs=1000] [zarr_path]
#        e.g. bash scripts/train_pijepa.sh metaworld_stick-push pjv5 0 0 1000
# Run dir = <OUT_ROOT>/<task>-pijepa-<addition_info>_seed<seed>.
#
# env: FUTURE_STEPS=8 (0 = cfg.n_action_steps) | GAMMA=0.5 | L1_MODE=sum|mean |
#      TDT_LAYERS=2 | TDT_HEADS=4 | TDT_FF=1024 | TDT_DROPOUT=0.0 | TDT_POOL=task|mean |
#      POOL_TYPE=softmax|mean|max | POOL_TEMP=1.0 | POOL_SCORE_HIDDEN=0 |
#      LAMBDA_JEPA_CLEAN=1|0 | LAMBDA_JEPA_POLL=1|0 | L1_WARMUP_EPOCHS=200 (1 = off) |
#      L2_CLEAN_WEIGHT=1.0 | L2_POLL_WEIGHT=0.3 | BN_HIDDEN=128,64,128 |
#      POLL_VOXEL_MIN=10 | POLL_VOXEL_MAX=50 | POLL_MODE=bbox|occupied | POLL_SEED=0 |
#      FPS_DEVICE=cpu|auto|cuda | BATCH=128 | LR= | WORKERS=8 | ROLLOUT_EVERY=200 (0 = off) |
#      MAX_STEPS=0 | SAVE_BEST=auto|1|0 | BEST_MONITOR= | BEST_K= | BEST_MODE= |
#      BEST_NAME=best.ckpt | EVAL_EPISODES= | OUT_ROOT=/hy-tmp/data/outputs |
#      PYTHON= | TRAIN_ENTRY=<trainer.py> (default PI-JEPA/train_pijepa.py) |
#      DRY_RUN=1 | RESUME=1 | NO_CKPT=1
set -euo pipefail

task_name=${1:?task_name}
addition_info=${2:?addition_info}
seed=${3:?seed}
gpu=${4:?gpu}
epochs=${5:-1000}
zarr_arg=${6:-}

pool_type=${POOL_TYPE:-softmax}
pool_temp=${POOL_TEMP:-1.0}
pool_hidden=${POOL_SCORE_HIDDEN:-0}
lambda_jepa_clean=${LAMBDA_JEPA_CLEAN:-1}
lambda_jepa_poll=${LAMBDA_JEPA_POLL:-1}
l1_warmup_epochs=${L1_WARMUP_EPOCHS:-200}
l2_clean_weight=${L2_CLEAN_WEIGHT:-1.0}
l2_poll_weight=${L2_POLL_WEIGHT:-0.3}
bn_hidden=${BN_HIDDEN:-128,64,128}
future_steps=${FUTURE_STEPS:-8}
gamma=${GAMMA:-0.5}
l1_mode=${L1_MODE:-sum}
tdt_layers=${TDT_LAYERS:-2}
tdt_heads=${TDT_HEADS:-4}
tdt_ff=${TDT_FF:-1024}
tdt_dropout=${TDT_DROPOUT:-0.0}
tdt_pool=${TDT_POOL:-task}
vmin=${POLL_VOXEL_MIN:-10}
vmax=${POLL_VOXEL_MAX:-50}
poll_mode=${POLL_MODE:-bbox}
poll_seed=${POLL_SEED:-0}
fps_device=${FPS_DEVICE:-cpu}
batch=${BATCH:-128}
lr=${LR:-}
workers=${WORKERS:-8}
rollout_every=${ROLLOUT_EVERY:-200}
max_steps=${MAX_STEPS:-0}
save_best=${SAVE_BEST:-auto}
best_monitor=${BEST_MONITOR:-}
best_k=${BEST_K:-}
best_mode=${BEST_MODE:-}
best_name=${BEST_NAME:-best.ckpt}
eval_episodes=${EVAL_EPISODES:-}

usage() {
    echo "  usage: bash scripts/train_pijepa.sh <task> <addition_info> <seed> <gpu> [epochs=1000] [zarr_path]"
    echo "  e.g.   bash scripts/train_pijepa.sh metaworld_stick-push 1024pj-v5 0 0"
}

# --- fail-fast guards ---------------------------------------------------------
num() { case "$2" in ''|*[!0-9,]*) echo "[pijepa] <$1> must be a number (got '$2')"; usage; exit 1 ;; esac; }
flt() { case "$2" in ''|*[!0-9.]*) echo "[pijepa] <$1> must be a float (got '$2')"; usage; exit 1 ;; esac; }
opt_int() {
    [ -n "$2" ] || return 0
    case "$2" in ''|*[!0-9]*) echo "[pijepa] <$1> must be an integer (got '$2')"; exit 1 ;; esac
}

num seed "${seed}"; num gpu "${gpu}"; num epochs "${epochs}"; num batch "${batch}"
num max_steps "${max_steps}"; num rollout_every "${rollout_every}"
num vmin "${vmin}"; num vmax "${vmax}"; num workers "${workers}"
num future_steps "${future_steps}"; num tdt_layers "${tdt_layers}"
num tdt_heads "${tdt_heads}"; num tdt_ff "${tdt_ff}"
num l1_warmup_epochs "${l1_warmup_epochs}"
flt l2_clean_weight "${l2_clean_weight}"; flt l2_poll_weight "${l2_poll_weight}"
flt pool_temp "${pool_temp}"; flt gamma "${gamma}"; flt tdt_dropout "${tdt_dropout}"
opt_int best_k "${best_k}"; opt_int eval_episodes "${eval_episodes}"

case "${lambda_jepa_clean}:${lambda_jepa_poll}" in
    0:0|0:1|1:0|1:1) ;;
    *) echo "[pijepa] LAMBDA_JEPA_CLEAN/LAMBDA_JEPA_POLL must be 1 (on) or 0 (off)"; exit 1 ;;
esac
case "${l1_mode}" in
    sum|mean) ;;
    *) echo "[pijepa] L1_MODE must be sum|mean"; exit 1 ;;
esac
case "${tdt_pool}" in
    task|mean) ;;
    *) echo "[pijepa] TDT_POOL must be task (softmax pool) or mean"; exit 1 ;;
esac
case "${bn_hidden}" in
    ''|*[!0-9,x]*) echo "[pijepa] BN_HIDDEN must look like 128,64,128"; exit 1 ;;
esac
case "${pool_type}" in
    softmax|mean|max) ;;
    *) echo "[pijepa] POOL_TYPE must be softmax|mean|max"; exit 1 ;;
esac
if [ "$((256 % tdt_heads))" -ne 0 ]; then
    echo "[pijepa] TDT_HEADS must divide d_model=256 (got ${tdt_heads})"; exit 1
fi
if [ "${tdt_layers}" -lt 1 ] || [ "${tdt_ff}" -lt 1 ]; then
    echo "[pijepa] TDT_LAYERS and TDT_FF must be >= 1"; exit 1
fi
case "${fps_device}" in auto|cpu|cuda) ;; *) echo "[pijepa] FPS_DEVICE must be auto|cpu|cuda"; exit 1 ;; esac
case "${poll_mode}" in bbox|occupied) ;; *) echo "[pijepa] POLL_MODE must be bbox|occupied"; exit 1 ;; esac
case "${save_best}" in auto|0|1) ;; *) echo "[pijepa] SAVE_BEST must be auto|0|1"; exit 1 ;; esac
case "${best_mode}" in ''|max|min) ;; *) echo "[pijepa] BEST_MODE must be max|min"; exit 1 ;; esac
if command -v nvidia-smi >/dev/null 2>&1; then
    _n_gpu=$(nvidia-smi -L 2>/dev/null | wc -l)
    for _g in $(echo "${gpu}" | tr ',' ' '); do
        if [ "${_g}" -ge "${_n_gpu}" ] 2>/dev/null; then
            echo "[pijepa] gpu index ${_g} >= visible GPU count ${_n_gpu}"; usage; exit 1
        fi
    done
    unset _g
fi


# --- paths: one name per experiment -------------------------------------------
# run dir = <OUT_ROOT>/<task>-pijepa-<addition_info>_seed<seed>; <addition_info> is the
# experiment tag, nothing else is appended. scripts/eval_pijepa.sh resolves
# the same dir (scripts/eval_policy.sh would need RUN_DIR=).
repo="$(cd "$(dirname "$0")/.." && pwd)"
py=${PYTHON:-${repo}/.venv/bin/python}
out_root=${OUT_ROOT:-/hy-tmp/data/outputs}
source "${repo}/PI-JEPA/run_dir.sh"
alg_tag=pijepa
exp_name=${task_name}-${alg_tag}-${addition_info}
run_dir=$(ablation_run_dir "${out_root}" "${task_name}" "${alg_tag}" "${addition_info}" "${seed}")
task_cfg="${repo}/3D-Diffusion-Policy/diffusion_policy_3d/config/task/${task_name}.yaml"
[ -f "${task_cfg}" ] || { echo "[pijepa] unknown task '${task_name}': ${task_cfg} not found"; exit 1; }
zarr_path=${zarr_arg:-${ZARR:-/hy-tmp/3DP/data/${task_name}_expert.zarr}}
[ -d "${zarr_path}" ] || { echo "[pijepa] zarr not found: ${zarr_path} (pass as arg6 or set ZARR=)"; exit 1; }

ablation_claim_run_dir "${run_dir}" "$(printf '{"script":"scripts/train_pijepa.sh","task":"%s","addition_info":"%s","seed":%s,"epochs":%s,"lambda_jepa_clean":%s,"lambda_jepa_poll":%s,"l2_clean_weight":%s,"l2_poll_weight":%s,"pool_type":"%s","bn_hidden":"%s","future_steps":%s,"gamma":%s,"l1_mode":"%s","tdt_layers":%s,"tdt_heads":%s,"tdt_ff":%s,"tdt_dropout":%s,"tdt_pool":"%s","zarr":"%s"}' \
    "${task_name}" "${addition_info}" "${seed}" "${epochs}" "${lambda_jepa_clean}" "${lambda_jepa_poll}" \
    "${l2_clean_weight}" "${l2_poll_weight}" "${pool_type}" "${bn_hidden}" \
    "${future_steps}" "${gamma}" "${l1_mode}" "${tdt_layers}" "${tdt_heads}" "${tdt_ff}" \
    "${tdt_dropout}" "${tdt_pool}" "${zarr_path}")" || exit 1

# point count of the stored clean cloud (the polluted view must match it)
n_points=$("${py}" -c "import zarr; print(int(zarr.open('${zarr_path}', mode='r')['data']['point_cloud'].shape[1]))")

export CUDA_VISIBLE_DEVICES=${gpu}
export HYDRA_FULL_ERROR=1

train_entry=${TRAIN_ENTRY:-${repo}/PI-JEPA/train_pijepa.py}
[ -f "${train_entry}" ] || { echo "[pijepa] trainer not found: ${train_entry} (TRAIN_ENTRY=)"; exit 1; }

cmd=( "${py}" "${train_entry}"
      --task "${task_name}" --zarr "${zarr_path}"
      --out "${run_dir}" --exp-name "${exp_name}"
      --seed "${seed}" --epochs "${epochs}" --batch "${batch}"
      --num-workers "${workers}" --device "cuda:0"
      --pool-type "${pool_type}" --pool-temp "${pool_temp}"
      --pool-score-hidden "${pool_hidden}"
      --lambda-jepa-clean "${lambda_jepa_clean}" \
      --lambda-jepa-poll "${lambda_jepa_poll}"
      --l2-clean-weight "${l2_clean_weight}" --l2-poll-weight "${l2_poll_weight}"
      --bn-hidden "${bn_hidden}"
      --future-steps "${future_steps}" --gamma "${gamma}" --l1-mode "${l1_mode}"
      --l1-warmup-epochs "${l1_warmup_epochs}"
      --tdt-layers "${tdt_layers}" --tdt-heads "${tdt_heads}"
      --tdt-ff "${tdt_ff}" --tdt-dropout "${tdt_dropout}"
      --tdt-pool "${tdt_pool}"
      --poll-voxel-min "${vmin}" --poll-voxel-max "${vmax}"
      --poll-mode "${poll_mode}" --poll-seed "${poll_seed}"
      --fps-device "${fps_device}" --rollout-every "${rollout_every}"
      --save-best "${save_best}" --best-name "${best_name}" )
if [ -n "${best_monitor}" ]; then cmd+=(--best-monitor "${best_monitor}"); fi
if [ -n "${best_k}" ]; then cmd+=(--best-k "${best_k}"); fi
if [ -n "${best_mode}" ]; then cmd+=(--best-mode "${best_mode}"); fi
if [ -n "${eval_episodes}" ]; then
    cmd+=(--runtime-override "task.env_runner.eval_episodes=${eval_episodes}")
fi
if [ -n "${lr}" ]; then cmd+=(--lr "${lr}"); fi
if [ "${max_steps}" != "0" ]; then cmd+=(--max-steps "${max_steps}"); fi
if [ "${RESUME:-0}" = "1" ]; then cmd+=(--resume); fi
if [ "${NO_CKPT:-0}" = "1" ]; then cmd+=(--no-ckpt); fi

echo "[pijepa] task=${task_name} seed=${seed} epochs=${epochs} batch=${batch} N=${n_points}"
echo "[pijepa] loss = ${l2_clean_weight}*L2_clean + ${l2_poll_weight}*L2_poll + ${lambda_jepa_clean}*L1_clean_seq + ${lambda_jepa_poll}*L1_poll_seq"
echo "[pijepa] L1 warmstart: $( [ "${l1_warmup_epochs}" -gt 1 ] && echo "linear 0->1 over the first ${l1_warmup_epochs} epoch(s)" || echo "off" )"
echo "[pijepa] seq = To+F frames (F=${future_steps}, 0=cfg.n_action_steps), gamma=${gamma}, l1_mode=${l1_mode}"
echo "[pijepa] tdt: d_model=256 layers=${tdt_layers} heads=${tdt_heads} ff=${tdt_ff} dropout=${tdt_dropout} input_pool=${tdt_pool}"
echo "[pijepa] bottleneck 320->$(echo "${bn_hidden}" | tr ',' '->')->256 | deploy: mlp->bn->softmax pool->proj|state->UNet (no TDT)"
echo "[pijepa] pool=${pool_type} T=${pool_temp} h=${pool_hidden} | pollution: voxel ${vmin}..${vmax} mode=${poll_mode} seed=${poll_seed} fps=${fps_device}"
echo "[pijepa] zarr=${zarr_path}"
echo "[pijepa] out(write)=${run_dir}"
echo "[pijepa] in-loop eval: every=${rollout_every} epoch(s) (0=off) | best ckpt: SAVE_BEST=${save_best} monitor=${best_monitor:-cfg(test_mean_score)} k=${best_k:-cfg(1)} mode=${best_mode:-cfg(max)} -> checkpoints/${best_name} | eval_episodes=${eval_episodes:-cfg}"
echo "[pijepa] eval hint: bash scripts/eval_pijepa.sh ${task_name} ${addition_info} ${seed} ${gpu}   (CKPT=best for checkpoints/best.ckpt)"
echo "[pijepa]        generic: RUN_DIR=${run_dir} PIJEPA=1 PIJEPA_V5=1 FUTURE_STEPS=${future_steps} TDT_LAYERS=${tdt_layers} TDT_HEADS=${tdt_heads} TDT_FF=${tdt_ff} bash scripts/eval_policy.sh dp3 ${task_name} ${addition_info} ${seed} ${gpu}"
if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "+ ${cmd[*]}"
else
    "${cmd[@]}"
fi

