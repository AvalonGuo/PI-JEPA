#!/usr/bin/env bash
# Adroit expert demos (single collector: point_cloud + raw full point cloud).
# Mirrors scripts/gen_demonstration_metaworld.sh argument-for-argument.
#
# bash scripts/gen_demonstration_adroit.sh hammer
# bash scripts/gen_demonstration_adroit.sh hammer 0 10 42 0 /hy-tmp/3DP/data 84 0
#   args: task, save_video (0/1 -> one mp4 per kept episode under <output_dir>/video),
#         num_episodes (default 10), seed (default 42), gpu_id (default 0),
#         output_dir (default /hy-tmp/3DP/data), pc_resolution (default 84),
#         extra_steps_after_success (default 0 = full episode, run to env horizon),
#         success_goal_threshold (default 10), min_episode_steps (default 100)
# output: <output_dir>/adroit_<task>_expert.zarr with data/
#   img (84x84), state, action, point_cloud (512), depth (84x84),
#   raw_point_cloud (object VLenArray float32, pre-FPS), raw_point_cloud_len
#   -> one file for DP3 training + JEPA (raw_point_cloud) + dr3d green cells (depth).
# note: pc_resolution must stay 84 for adroit (depth feeds the 84x84 green-cell
#       back-projection; the 512-point observation comes from the same wrapper).
# env: OVERWRITE=1 (replace an existing zarr) | APPEND=1 (append episodes)
#      DRY_RUN=1 (print only) | PYTHON=<interpreter path>
#      EXTRA_STEPS_AFTER_SUCCESS= | SUCCESS_GOAL_THRESHOLD= | MIN_EPISODE_STEPS=
#      ALLOW_MIXED_TRUNCATION=1 (append onto a zarr with another truncation policy)
set -euo pipefail

task=${1:?task}
save_video=${2:-0}
num_episodes=${3:-10}
seed=${4:-42}
gpu_id=${5:-0}
output_dir=${6:-/hy-tmp/3DP/data}
pc_resolution=${7:-84}
extra_steps=${8:-${EXTRA_STEPS_AFTER_SUCCESS:-0}}
success_threshold=${9:-${SUCCESS_GOAL_THRESHOLD:-10}}
min_episode_steps=${10:-${MIN_EPISODE_STEPS:-100}}

usage() {
    echo "  usage: bash scripts/gen_demonstration_adroit.sh <task> [save_video=0] [num_episodes=10] [seed=42] [gpu=0] [output_dir=/hy-tmp/3DP/data] [pc_resolution=84] [extra_steps_after_success=0] [success_goal_threshold=10] [min_episode_steps=100]"
    echo "  e.g.   bash scripts/gen_demonstration_adroit.sh hammer 0 10 42 0 /hy-tmp/3DP/data 84 0"
}

for _nv in "num_episodes:${num_episodes}" "seed:${seed}" "gpu_id:${gpu_id}" \
           "pc_resolution:${pc_resolution}" "extra_steps:${extra_steps}" \
           "min_episode_steps:${min_episode_steps}"; do
    _n=${_nv%%:*}; _v=${_nv#*:}
    case "${_v}" in
        ''|*[!0-9]*) echo "[adroit] <${_n}> must be an integer (got '${_v}')"; usage; exit 1 ;;
    esac
done
case "${save_video}" in
    0|1|true|false) ;;
    *) echo "[adroit] <save_video> must be 0/1 (got '${save_video}')"
       echo "         note: arg2 used to be <seed>; the new order matches gen_demonstration_metaworld.sh"
       usage; exit 1 ;;
esac
case "${pc_resolution}" in
    84|128|256) ;;
    *) echo "[adroit] <pc_resolution> must be one of 84/128/256 (got '${pc_resolution}')"
       usage; exit 1 ;;
esac
if [ "${pc_resolution}" != "84" ]; then
    echo "[adroit] WARN: pc_resolution=${pc_resolution} -> data/depth is ${pc_resolution}x${pc_resolution};"
    echo "         green-cell back-projection assumes 84x84 (keep 84 unless intended)"
fi
case "${success_threshold}" in
    ''|*[!0-9.]*) echo "[adroit] <success_goal_threshold> must be a number (got '${success_threshold}')"
       usage; exit 1 ;;
esac
video_flag=0
if [ "${save_video}" = "1" ] || [ "${save_video}" = "true" ]; then
    video_flag=1
fi
if command -v nvidia-smi >/dev/null 2>&1; then
    _n_gpu=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "${gpu_id}" -ge "${_n_gpu}" ] 2>/dev/null; then
        echo "[adroit] gpu_id ${gpu_id} >= visible GPU count ${_n_gpu}"; usage; exit 1
    fi
fi

zarr_path="${output_dir}/adroit_${task}_expert.zarr"
if [ "${extra_steps}" -gt 0 ]; then
    policy_desc="success+${extra_steps} steps (truncated)"
else
    policy_desc="full (no truncation, run to env horizon)"
fi
extra_py=()
[ "${OVERWRITE:-0}" = "1" ] && extra_py+=( --overwrite )
[ "${APPEND:-0}" = "1" ] && extra_py+=( --append )
[ "${ALLOW_MIXED_TRUNCATION:-0}" = "1" ] && extra_py+=( --allow_mixed_truncation )
[ "${video_flag}" = "1" ] && extra_py+=( --save_video )

repo="$(cd "$(dirname "$0")/.." && pwd)"
py=${PYTHON:-}
if [ -z "${py}" ] && [ -x "${repo}/.venv/bin/python" ]; then py="${repo}/.venv/bin/python"; fi
py=${py:-python}

echo "[adroit] task=${task} episodes=${num_episodes} seed=${seed} gpu=${gpu_id}"
echo "[adroit] extra_steps_after_success=${extra_steps} -> episode_len_policy=${policy_desc}"
echo "[adroit] zarr=${zarr_path}"
if [ -e "${zarr_path}" ] && [ "${OVERWRITE:-0}" != "1" ] && [ "${APPEND:-0}" != "1" ]; then
    echo "[adroit] note: ${zarr_path} exists -> collector refuses to overwrite (OVERWRITE=1 to replace, APPEND=1 to extend)"
fi

cd "${repo}/third_party/VRL3/src"
cmd=(env "CUDA_VISIBLE_DEVICES=${gpu_id}" "${py}" gen_demonstration_expert.py \
     --env_name "${task}" \
     --num_episodes "${num_episodes}" \
     --root_dir "${output_dir}" \
     --output_dir "${output_dir}" \
     --expert_ckpt_path "../ckpts/vrl3_${task}.pt" \
     --img_size 84 \
     --pc_resolution "${pc_resolution}" \
     --not_use_multi_view \
     --use_point_crop \
     --seed "${seed}" \
     --success_goal_threshold "${success_threshold}" \
     --extra_steps_after_success "${extra_steps}" \
     --min_episode_steps "${min_episode_steps}" \
     "${extra_py[@]}" )
if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "+ ${cmd[*]}"
else
    "${cmd[@]}"
fi
cd - >/dev/null
