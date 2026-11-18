#!/usr/bin/env bash
# Metaworld expert demos (single collector: point_cloud + raw full point cloud).
#
# bash scripts/gen_demonstration_metaworld.sh assembly
# bash scripts/gen_demonstration_metaworld.sh assembly 0 10 42 0 /hy-tmp/3DP/data 128 1024
#   args: task, save_video (0/1 -> one mp4 per kept episode under <output_dir>/video),
#         num_episodes (default 10), seed (default 42), gpu_id (default 0),
#         output_dir (default /hy-tmp/3DP/data), pc_resolution (default 128),
#         num_points (default 1024)
# output: <output_dir>/metaworld_<task>_expert.zarr with data/
#   img, state, full_state, point_cloud, depth, action,
#   raw_point_cloud (object VLenArray float32, pre-FPS), raw_point_cloud_len
#   -> one file for both DP3 baseline and JEPA (raw_point_cloud) consumers.
# env: SEED_RULE=per_attempt|per_episode (default per_attempt)
#      MAX_ATTEMPTS=N (0 = 20 * num_episodes)
#      OVERWRITE=1 (replace an existing zarr) | DRY_RUN=1 (print only)
#      PYTHON=<interpreter path>
set -euo pipefail

task=${1:?task}
save_video=${2:-0}
num_episodes=${3:-10}
seed=${4:-42}
gpu_id=${5:-0}
output_dir=${6:-/hy-tmp/3DP/data}
pc_resolution=${7:-128}
num_points=${8:-1024}
seed_rule=${SEED_RULE:-per_attempt}

usage() {
    echo "  usage: bash scripts/gen_demonstration_metaworld.sh <task> [save_video=0] [num_episodes=10] [seed=42] [gpu=0] [output_dir=/hy-tmp/3DP/data] [pc_resolution=128] [num_points=1024]"
    echo "  e.g.   bash scripts/gen_demonstration_metaworld.sh assembly 0 10 42 0 /hy-tmp/3DP/data 128 1024"
}

for _nv in "num_episodes:${num_episodes}" "seed:${seed}" "gpu_id:${gpu_id}" \
           "pc_resolution:${pc_resolution}" "num_points:${num_points}"; do
    _n=${_nv%%:*}; _v=${_nv#*:}
    case "${_v}" in
        ''|*[!0-9]*) echo "[mw] <${_n}> must be an integer (got '${_v}')"; usage; exit 1 ;;
    esac
done
if [ "${pc_resolution}" -lt 32 ]; then
    echo "[mw] <pc_resolution> must be >= 32 (render/point-cloud resolution, got '${pc_resolution}')"
    usage; exit 1
fi
case "${seed_rule}" in
    per_attempt|per_episode) ;;
    *) echo "[mw] SEED_RULE must be per_attempt or per_episode (got '${seed_rule}')"; usage; exit 1 ;;
esac
case "${save_video}" in
    0|1|true|false) ;;
    *) echo "[mw] <save_video> must be 0/1 (got '${save_video}')"; usage; exit 1 ;;
esac
video_flag=0
if [ "${save_video}" = "1" ] || [ "${save_video}" = "true" ]; then
    video_flag=1
fi
if command -v nvidia-smi >/dev/null 2>&1; then
    _n_gpu=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "${gpu_id}" -ge "${_n_gpu}" ] 2>/dev/null; then
        echo "[mw] gpu_id ${gpu_id} >= visible GPU count ${_n_gpu}"; usage; exit 1
    fi
fi

zarr_path="${output_dir}/metaworld_${task}_expert.zarr"
extra=()
[ "${OVERWRITE:-0}" = "1" ] && extra+=( --overwrite )
[ -n "${MAX_ATTEMPTS:-}" ] && extra+=( --max-attempts "${MAX_ATTEMPTS}" )
[ "${video_flag}" = "1" ] && extra+=( --save_video )
repo="$(cd "$(dirname "$0")/.." && pwd)"
py=${PYTHON:-}
if [ -z "${py}" ] && [ -x "${repo}/.venv/bin/python" ]; then py="${repo}/.venv/bin/python"; fi
py=${py:-python}

echo "[mw] task=${task} episodes=${num_episodes} seed=${seed} rule=${seed_rule} gpu=${gpu_id}"
echo "[mw] zarr=${zarr_path}"
if [ -e "${zarr_path}" ] && [ "${OVERWRITE:-0}" != "1" ]; then
    echo "[mw] note: ${zarr_path} exists -> collector refuses to overwrite (set OVERWRITE=1 to replace)"
fi

cd "${repo}/third_party/Metaworld"
cmd=(env "CUDA_VISIBLE_DEVICES=${gpu_id}" "${py}" gen_demonstration_expert.py \
     --env_name "${task}" \
     --num_episodes "${num_episodes}" \
     --root_dir "${output_dir}" \
     --seed "${seed}" \
     --seed-rule "${seed_rule}" \
     --pc_resolution "${pc_resolution}" \
     --num_points "${num_points}" \
     "${extra[@]}" )
if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "+ ${cmd[*]}"
else
    "${cmd[@]}"
fi
cd - >/dev/null
