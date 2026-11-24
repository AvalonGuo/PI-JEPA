#!/usr/bin/env bash
# PI-JEPA v5 eval launcher -- counterpart of scripts/train_pijepa.sh.
# The v5 hyper-parameters are read back from the ckpt (cfg.pijepa_v5.*, encoder pool_*):
# eval.py rebuilds the policy from the CLI cfg, where a missing env var is silently
# replaced by a dp3 default, i.e. the numbers would be meaningless. mmap=True.
#
# Usage: bash scripts/eval_pijepa.sh <task> <addition_info> <seed> <gpu> \
#            [save_video=false] [episodes=50] [suite=false] [extra_steps=0] [run_tag] [ckpt]
#   <addition_info> run dir tag <task>-pijepa-<addition_info>_seed<seed>
#   <suite>         false | catgen | posgen | dynsphere | dynhand
#   <extra_steps>   0 = never truncate (Metaworld_Eval.md) | 15 = trainer's rollout
#   [ckpt]          latest | best | epoch=NNNN-*.ckpt | /abs/path.ckpt
# env: CKPT= (wins over [ckpt]) | OUT_ROOT= | PYTHON= | DRY_RUN=1 | TEE=1 | SUMMARY=1 |
#      LOG= | NUM_POINTS= | ZARR= | BN_HIDDEN= | FUTURE_STEPS= | TDT_* | POOL_* (ckpt wins) |
#      V5_TARGET=<dotted class, default pijepa_policy.PIJEPAPolicyV5>
#      subclass; PYTHONPATH must make it importable> |
#      METRICS_DIR=<media dir>|none | METRIC_TAG=pijepa_v5
# e.g. bash scripts/eval_pijepa.sh metaworld_stick-push pjv5 0 0
#      CKPT=best bash scripts/eval_pijepa.sh metaworld_stick-push pjv5 0 0 false 50 catgen 0
set -euo pipefail

task_name=${1:-}
addition_info=${2:-}
seed=${3:-}
gpu=${4:-}
save_video=${5:-false}
episodes=${6:-50}
suite=${7:-false}
extra_steps=${8:-0}
run_tag=${9:-}
ckpt_sel=${10:-${CKPT:-latest}}

v5_target=${V5_TARGET:-pijepa_policy.PIJEPAPolicyV5}

usage() {
    cat <<'EOF'
  usage: bash scripts/eval_pijepa.sh <task> <addition_info> <seed> <gpu> \
              [save_video=false] [episodes=50] [suite=false] [extra_steps=0] [run_tag] [ckpt]
  e.g.   bash scripts/eval_pijepa.sh metaworld_stick-push pjv5 0 0
         CKPT=best bash scripts/eval_pijepa.sh metaworld_stick-push pjv5 0 0
  ENV: CKPT=latest|best|epoch=NNNN-*.ckpt|/abs/path | DRY_RUN=1 | OUT_ROOT= | TEE=0 | SUMMARY=0 | LOG= | NUM_POINTS= | ZARR=
EOF
}

case "${task_name}" in
    -h|--help|help) usage; exit 0 ;;
esac
if [ -z "${task_name}" ] || [ -z "${addition_info}" ] || [ -z "${seed}" ] || [ -z "${gpu}" ]; then
    usage
    exit 1
fi

# --- fail-fast guards ---------------------------------------------------------
for _nm_val in "seed:${seed}" "gpu:${gpu}" "episodes:${episodes}" "extra_steps:${extra_steps}"; do
    _nm=${_nm_val%%:*}; _val=${_nm_val#*:}
    case "${_val}" in
        ''|*[!0-9]*) echo "[eval5] <${_nm}> must be a non-negative integer (got '${_val}')"; usage; exit 1 ;;
    esac
done
unset _nm_val _nm _val

case "${save_video}" in
    true|false) ;;
    *) echo "[eval5] <save_video> must be true|false (got '${save_video}')"; exit 1 ;;
esac

case "${suite}" in
    false)                           distractor_suite="" ;;
    catgen|posgen|dynsphere|dynhand) distractor_suite="${suite}" ;;
    *) echo "[eval5] <suite> must be false|catgen|posgen|dynsphere|dynhand (got '${suite}')"; exit 1 ;;
esac

repo="$(cd "$(dirname "$0")/.." && pwd)"
py=${PYTHON:-${repo}/.venv/bin/python}
out_root=${OUT_ROOT:-${repo}/outputs}

task_cfg="${repo}/3D-Diffusion-Policy/diffusion_policy_3d/config/task/${task_name}.yaml"
[ -f "${task_cfg}" ] || { echo "[eval5] unknown task '${task_name}': ${task_cfg} not found"; exit 1; }
[ -x "${py}" ] || { echo "[eval5] python not executable: ${py} (set PYTHON=)"; exit 1; }

if command -v nvidia-smi >/dev/null 2>&1; then
    _n_gpu=$(nvidia-smi -L 2>/dev/null | wc -l)
    for _g in $(echo "${gpu}" | tr ',' ' '); do
        if [ "${_g}" -ge "${_n_gpu}" ] 2>/dev/null; then
            echo "[eval5] gpu index ${_g} >= visible GPU count ${_n_gpu}"; exit 1
        fi
    done
    unset _g
fi

# --- run dir / checkpoint -----------------------------------------------------
source "${repo}/PI-JEPA/run_dir.sh"
alg_tag=pijepa
run_dir=$(pijepa_run_dir "${out_root}" "${task_name}" "${alg_tag}" "${addition_info}" "${seed}")
if [ ! -d "${run_dir}" ]; then
    echo "[eval5] run dir not found: ${run_dir}"
    echo "[eval5] hint: <addition_info> names the dir (<task>-${alg_tag}-<addition_info>_seed<seed>);"
    echo "[eval5]       OUT_ROOT=${out_root} -- ls it to see what exists"
    exit 1
fi
[ -f "${run_dir}/run.json" ] || \
    echo "[eval5] WARN: ${run_dir}/run.json missing (not a launcher-created run dir?)"

ckpt_dir="${run_dir}/checkpoints"
case "${ckpt_sel}" in
    latest) ckpt_path="${ckpt_dir}/latest.ckpt" ;;
    best)   ckpt_path="${ckpt_dir}/best.ckpt" ;;
    /*)     ckpt_path="${ckpt_sel}" ;;
    *)      ckpt_path="${ckpt_dir}/${ckpt_sel}" ;;
esac
if [ -L "${ckpt_path}" ] && [ ! -e "${ckpt_path}" ]; then
    echo "[eval5] ${ckpt_path} is a dangling symlink -> $(readlink "${ckpt_path}" 2>/dev/null)"
    echo "[eval5] the rotating topk manager removed its target; use CKPT=latest"
    [ "${DRY_RUN:-0}" = "1" ] || exit 1
fi
ckpt_real=""
ckpt_tag="${ckpt_sel}"
if [ -f "${ckpt_path}" ]; then
    ckpt_real=$(realpath -e "${ckpt_path}" 2>/dev/null || true)
    ckpt_tag=$(basename "${ckpt_real:-${ckpt_path}}" .ckpt | tr '=' '-')
else
    echo "[eval5] checkpoint not found: ${ckpt_path}"
    echo "[eval5] hint: ls ${ckpt_dir}   (a run trained with SAVE_BEST=0 has no best.ckpt)"
    [ "${DRY_RUN:-0}" = "1" ] || exit 1
fi

# --- V5 hyper-parameters, read back from the checkpoint's own cfg --------------
declare -A META=()
_resolved=0
if [ -n "${ckpt_real}" ] && [ -f "${ckpt_real}" ]; then
    if _meta=$("${py}" - "${ckpt_real}" "${task_cfg}" <<'PY'
import sys

import dill
import torch

ckpt, task_yaml = sys.argv[1], sys.argv[2]


def load(path):
    try:
        return torch.load(path, pickle_module=dill, map_location="cpu",
                          weights_only=False, mmap=True)
    except Exception:                       # old torch / fs without mmap support
        with open(path, "rb") as f:
            return torch.load(f, pickle_module=dill, map_location="cpu",
                              weights_only=False)


def get(node, *keys, default=None):
    cur = node
    for key in keys:
        if cur is None:
            return default
        try:
            cur = cur[key]
        except Exception:                   # noqa: BLE001
            return default
    return default if cur is None else cur


payload = load(ckpt)
cfg = payload["cfg"]
v5 = get(cfg, "pijepa_v5", default={}) or {}
enc = get(cfg, "policy", "pointcloud_encoder_cfg", default={}) or {}
tdt = v5.get("tdt_cfg") or {}
bn = (v5.get("bottleneck_cfg") or {}).get("hidden") or [128, 64, 128]
try:
    epoch = int(dill.loads(payload["pickles"]["epoch"]))
except Exception:                           # noqa: BLE001
    epoch = -1

vals = {
    "epoch": epoch,
    "bn_hidden": ",".join(str(int(x)) for x in bn),
    "future_steps": tdt.get("future_steps", v5.get("future_steps", 8)),
    "tdt_layers": tdt.get("layers", 2),
    "tdt_heads": tdt.get("heads", 4),
    "tdt_ff": tdt.get("dim_ff", 1024),
    "tdt_dropout": tdt.get("dropout", 0.0),
    "pool_type": enc.get("pool_type") or v5.get("pool_type") or "max",
    "pool_temp": enc.get("pool_temperature", 1.0),
    "pool_hidden": enc.get("pool_score_hidden", 0),
    "num_points": v5.get("num_points", 512),
    "zarr": get(cfg, "task", "dataset", "zarr_path", default="") or v5.get("zarr", ""),
    "use_ema": get(cfg, "training", "use_ema", default=True),
    "ckpt_task": get(cfg, "task", "name", default=""),
}

# the eval cfg decides which runner is built: compare it with the trained one
try:
    from omegaconf import OmegaConf
    trained_runner = get(cfg, "task", "env_runner", "_target_", default="")
    want_runner = OmegaConf.load(task_yaml).env_runner._target_
    vals["task_check"] = "ok" if trained_runner == want_runner else \
        "mismatch(trained=%s eval=%s)" % (trained_runner, want_runner)
except Exception as exc:                    # noqa: BLE001
    vals["task_check"] = "unchecked(%s)" % type(exc).__name__

for key, val in vals.items():
    print("%s=%s" % (key, val))
PY
    ); then
        while IFS= read -r _line; do
            [ -n "${_line}" ] || continue
            META["${_line%%=*}"]="${_line#*=}"
        done <<< "${_meta}"
        _resolved=1
    fi
fi

meta_or() { printf '%s' "${META[$1]:-${2}}"; }

if [ "${_resolved}" = "1" ]; then
    bn_hidden=${BN_HIDDEN:-$(meta_or bn_hidden 128,64,128)}
    future_steps=${FUTURE_STEPS:-$(meta_or future_steps 8)}
    tdt_layers=${TDT_LAYERS:-$(meta_or tdt_layers 2)}
    tdt_heads=${TDT_HEADS:-$(meta_or tdt_heads 4)}
    tdt_ff=${TDT_FF:-$(meta_or tdt_ff 1024)}
    tdt_dropout=${TDT_DROPOUT:-$(meta_or tdt_dropout 0.0)}
    pool_type=${POOL_TYPE:-$(meta_or pool_type softmax)}
    pool_temp=${POOL_TEMP:-$(meta_or pool_temp 1.0)}
    pool_hidden=${POOL_SCORE_HIDDEN:-$(meta_or pool_hidden 0)}
    num_points=${NUM_POINTS:-$(meta_or num_points 1024)}
    zarr_path=${ZARR:-$(meta_or zarr "${repo}/3D-Diffusion-Policy/data/${task_name}_expert.zarr")}
    ckpt_epoch=$(meta_or epoch -1)
    case "$(meta_or task_check ok)" in
        ok) ;;
        *) echo "[eval5] WARN: task/runner mismatch -- $(meta_or task_check '')" ;;
    esac
    case "$(meta_or use_ema True)" in
        True|true|1) ;;
        *) echo "[eval5] WARN: ckpt trained with use_ema=$(meta_or use_ema '?'); eval runs the raw model" ;;
    esac
else
    echo "[eval5] WARN: could not read the V5 hyper-parameters from the ckpt;"
    echo "[eval5]       falling back to BN_HIDDEN/TDT_*/POOL_* env vars + dp3 defaults"
    bn_hidden=${BN_HIDDEN:-128,64,128}
    future_steps=${FUTURE_STEPS:-8}
    tdt_layers=${TDT_LAYERS:-2}
    tdt_heads=${TDT_HEADS:-4}
    tdt_ff=${TDT_FF:-1024}
    tdt_dropout=${TDT_DROPOUT:-0.0}
    pool_type=${POOL_TYPE:-softmax}
    pool_temp=${POOL_TEMP:-1.0}
    pool_hidden=${POOL_SCORE_HIDDEN:-0}
    num_points=${NUM_POINTS:-1024}
    zarr_path=${ZARR:-"${repo}/3D-Diffusion-Policy/data/${task_name}_expert.zarr"}
    ckpt_epoch=-1
fi

# --- eval layout --------------------------------------------------------------
video_subdir="${distractor_suite:-clean}"
[ -n "${run_tag}" ] && video_subdir="${video_subdir}-${run_tag}"

if [ "${ckpt_sel}" = "latest" ]; then
    eval_dir="${run_dir}"
else
    # hydra outputs only: the exact ckpt file goes in via +training.resume_ckpt
    eval_dir="${run_dir}/eval_ckpt_${ckpt_tag}"
fi

# metrics json: <eval_dir>/media/<suite>/metrics_<tag>.json
metrics_root=${METRICS_DIR:-${eval_dir}/media}
metric_tag=${METRIC_TAG:-pijepa_v5}

# --- hydra overrides (same order as scripts/eval_policy.sh) -------------------
args=(
    --config-name=dp3.yaml
    task="${task_name}"
    hydra.run.dir="${eval_dir}"
    training.debug=False
    training.seed="${seed}"
    training.device="cuda:0"
    exp_name="${task_name}-${alg_tag}-${addition_info}"
    logging.mode=offline
    checkpoint.save_ckpt=False
    save_eval_video="${save_video}"
    video_subdir="${video_subdir}"
    task.env_runner.eval_episodes="${episodes}"
    ++task.env_runner.extra_steps_after_success="${extra_steps}"
    policy._target_="${v5_target}"
    "+policy.bottleneck_cfg.hidden=[${bn_hidden}]"
    +policy.tdt_cfg.future_steps="${future_steps}"
    +policy.tdt_cfg.layers="${tdt_layers}"
    +policy.tdt_cfg.heads="${tdt_heads}"
    +policy.tdt_cfg.dim_ff="${tdt_ff}"
    +policy.tdt_cfg.dropout="${tdt_dropout}"
    policy.pointcloud_encoder_cfg.pool_type="${pool_type}"
    policy.pointcloud_encoder_cfg.pool_temperature="${pool_temp}"
    policy.pointcloud_encoder_cfg.pool_score_hidden="${pool_hidden}"
    task.dataset.zarr_path="${zarr_path}"
)

# Only runners whose __init__ takes num_points (metaworld) accept the override;
# adroit/dexart fix N inside their gym wrapper (same rule as scripts/eval_policy.sh).
case "${task_name}" in
    metaworld_*) _takes_np=1 ;;
    *)           _takes_np=0 ;;
esac
if [ "${_takes_np}" = "1" ]; then
    args+=(++task.env_runner.num_points="${num_points}")
else
    echo "[eval5] runner has no num_points kwarg -> wrapper-fixed N=${num_points} stays in effect"
fi
if [ -n "${distractor_suite}" ]; then
    args+=("+task.env_runner.distractor_suite=${distractor_suite}")
fi
# optional pre-FPS distractor cluster (diffusion_policy_3d/cluster_distractor.py).
# Unset CLUSTER_POINTS -> no override at all, i.e. historical behaviour.
if [ -n "${CLUSTER_POINTS:-}" ]; then
    args+=("+task.env_runner.cluster_points=${CLUSTER_POINTS}")
    [ -n "${CLUSTER_CLUSTERS:-}" ]    && args+=("+task.env_runner.cluster_clusters=${CLUSTER_CLUSTERS}")
    [ -n "${CLUSTER_RADIUS_MODE:-}" ] && args+=("+task.env_runner.cluster_radius_mode=${CLUSTER_RADIUS_MODE}")
    [ -n "${CLUSTER_RADIUS_CM:-}" ]   && args+=("+task.env_runner.cluster_radius_cm=${CLUSTER_RADIUS_CM}")
    [ -n "${CLUSTER_SPEC:-}" ]        && args+=("+task.env_runner.cluster_spec=${CLUSTER_SPEC}")
fi
if [ -n "${ckpt_real}" ]; then
    args+=("+training.resume_ckpt=${ckpt_real}")
fi

# --- plan / run ---------------------------------------------------------------
echo "[eval5] task=${task_name} run_dir=${run_dir}"
echo "[eval5] ckpt=${ckpt_real:-${ckpt_path}} (epoch=${ckpt_epoch}) -> eval_dir=${eval_dir}"
echo "[eval5] suite=${suite} episodes=${episodes} extra_steps=${extra_steps}" \
     "save_video=${save_video} video_subdir=${video_subdir}"
if [ "${metrics_root}" != "none" ]; then
    echo "[eval5] metrics -> ${metrics_root}/${video_subdir}/metrics_${metric_tag}.json"
fi
echo "[eval5] v5: target=${v5_target##*.} bn=[${bn_hidden}] F=${future_steps}" \
     "tdt=${tdt_layers}/${tdt_heads}/${tdt_ff}/${tdt_dropout}"
echo "[eval5] v5: pool=${pool_type} T=${pool_temp} h=${pool_hidden} | N=${num_points} | zarr=${zarr_path}"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "+ ${py} eval.py ${args[*]}"
    exit 0
fi

log_path=${LOG:-${run_dir}/eval_logs/${video_subdir}_$(date +%Y%m%d-%H%M%S).log}
mkdir -p "$(dirname "${log_path}")"

export PYTHONPATH="${repo}/PI-JEPA:${repo}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export DR3D_QUIET=${DR3D_QUIET:-1}

set +e
if [ "${TEE:-1}" = "1" ]; then
    ( cd "${repo}/3D-Diffusion-Policy" && "${py}" eval.py "${args[@]}" ) 2>&1 | tee "${log_path}"
    rc=${PIPESTATUS[0]}
else
    ( cd "${repo}/3D-Diffusion-Policy" && "${py}" eval.py "${args[@]}" ) 2>&1
    rc=$?
fi
set -e
echo "[eval5] exit=${rc} log=${log_path}"

# --- machine-readable summary -------------------------------------------------
if [ "${SUMMARY:-1}" = "1" ] && [ -f "${log_path}" ]; then
    EV_LOG="${log_path}" EV_JSON="${log_path%.log}.json" \
    EV_RUN_DIR="${run_dir}" EV_EVAL_DIR="${eval_dir}" \
    EV_CKPT="${ckpt_real:-${ckpt_path}}" EV_CKPT_EPOCH="${ckpt_epoch}" \
    EV_TASK="${task_name}" EV_ADDITION_INFO="${addition_info}" EV_SEED="${seed}" \
    EV_SUITE="${suite}" EV_EPISODES="${episodes}" EV_EXTRA_STEPS="${extra_steps}" \
    EV_SAVE_VIDEO="${save_video}" EV_SUBDIR="${video_subdir}" EV_EXIT_CODE="${rc}" \
    EV_V5_TARGET="${v5_target}" EV_BN_HIDDEN="${bn_hidden}" \
    EV_FUTURE_STEPS="${future_steps}" \
    EV_TDT="${tdt_layers}/${tdt_heads}/${tdt_ff}/${tdt_dropout}" \
    EV_POOL="${pool_type}/${pool_temp}/${pool_hidden}" \
    EV_NUM_POINTS="${num_points}" EV_ZARR="${zarr_path}" \
    "${py}" - <<'PY'
import json
import os
import re

keys = ("test_mean_score", "mean_success_rates", "mean_n_goal_achieved",
        "mean_traj_rewards", "SR_test_L3", "SR_test_L5")
pat = re.compile(r"^(%s):\s*(-?[0-9.eE+]+)\s*$" % "|".join(keys))
scores = {}
for line in open(os.environ["EV_LOG"], errors="replace"):
    hit = pat.match(line.strip())
    if hit:
        scores[hit.group(1)] = float(hit.group(2))      # last line wins

prov = {k[3:].lower(): v for k, v in os.environ.items() if k.startswith("EV_")}
prov.pop("log", None)
prov.pop("json", None)
prov["scores"] = scores
with open(os.environ["EV_JSON"], "w") as handle:
    json.dump(prov, handle, indent=2, sort_keys=True)
line = " ".join("%s=%.4f" % (k, scores[k]) for k in keys if k in scores)
print("[eval5] " + (line or "no score parsed -- check the log"))
print("[eval5] summary -> " + os.environ["EV_JSON"])
PY
    if [ "${metrics_root}" != "none" ]; then
        mkdir -p "${metrics_root}/${video_subdir}"
        cp "${log_path%.log}.json" \
           "${metrics_root}/${video_subdir}/metrics_${metric_tag}.json"
    fi
fi

exit "${rc}"
