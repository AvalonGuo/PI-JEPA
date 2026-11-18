DEBUG=False
wandb_mode=offline
save_ckpt=False

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed=${4}
exp_name=${task_name}-${alg_name}-${addition_info}
# RUN_DIR lets an ablation run (e.g. ...-meanpool_seed0) be evaluated without
# editing this script; default keeps the baseline path unchanged.
run_dir=${RUN_DIR:-"/hy-tmp/data/outputs/${exp_name}_seed${seed}"}
# fail fast when the run dir has no checkpoint: eval.py would otherwise evaluate a
# randomly initialised policy ("AttributeError: 'ParameterDict' object has no
# attribute 'point_cloud'" at the first predict_action).
_ckpt_sel=${CKPT:-latest}
case "${_ckpt_sel}" in
    latest) _ckpt="${run_dir}/checkpoints/latest.ckpt" ;;
    best)   _ckpt="${run_dir}/checkpoints/best.ckpt" ;;
    /*)     _ckpt="${_ckpt_sel}" ;;
    *.ckpt) _ckpt="${run_dir}/checkpoints/${_ckpt_sel}" ;;
    *)      _ckpt="${run_dir}/checkpoints/${_ckpt_sel}.ckpt" ;;
esac
# a rotating topk manager (checkpoint.topk.k > 1) can leave a dangling symlink
if [ -L "${_ckpt}" ] && [ ! -e "${_ckpt}" ]; then
    echo "[eval_policy] ${_ckpt} is a dangling symlink -> $(readlink "${_ckpt}" 2>/dev/null)"
    echo "[eval_policy] its target was rotated away; use CKPT=latest or an explicit file"
    [ "${DRY_RUN:-0}" = "1" ] || exit 1
fi
if [ ! -f "${_ckpt}" ]; then
    echo "[eval_policy] no checkpoint: ${_ckpt}"
    echo "[eval_policy] run_dir=${run_dir}  (RUN_DIR override or <task>-<alg>-<info>_seed<seed>)"
    echo "[eval_policy] hint: pass the dir that really holds checkpoints/${_ckpt_sel}.ckpt,"
    echo "[eval_policy]       e.g. RUN_DIR=/hy-tmp/data/outputs/<task>-<alg>-<info>_seed<seed>"
    if [ "${DRY_RUN:-0}" != "1" ]; then
        exit 1
    fi
fi

# absolute path so eval.py (which chdir's to /root) can still find it
_ckpt_real=$(realpath -e "${_ckpt}" 2>/dev/null || true)

# eval_dir holds hydra outputs only; the weights go in via training.resume_ckpt
eval_dir="${run_dir}"
if [ "${_ckpt_sel}" != "latest" ]; then
    _ckpt_tag=$(basename "${_ckpt}" .ckpt | tr '=' '-')
    eval_dir="${run_dir}/eval_ckpt_${_ckpt_tag}"
fi
_ckpt_info="$(stat -c %s "${_ckpt_real}" 2>/dev/null) B, $(stat -c %y "${_ckpt_real}" 2>/dev/null | cut -d. -f1)"
if [ -z "${_ckpt_real}" ]; then
    echo "[eval_policy] ckpt=${_ckpt} (unresolved; dry-run only)"
elif [ "${_ckpt}" = "${_ckpt_real}" ]; then
    echo "[eval_policy] ckpt=${_ckpt_real} (${_ckpt_info})"
else
    # rotated topk symlink: show both the alias and the file it resolves to
    echo "[eval_policy] ckpt=${_ckpt} -> ${_ckpt_real} (${_ckpt_info})"
fi
if [ "${_ckpt_sel}" = "latest" ]; then
    echo "[eval_policy] eval_dir=${eval_dir} (run dir, same as the ckpt's)"
elif [ -z "${_ckpt_real}" ]; then
    echo "[eval_policy] eval_dir=${eval_dir} (hydra outputs only; no ckpt resolved)"
else
    echo "[eval_policy] eval_dir=${eval_dir} (hydra outputs only; ckpt passed via training.resume_ckpt)"
fi

gpu_id=${5}
save_video=${6:-true}
eval_episodes=${7:-50}
distractor_arg=${8:-false}
extra_steps=${9:-${EXTRA_STEPS_AFTER_SUCCESS:-15}}
case "${extra_steps}" in
    ''|*[!0-9]*)
        echo "[eval_policy] arg9 (extra_steps_after_success) must be an integer >= 0 (0 = 不截断), got: '${extra_steps}'"
        echo "  usage: bash scripts/eval_policy.sh <alg> <task> <info> <seed> <gpu> [save_video] [episodes] [false|catgen|posgen|dynsphere|dynhand] [extra_steps=15] [run_tag]"
        exit 1 ;;
esac
run_tag=${10:-}
distractor_suite=""
case "$distractor_arg" in
  false)      ;;
  catgen)     distractor_suite=catgen ;;
  posgen)     distractor_suite=posgen ;;
  dynsphere)  distractor_suite=dynsphere ;;
  dynhand)    distractor_suite=dynhand ;;
  *) echo "[eval_policy] arg8 must be false|catgen|posgen|dynsphere|dynhand, got: ${distractor_arg}"; exit 1 ;;
esac
if [ -n "${distractor_suite}" ]; then
    video_subdir="${distractor_suite}"
else
    video_subdir="clean"
fi
[ -n "${run_tag}" ] && video_subdir="${video_subdir}-${run_tag}"

repo_root="$(cd "$(dirname "$0")/.." && pwd)"

# num_points (FPS rows): NUM_POINTS > ckpt-recorded N (pijepa block, else the
# training zarr's rows) > ZARR_PATH's rows > benchmark default
# (adroit 512 | metaworld 1024 | dexart/realdex 1024). mmap only: no 4 GB RSS.
_np_probe=$(
    "${PYTHON:-python}" - "${_ckpt_real:-${_ckpt}}" "${ZARR_PATH:-}" \
        "${repo_root}/3D-Diffusion-Policy/diffusion_policy_3d/config/task/${task_name}.yaml" <<'PY' 2>/dev/null
import os
import sys

ckpt, zarr_override, task_yaml = sys.argv[1:4]


def stored_rows(path):
    """Rows of data/point_cloud in a zarr store (None when unreadable)."""
    try:
        import zarr
        return int(zarr.open(path, mode="r")["data"]["point_cloud"].shape[1])
    except Exception:
        return None


def ckpt_rows(path):
    """(rows, source) recorded by the training run."""
    if not (path and os.path.isfile(path)):
        return None, "no-ckpt"
    try:
        import dill
        import torch
        payload = torch.load(path, pickle_module=dill, map_location="cpu",
                             weights_only=False, mmap=True)
        cfg = payload.get("cfg")
    except Exception as exc:                       # old torch / fs without mmap
        return None, "ckpt-unreadable(%s)" % type(exc).__name__
    for key in ("pijepa_v5", "pijepa", "pijepa_v4"):
        block = cfg.get(key) if cfg is not None else None
        try:
            n = int(block.get("num_points"))
        except Exception:
            continue
        if n > 0:
            return n, "ckpt:%s.num_points" % key
    try:
        zpath = str(cfg.task.dataset.zarr_path)
    except Exception:
        zpath = ""
    if not zpath:
        return None, "ckpt-no-zarr"
    n = stored_rows(zpath)
    if n:
        return n, "ckpt:zarr(%s)" % os.path.basename(zpath)
    return None, "ckpt-zarr-unreadable"


def runner_takes_num_points(path):
    """Only MetaworldRunner.__init__ accepts num_points (adroit/dexart do not)."""
    try:
        from omegaconf import OmegaConf
        target = str(OmegaConf.load(path).env_runner._target_)
    except Exception:
        return None
    return target.endswith("metaworld_runner.MetaworldRunner")


rows, source = ckpt_rows(ckpt)
if rows is None and zarr_override:
    rows = stored_rows(zarr_override)
    if rows:
        source = "ZARR_PATH(%s)" % os.path.basename(zarr_override)
print("num_points=%s" % (rows if rows else "NA"))
print("num_points_source=%s" % source)
takes = runner_takes_num_points(task_yaml)
print("runner_num_points=%s" % ("" if takes is None else int(takes)))
PY
)
_np=""
_np_src=""
_takes_np=""
if [ -n "${_np_probe}" ]; then
    while IFS='=' read -r _np_key _np_val; do
        case "${_np_key}" in
            num_points)        [ "${_np_val}" = "NA" ] || _np=${_np_val} ;;
            num_points_source) _np_src=${_np_val} ;;
            runner_num_points) [ -n "${_np_val}" ] && _takes_np=${_np_val} ;;
        esac
    done <<< "${_np_probe}"
fi
case "${task_name}" in
    adroit_*)           _bench_np=512  ;;
    dexart_*|realdex_*) _bench_np=1024 ;;
    metaworld_*)        _bench_np=1024 ;;
    *)                  _bench_np=512  ;;
esac
if [ -n "${NUM_POINTS:-}" ]; then
    _np=${NUM_POINTS}
    _np_src="env:NUM_POINTS"
elif [ -z "${_np}" ]; then
    _np=${_bench_np}
    _np_src="benchmark(${_np_src:-probe-unavailable})"
fi
if [ -z "${_takes_np}" ]; then
    case "${task_name}" in
        metaworld_*) _takes_np=1 ;;
        *)           _takes_np=0 ;;
    esac
fi
case "${_np}" in
    ''|*[!0-9]*)
        echo "[eval_policy] num_points must be a positive integer, got: '${_np}' (source=${_np_src})"
        exit 1 ;;
esac
echo "[eval_policy] num_points=${_np} (source=${_np_src}, benchmark=${_bench_np}, num_points_kwarg=${_takes_np})"
if [ "${_np}" != "${_bench_np}" ]; then
    echo "[eval_policy] WARN: N=${_np} != benchmark default ${_bench_np}; using ${_np} (NUM_POINTS= overrides)"
fi

cd 3D-Diffusion-Policy

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}
export DR3D_QUIET=${DR3D_QUIET:-1}
_base_args=(
    --config-name=${config_name}.yaml
    task=${task_name}
    hydra.run.dir=${eval_dir}
    training.debug=$DEBUG
    training.seed=${seed}
    training.device="cuda:0"
    exp_name=${exp_name}
    logging.mode=${wandb_mode}
    checkpoint.save_ckpt=${save_ckpt}
    save_eval_video=${save_video}
    video_subdir=${video_subdir}
    task.env_runner.eval_episodes=${eval_episodes}
)
_base_args+=(
    ++task.env_runner.extra_steps_after_success=${extra_steps}
)
_extra_args=()
if [ -n "${distractor_suite}" ]; then
    _extra_args+=(+task.env_runner.distractor_suite=${distractor_suite})
fi
# optional pre-FPS distractor cluster (diffusion_policy_3d/cluster_distractor.py).
# Unset CLUSTER_POINTS -> no override at all, i.e. historical behaviour.
if [ -n "${CLUSTER_POINTS:-}" ]; then
    _extra_args+=(+task.env_runner.cluster_points=${CLUSTER_POINTS})
    [ -n "${CLUSTER_CLUSTERS:-}" ]    && _extra_args+=(+task.env_runner.cluster_clusters=${CLUSTER_CLUSTERS})
    [ -n "${CLUSTER_RADIUS_MODE:-}" ] && _extra_args+=(+task.env_runner.cluster_radius_mode=${CLUSTER_RADIUS_MODE})
    [ -n "${CLUSTER_RADIUS_CM:-}" ]   && _extra_args+=(+task.env_runner.cluster_radius_cm=${CLUSTER_RADIUS_CM})
    [ -n "${CLUSTER_SPEC:-}" ]        && _extra_args+=(+task.env_runner.cluster_spec=${CLUSTER_SPEC})
fi
# hand the resolved ckpt to train.py::eval(); without it the workspace looks for
# <hydra.run.dir>/checkpoints/latest.ckpt, which eval_dir does not contain
_resume_args=()
if [ -n "${_ckpt_real}" ]; then
    _resume_args+=(++training.resume_ckpt="${_ckpt_real}")
fi
# optional encoder pooling override: eval.py rebuilds the model from the CLI cfg
# (not from the ckpt cfg), so an ablation ckpt trained with mean/softmax pooling
# MUST pass POOL_TYPE=<pool> here, otherwise it is silently evaluated with the
# dp3.yaml default (max). POOL_TEMP / POOL_SCORE_HIDDEN must match training too.
_pool_args=()
if [ -n "${POOL_TYPE:-}" ]; then
    case "${POOL_TYPE}" in
      max|mean|softmax) _pool_args+=(policy.pointcloud_encoder_cfg.pool_type=${POOL_TYPE}) ;;
      *) echo "[eval_policy] POOL_TYPE must be max|mean|softmax, got: '${POOL_TYPE}'"; exit 1 ;;
    esac
fi
if [ -n "${POOL_TEMP:-}" ]; then
    _pool_args+=(policy.pointcloud_encoder_cfg.pool_temperature=${POOL_TEMP})
fi
if [ -n "${POOL_SCORE_HIDDEN:-}" ]; then
    _pool_args+=(policy.pointcloud_encoder_cfg.pool_score_hidden=${POOL_SCORE_HIDDEN})
fi
# PI-JEPA checkpoints carry `obs_encoder.bottleneck.*`.  eval.py rebuilds the policy
# from the CLI cfg (not from the ckpt cfg), so the class + bottleneck width must be
# passed here; PYTHONPATH makes the PI-JEPA/ modules importable from the eval process
# (eval.py chdir's to /root, so the repo root is not on sys.path otherwise).
# PIJEPA_V5 checkpoints additionally carry `obs_encoder.tdt.*`: the TDT is train-only
# (the TDT is train-only) but it stays in the module tree, and train.py reloads with
# strict=True, so the class AND the TDT shape must match the training values.
_ablation_args=()
if [ "${PIJEPA:-0}" = "1" ] || [ "${PIJEPA_V5:-0}" = "1" ]; then
    export PYTHONPATH="${repo_root}/PI-JEPA:${repo_root}:${PYTHONPATH:-}"
    _pijepa_target=pijepa_policy.PIJEPAPolicy
    if [ "${PIJEPA_V5:-0}" = "1" ]; then
        _pijepa_target=pijepa_policy.PIJEPAPolicyV5
    fi
    # `_target_` already exists in dp3.yaml -> plain override (a `+` prefix makes
    # hydra raise "Could not append to config"); `bottleneck_cfg` is new -> `+`.
    _ablation_args+=(
        policy._target_=${_pijepa_target}
        +policy.bottleneck_cfg.hidden="[${BN_HIDDEN:-128,64,128}]"
    )
    if [ "${PIJEPA_V5:-0}" = "1" ]; then
        _ablation_args+=(
            +policy.tdt_cfg.future_steps="${FUTURE_STEPS:-8}"
            +policy.tdt_cfg.layers="${TDT_LAYERS:-2}"
            +policy.tdt_cfg.heads="${TDT_HEADS:-4}"
            +policy.tdt_cfg.dim_ff="${TDT_FF:-1024}"
            +policy.tdt_cfg.dropout="${TDT_DROPOUT:-0.0}"
        )
        echo "[eval_policy] PIJEPA_V5: target=${_pijepa_target} F=${FUTURE_STEPS:-8} layers=${TDT_LAYERS:-2} heads=${TDT_HEADS:-4} ff=${TDT_FF:-1024} (must match training)"
    fi
fi
# same zarr as training: eval fits the normalizer from it before loading the ckpt
if [ -n "${ZARR_PATH:-}" ]; then
    _ablation_args+=(task.dataset.zarr_path="${ZARR_PATH}")
fi
# Only runners whose __init__ takes num_points get the flag (Adroit/DexArt
# hard-code 512/1024 in their wrappers). `++` = add-or-override.
if [ "${_takes_np}" = "1" ]; then
    _ablation_args+=(++task.env_runner.num_points="${_np}")
else
    echo "[eval_policy] runner has no num_points kwarg -> wrapper-fixed N=${_bench_np} stays in effect"
    if [ "${_np}" != "${_bench_np}" ]; then
        echo "[eval_policy] WARN: resolved N=${_np} != wrapper-fixed N=${_bench_np}; the runner still feeds ${_bench_np} rows"
    fi
fi
# record which file was evaluated (eval_dir holds no ckpt copy to inspect later)
if [ "${DRY_RUN:-0}" != "1" ] && [ -n "${_ckpt_real}" ]; then
    mkdir -p "${eval_dir}"
    printf '{"ckpt":"%s","ckpt_sel":"%s","size":%s,"mtime":"%s","suite":"%s","episodes":%s,"extra_steps":%s,"seed":%s}\n' \
        "${_ckpt_real}" "${_ckpt_sel}" "$(stat -c %s "${_ckpt_real}" 2>/dev/null || echo null)" \
        "$(stat -c %y "${_ckpt_real}" 2>/dev/null | cut -d. -f1)" \
        "${distractor_suite:-clean}" "${eval_episodes}" "${extra_steps}" "${seed}" \
        > "${eval_dir}/ckpt_used.json" || true
fi
if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "+ ${PYTHON:-python} eval.py ${_base_args[*]} ${_extra_args[*]} ${_resume_args[*]} ${_pool_args[*]} ${_ablation_args[*]}"
else
    "${PYTHON:-python}" eval.py "${_base_args[@]}" "${_extra_args[@]}" "${_resume_args[@]}" "${_pool_args[@]}" "${_ablation_args[@]}"
fi
