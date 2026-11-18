#!/usr/bin/env bash
# Ablation 1 -- eval wrapper for a single-view PI-JEPA checkpoint.
#
# The deploy graph is identical to v5 (mlp -> bottleneck -> pool -> final_proj|state
# -> UNet), so this only checks that the checkpoint really is a single-view run
# (cfg.pijepa_v5: l2_poll_weight == 0 and lambda_jepa_poll == 0) and then forwards
# every argument to scripts/eval_pijepa.sh, which reads the v5 hyper-parameters
# (BN_HIDDEN / FUTURE_STEPS / TDT_*) back from the ckpt.
#
# Usage:
#   bash PI-JEPA/ablation/eval_ablation1_single_view.sh <task> <tag> <seed> <gpu> \
#            [save_video=false] [episodes=50] [suite=false] [extra_steps=0] [run_tag] [ckpt]
#   <tag>    run dir tag <task>-pijepa-<tag>_seed<seed> (training default: pjv5sv)
#   <suite>  false | catgen | posgen | dynsphere | dynhand
#   [ckpt]   latest | best | epoch=NNNN-*.ckpt | /abs/path.ckpt
#   e.g. bash PI-JEPA/ablation/eval_ablation1_single_view.sh adroit_hammer pjv5sv 0 0
#        CKPT=best bash PI-JEPA/ablation/eval_ablation1_single_view.sh adroit_hammer pjv5sv 0 0 false 50 catgen 0
# env: ALLOW_ANY=1 skips the single-view check (debugging only) | CKPT= | OUT_ROOT= |
#      PYTHON= | DRY_RUN=1 | TEE= | SUMMARY= | LOG= | NUM_POINTS= | ZARR= | METRICS_DIR= |
#      METRIC_TAG= and the scripts/eval_pijepa.sh overrides (BN_HIDDEN / FUTURE_STEPS / ...
#      are ignored: the ckpt wins).
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

usage() {
    cat <<'EOF'
  usage: bash PI-JEPA/ablation/eval_ablation1_single_view.sh <task> <tag> <seed> <gpu> \
              [save_video=false] [episodes=50] [suite=false] [extra_steps=0] [run_tag] [ckpt]
  e.g.   bash PI-JEPA/ablation/eval_ablation1_single_view.sh adroit_hammer pjv5sv 0 0
         CKPT=best bash PI-JEPA/ablation/eval_ablation1_single_view.sh adroit_hammer pjv5sv 0 0 false 50 catgen 0
  ENV: ALLOW_ANY=1 | CKPT=latest|best|epoch=NNNN-*.ckpt|/abs/path | DRY_RUN=1 | OUT_ROOT= | PYTHON=
EOF
}

if [ -z "${task_name}" ] || [ -z "${addition_info}" ] || [ -z "${seed}" ] || [ -z "${gpu}" ]; then
    usage
    exit 1
fi

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
py=${PYTHON:-${repo}/.venv/bin/python}
out_root=${OUT_ROOT:-/hy-tmp/data/outputs}

# --- run dir / checkpoint (same convention as the trainer) ----------------------
source "${repo}/PI-JEPA/run_dir.sh"
alg_tag=pijepa
run_dir=$(ablation_run_dir "${out_root}" "${task_name}" "${alg_tag}" "${addition_info}" "${seed}")
if [ ! -d "${run_dir}" ]; then
    echo "[ablation1] run dir not found: ${run_dir}"
    echo "[ablation1] hint: <tag> names the dir (<task>-${alg_tag}-<tag>_seed<seed>),"
    echo "[ablation1]       the trainer default is pjv5sv; OUT_ROOT=${out_root}"
    exit 1
fi

ckpt_dir="${run_dir}/checkpoints"
case "${ckpt_sel}" in
    latest) ckpt_path="${ckpt_dir}/latest.ckpt" ;;
    best)   ckpt_path="${ckpt_dir}/best.ckpt" ;;
    /*)     ckpt_path="${ckpt_sel}" ;;
    *)      ckpt_path="${ckpt_dir}/${ckpt_sel}" ;;
esac
if [ ! -f "${ckpt_path}" ]; then
    echo "[ablation1] checkpoint not found: ${ckpt_path}"
    echo "[ablation1] hint: ls ${ckpt_dir}   (SAVE_BEST=0 runs have no best.ckpt)"
    [ "${DRY_RUN:-0}" = "1" ] || exit 1
    echo "[ablation1] DRY_RUN=1: continuing without the single-view check"
    ckpt_real=""
else
    ckpt_real=$(realpath -e "${ckpt_path}" 2>/dev/null || true)
fi

# --- single-view sanity check: poll terms must be unweighted -------------------
if [ "${ALLOW_ANY:-0}" = "1" ]; then
    echo "[ablation1] ALLOW_ANY=1: skipping the single-view check"
elif [ -n "${ckpt_real}" ]; then
    [ -x "${py}" ] || { echo "[ablation1] python not executable: ${py} (set PYTHON=)"; exit 1; }
    "${py}" - "${ckpt_real}" <<'PY' || exit 1
import sys

import dill
import torch

path = sys.argv[1]
try:
    payload = torch.load(path, pickle_module=dill, map_location="cpu",
                         weights_only=False, mmap=True)
except Exception:                            # old torch / fs without mmap support
    with open(path, "rb") as f:
        payload = torch.load(f, pickle_module=dill, map_location="cpu",
                             weights_only=False)

cfg = payload.get("cfg")
try:
    v5 = cfg["pijepa_v5"]
except Exception:                            # plain DP3 / older PI-JEPA ckpt
    print("[ablation1] %s has no cfg.pijepa_v5 -> not a PI-JEPA v5 checkpoint" % path)
    print("[ablation1] use scripts/eval_pijepa.sh (v5) or scripts/eval_policy.sh with PIJEPA=1 (older two-view ckpt)")
    sys.exit(1)

if v5 is None:
    print("[ablation1] cfg.pijepa_v5 is empty -> not a PI-JEPA v5 checkpoint")
    sys.exit(1)

w_p = float(v5.get("l2_poll_weight", -1.0))
lam_p = float(v5.get("lambda_jepa_poll", -1.0))
if abs(w_p) < 1e-12 and abs(lam_p) < 1e-12:
    print("[ablation1] single-view confirmed: l2_poll_weight=%g lambda_jepa_poll=%g" % (w_p, lam_p))
else:
    print("[ablation1] NOT a single-view checkpoint: l2_poll_weight=%g lambda_jepa_poll=%g"
          % (w_p, lam_p))
    print("[ablation1] this ckpt was trained with the polluted view in the objective;")
    print("[ablation1] evaluate it with bash scripts/eval_pijepa.sh %s %s %s %s"
          % (cfg.get("task", {}).get("name", "<task>"), "<tag>", "<seed>", "<gpu>"))
    print("[ablation1] (or set ALLOW_ANY=1 if you know what you are doing)")
    sys.exit(1)
PY
fi

echo "[ablation1] single-view | task=${task_name} tag=${addition_info} seed=${seed} ckpt=${ckpt_sel}"
echo "[ablation1] forward -> scripts/eval_pijepa.sh (v5 deploy graph, ckpt cfg wins)"

exec bash "${repo}/scripts/eval_pijepa.sh" \
     "${task_name}" "${addition_info}" "${seed}" "${gpu}" \
     "${save_video}" "${episodes}" "${suite}" "${extra_steps}" "${run_tag}" "${ckpt_sel}"
