#!/usr/bin/env bash
# Ablation 3 -- eval wrapper for a w/o-Refiner PI-JEPA checkpoint.
#
# Ground truth is the checkpoint's own state dict: a v5 / ablation1 / ablation2 ckpt
# carries the refiner (`obs_encoder.bottleneck.*`), an ablation3 ckpt has none.  After
# that check every argument is forwarded to scripts/eval_pijepa.sh with
# V5_TARGET=ablation3_policy.PIJEPAPolicyV5NoRefiner, so eval.py rebuilds the same
# (refiner-free) module tree and train.py's strict load_state_dict succeeds.
#
# That class accepts and ignores bottleneck_cfg, so the [eval5] line still prints a
# bn=[...] width taken from the ckpt/dp3 default -- it never reaches the model.
#
# Usage:
#   bash PI-JEPA/ablation/eval_ablation3_no_refiner.sh <task> <tag> <seed> <gpu> \
#            [save_video=false] [episodes=50] [suite=false] [extra_steps=0] [run_tag] [ckpt]
#   <tag>    run dir tag <task>-pijepa-<tag>_seed<seed> (training default: pjv5nr)
#   <suite>  false | catgen | posgen | dynsphere | dynhand
#   [ckpt]   latest | best | epoch=NNNN-*.ckpt | /abs/path.ckpt
#   e.g. bash PI-JEPA/ablation/eval_ablation3_no_refiner.sh adroit_hammer pjv5nr 0 0
#        CKPT=best bash PI-JEPA/ablation/eval_ablation3_no_refiner.sh adroit_hammer pjv5nr 0 0 false 50 catgen 0
# env: ALLOW_ANY=1 skips the refiner check (debugging only) | CKPT= | OUT_ROOT= | PYTHON= |
#      DRY_RUN=1 | TEE= | SUMMARY= | LOG= | NUM_POINTS= | ZARR= | METRICS_DIR= | METRIC_TAG=
#      and the eval_pijepa.sh overrides (FUTURE_STEPS / TDT_* / POOL_*: the ckpt wins).
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
  usage: bash PI-JEPA/ablation/eval_ablation3_no_refiner.sh <task> <tag> <seed> <gpu> \
              [save_video=false] [episodes=50] [suite=false] [extra_steps=0] [run_tag] [ckpt]
  e.g.   bash PI-JEPA/ablation/eval_ablation3_no_refiner.sh adroit_hammer pjv5nr 0 0
         CKPT=best bash PI-JEPA/ablation/eval_ablation3_no_refiner.sh adroit_hammer pjv5nr 0 0 false 50 catgen 0
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
target="ablation3_policy.PIJEPAPolicyV5NoRefiner"

# --- run dir / checkpoint (same convention as the trainer) ----------------------
source "${repo}/PI-JEPA/run_dir.sh"
alg_tag=pijepa
run_dir=$(ablation_run_dir "${out_root}" "${task_name}" "${alg_tag}" "${addition_info}" "${seed}")
if [ ! -d "${run_dir}" ]; then
    echo "[ablation3] run dir not found: ${run_dir}"
    echo "[ablation3] hint: <tag> names the dir (<task>-${alg_tag}-<tag>_seed<seed>),"
    echo "[ablation3]       the trainer default is pjv5nr; OUT_ROOT=${out_root}"
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
    echo "[ablation3] checkpoint not found: ${ckpt_path}"
    echo "[ablation3] hint: ls ${ckpt_dir}   (SAVE_BEST=0 runs have no best.ckpt)"
    [ "${DRY_RUN:-0}" = "1" ] || exit 1
    echo "[ablation3] DRY_RUN=1: continuing without the refiner check"
    ckpt_real=""
else
    ckpt_real=$(realpath -e "${ckpt_path}" 2>/dev/null || true)
fi

# --- refiner check: the module must be absent from the state dict --------------
if [ "${ALLOW_ANY:-0}" = "1" ]; then
    echo "[ablation3] ALLOW_ANY=1: skipping the refiner check"
elif [ -n "${ckpt_real}" ]; then
    [ -x "${py}" ] || { echo "[ablation3] python not executable: ${py} (set PYTHON=)"; exit 1; }
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

state = (payload.get("state_dicts") or {}).get("model") or {}
keys = list(state.keys())
if not keys:
    print("[ablation3] %s has no state_dicts['model'] -> cannot verify" % path)
    sys.exit(1)

bn = [k for k in keys if k.startswith("obs_encoder.bottleneck.")]
if not bn:
    print("[ablation3] refiner=none confirmed: %d state-dict entries, no "
          "obs_encoder.bottleneck.* key" % len(keys))
    sys.exit(0)

cfg = payload.get("cfg") or {}
v5 = cfg.get("pijepa_v5") or {}
lam_c = float(v5.get("lambda_jepa_clean", 1.0))
lam_p = float(v5.get("lambda_jepa_poll", 1.0))
w_p = float(v5.get("l2_poll_weight", -1.0))
task = (cfg.get("task") or {}).get("name", "<task>")
print("[ablation3] NOT a w/o-Refiner checkpoint: %d obs_encoder.bottleneck.* key(s), "
      "e.g. %s" % (len(bn), bn[0]))
if abs(lam_c) < 1e-12 and abs(lam_p) < 1e-12:
    if abs(w_p) < 1e-12:
        print("[ablation3] this is an ABLATION1 (single view) ckpt:")
        print("[ablation3] bash PI-JEPA/ablation/eval_ablation1_single_view.sh %s <tag> <seed> <gpu>" % task)
    else:
        print("[ablation3] this is an ABLATION2 (no L1) ckpt:")
        print("[ablation3] bash PI-JEPA/ablation/eval_ablation2_no_l1.sh %s <tag> <seed> <gpu>" % task)
else:
    print("[ablation3] this is a v5 ckpt (refiner present):")
    print("[ablation3] bash scripts/eval_pijepa.sh %s <tag> <seed> <gpu>" % task)
print("[ablation3] (or set ALLOW_ANY=1 if you know what you are doing)")
sys.exit(1)
PY
fi

echo "[ablation3] w/o Refiner | task=${task_name} tag=${addition_info} seed=${seed} ckpt=${ckpt_sel}"
echo "[ablation3] target=${target} (bottleneck_cfg ignored) -> scripts/eval_pijepa.sh"
export V5_TARGET="${target}"
export PYTHONPATH="${repo}/PI-JEPA/ablation${PYTHONPATH:+:${PYTHONPATH}}"

exec bash "${repo}/scripts/eval_pijepa.sh" \
     "${task_name}" "${addition_info}" "${seed}" "${gpu}" \
     "${save_video}" "${episodes}" "${suite}" "${extra_steps}" "${run_tag}" "${ckpt_sel}"
