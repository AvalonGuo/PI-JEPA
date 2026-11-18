#!/usr/bin/env bash
# Ablation 1 -- single-view PI-JEPA: the pollution branch is weighted to zero; v5 arch + TDT.
#   loss: l2_clean_weight*L2_clean + lambda_jepa_clean*L1_clean_seq
#
# The ablation is exactly L2_POLL_WEIGHT=0 LAMBDA_JEPA_POLL=0 -- no model change.  This script
# pins them, rejects non-zero overrides and defaults the tag to pjv5sv.  The poll branch is
# still forwarded and backwarded with weight 0 (so it contributes no update), which keeps L2p
# in pijepa_v5_metrics.jsonl as a free monitor; L1p is skipped by the lambda guard
# (PI-JEPA/train_pijepa.py:196) and is therefore absent from the metrics.  Deploy path is
# identical to v5 -> eval is eval_ablation1_single_view.sh (which wraps eval_pijepa.sh).
# Every other env var (FUTURE_STEPS GAMMA TDT_* BN_HIDDEN POOL_* L2_CLEAN_WEIGHT
# LAMBDA_JEPA_CLEAN BATCH LR WORKERS ...) is forwarded to scripts/train_pijepa.sh; see the
# env list in its header.
#
# Usage: bash PI-JEPA/ablation/train_ablation1_single_view.sh <task> <seed> <gpu> [tag=pjv5sv] [epochs=1000] [zarr_path]
#   e.g. bash PI-JEPA/ablation/train_ablation1_single_view.sh adroit_hammer 0 0
#        DRY_RUN=1 bash PI-JEPA/ablation/train_ablation1_single_view.sh adroit_hammer 0 0
#        BATCH=64 OUT_ROOT=/hy-tmp/out bash PI-JEPA/ablation/train_ablation1_single_view.sh metaworld_stick-push 0 0 pjv5sv 600
set -euo pipefail

task_name=${1:-}
seed=${2:-}
gpu=${3:-}
tag=${4:-pjv5sv}
epochs=${5:-1000}
zarr_arg=${6:-}

usage() {
    echo "  usage: bash PI-JEPA/ablation/train_ablation1_single_view.sh <task> <seed> <gpu> [tag=pjv5sv] [epochs=1000] [zarr_path]"
    echo "  e.g.   bash PI-JEPA/ablation/train_ablation1_single_view.sh adroit_hammer 0 0"
    echo "         DRY_RUN=1 bash PI-JEPA/ablation/train_ablation1_single_view.sh adroit_hammer 0 0"
}

if [ -z "${task_name}" ] || [ -z "${seed}" ] || [ -z "${gpu}" ]; then
    usage
    exit 1
fi

# --- guard: a non-zero poll weight would silently train two views ---------------
for _pair in "L2_POLL_WEIGHT:${L2_POLL_WEIGHT:-0}" "LAMBDA_JEPA_POLL:${LAMBDA_JEPA_POLL:-0}"; do
    _name=${_pair%%:*}
    _val=${_pair#*:}
    case "${_val}" in
        0|0.0|0.00|0.000|0.0000) ;;
        *) echo "[ablation1] ${_name}=${_val} conflicts with single-view (the polluted view must stay unweighted)"
           echo "[ablation1] unset it (or set it to 0) -- the weight is pinned to 0 here"
           exit 1 ;;
    esac
done
unset _pair _name _val

# --- ablation1 invariants -------------------------------------------------------
export L2_POLL_WEIGHT=0          # no L2 on the polluted view
export LAMBDA_JEPA_POLL=0        # no JEPA L1 on the polluted view

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
launcher="${repo}/scripts/train_pijepa.sh"
if [ ! -f "${launcher}" ]; then
    echo "[ablation1] launcher not found: ${launcher}"
    exit 1
fi

echo "[ablation1] single-view PI-JEPA | poll view weight = 0 (L2_poll, L1_poll)"
echo "[ablation1] loss = ${L2_CLEAN_WEIGHT:-1.0}*L2_clean + ${LAMBDA_JEPA_CLEAN:-1}*L1_clean_seq" \
     "+ 0*L1_poll_seq | F=$( [ -n "${FUTURE_STEPS:-}" ] && echo "${FUTURE_STEPS}" || echo 8 )" \
     "gamma=${GAMMA:-0.5} l1_mode=${L1_MODE:-sum}"
echo "[ablation1] task=${task_name} tag=${tag} seed=${seed} epochs=${epochs} gpu=${gpu} batch=${BATCH:-128}"
echo "[ablation1] eval: bash PI-JEPA/ablation/eval_ablation1_single_view.sh ${task_name} ${tag} ${seed} ${gpu}"

exec bash "${launcher}" \
     "${task_name}" "${tag}" "${seed}" "${gpu}" "${epochs}" ${zarr_arg:+"${zarr_arg}"}
