#!/usr/bin/env bash
# Ablation 2 -- PI-JEPA v5 without the JEPA auxiliary loss (no L1); two-view L2 kept.
#   clean/poll: two DP3 views (poll = voxel clutter)
#   loss: l2_clean_weight*L2_clean + l2_poll_weight*L2_poll   (ablation1 also zeroes L2_poll)
#
# The ablation is exactly LAMBDA_JEPA_CLEAN=0 LAMBDA_JEPA_POLL=0 -- no model change.  This
# script pins them, rejects non-zero overrides and defaults the tag to pjv5nl1.  train_step_v5
# drops the L1 loss terms on that guard (PI-JEPA/train_pijepa.py:196) -> no auxiliary
# gradient; but encode_views (line 188) still runs, so the TDT forward and the future-frame
# target are computed and discarded: same tree as v5, not cheaper.  Deploy path is identical
# -> eval is eval_ablation2_no_l1.sh (which wraps eval_pijepa.sh).  Every other env var
# (FUTURE_STEPS GAMMA TDT_* BN_HIDDEN POOL_* L2_*_WEIGHT BATCH LR WORKERS ...) is forwarded
# to scripts/train_pijepa.sh; see the env list in its header.
#
# Usage: bash PI-JEPA/ablation/train_ablation2_no_l1.sh <task> <seed> <gpu> [tag=pjv5nl1] [epochs=1000] [zarr_path]
#   e.g. bash PI-JEPA/ablation/train_ablation2_no_l1.sh adroit_hammer 0 0
#        DRY_RUN=1 bash PI-JEPA/ablation/train_ablation2_no_l1.sh adroit_hammer 0 0
#        BATCH=64 OUT_ROOT=/hy-tmp/out bash PI-JEPA/ablation/train_ablation2_no_l1.sh metaworld_stick-push 0 0 pjv5nl1 600
set -euo pipefail

task_name=${1:-}
seed=${2:-}
gpu=${3:-}
tag=${4:-pjv5nl1}
epochs=${5:-1000}
zarr_arg=${6:-}

usage() {
    echo "  usage: bash PI-JEPA/ablation/train_ablation2_no_l1.sh <task> <seed> <gpu> [tag=pjv5nl1] [epochs=1000] [zarr_path]"
    echo "  e.g.   bash PI-JEPA/ablation/train_ablation2_no_l1.sh adroit_hammer 0 0"
    echo "         DRY_RUN=1 bash PI-JEPA/ablation/train_ablation2_no_l1.sh adroit_hammer 0 0"
}

if [ -z "${task_name}" ] || [ -z "${seed}" ] || [ -z "${gpu}" ]; then
    usage
    exit 1
fi

# --- guard: a non-zero L1 weight would silently re-enable the auxiliary loss -----
for _pair in "LAMBDA_JEPA_CLEAN:${LAMBDA_JEPA_CLEAN:-0}" "LAMBDA_JEPA_POLL:${LAMBDA_JEPA_POLL:-0}"; do
    _name=${_pair%%:*}
    _val=${_pair#*:}
    case "${_val}" in
        0|0.0|0.00|0.000|0.0000) ;;
        *) echo "[ablation2] ${_name}=${_val} conflicts with no-L1 (the JEPA auxiliary loss must stay off)"
           echo "[ablation2] unset it (or set it to 0) -- the weight is pinned to 0 here"
           exit 1 ;;
    esac
done
unset _pair _name _val

# --- ablation2 invariants -------------------------------------------------------
export LAMBDA_JEPA_CLEAN=0       # no JEPA L1 on the clean view
export LAMBDA_JEPA_POLL=0        # no JEPA L1 on the polluted view

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
launcher="${repo}/scripts/train_pijepa.sh"
if [ ! -f "${launcher}" ]; then
    echo "[ablation2] launcher not found: ${launcher}"
    exit 1
fi

echo "[ablation2] no-L1 PI-JEPA | JEPA auxiliary weight = 0 (L1_clean, L1_poll)"
echo "[ablation2] loss = ${L2_CLEAN_WEIGHT:-1.0}*L2_clean + ${L2_POLL_WEIGHT:-0.3}*L2_poll (two views, no L1)"
echo "[ablation2] TDT output / L1 warmup do not affect the loss (L1_WARMUP_EPOCHS=${L1_WARMUP_EPOCHS:-200} no-op; TDT forward still runs)"
echo "[ablation2] task=${task_name} tag=${tag} seed=${seed} epochs=${epochs} gpu=${gpu} batch=${BATCH:-128}"
echo "[ablation2] eval: bash PI-JEPA/ablation/eval_ablation2_no_l1.sh ${task_name} ${tag} ${seed} ${gpu}"

exec bash "${launcher}" \
     "${task_name}" "${tag}" "${seed}" "${gpu}" "${epochs}" ${zarr_arg:+"${zarr_arg}"}
