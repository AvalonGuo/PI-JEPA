#!/usr/bin/env bash
# Ablation 3 -- PI-JEPA v5 without the Refiner (PIJEPA-BottleNeck); the TDT is kept.
#   clean/poll: pc -> extractor.mlp -> softmax pool -> final_proj|state -> UNet
#   TDT input : pool_task(mlp(clean|poll))            (no refiner)
#   TDT target: pool_task(mlp(obs frames|future))     (stop-grad)
#   loss      : v5 defaults (L2_clean, L2_poll, lambda_jepa_* * SmoothL1)
#
# The ablation is structural: obs_encoder.bottleneck does not exist (0 params, no
# `obs_encoder.bottleneck.*` keys), so this ckpt is NOT interchangeable with v5 /
# ablation1 / ablation2 -- evaluate it with eval_ablation3_no_refiner.sh.
# Implementation: PI-JEPA/ablation/ablation3_policy.py (the two classes) +
# train_ablation3_no_refiner.py (patches build_policy + save_ckpt of the v5 trainer and
# writes ablation3_no_refiner.json into the run dir); everything else (data, losses,
# warmup, EMA, ckpt cadence, loop) is scripts/train_pijepa.sh.
#
# Usage: bash PI-JEPA/ablation/train_ablation3_no_refiner.sh <task> <seed> <gpu> [tag=pjv5nr] [epochs=1000] [zarr_path]
#   e.g. bash PI-JEPA/ablation/train_ablation3_no_refiner.sh adroit_hammer 0 0
#        DRY_RUN=1 bash PI-JEPA/ablation/train_ablation3_no_refiner.sh adroit_hammer 0 0
#        BATCH=64 OUT_ROOT=/hy-tmp/out bash PI-JEPA/ablation/train_ablation3_no_refiner.sh metaworld_stick-push 0 0 pjv5nr 600
# env: TRAIN_ENTRY is set here (setting it externally is an error); BN_HIDDEN is ignored
#      (there is no refiner); every other env var (FUTURE_STEPS GAMMA TDT_* POOL_*
#      L2_*_WEIGHT LAMBDA_JEPA_* BATCH LR WORKERS OUT_ROOT DRY_RUN ...) is forwarded to
#      scripts/train_pijepa.sh -- see the env list in its header.
set -euo pipefail

task_name=${1:-}
seed=${2:-}
gpu=${3:-}
tag=${4:-pjv5nr}
epochs=${5:-1000}
zarr_arg=${6:-}

usage() {
    echo "  usage: bash PI-JEPA/ablation/train_ablation3_no_refiner.sh <task> <seed> <gpu> [tag=pjv5nr] [epochs=1000] [zarr_path]"
    echo "  e.g.   bash PI-JEPA/ablation/train_ablation3_no_refiner.sh adroit_hammer 0 0"
    echo "         DRY_RUN=1 bash PI-JEPA/ablation/train_ablation3_no_refiner.sh adroit_hammer 0 0"
}

if [ -z "${task_name}" ] || [ -z "${seed}" ] || [ -z "${gpu}" ]; then
    usage
    exit 1
fi

# --- guards: the refiner must not come back through the environment --------------
if [ -n "${TRAIN_ENTRY:-}" ]; then
    echo "[ablation3] TRAIN_ENTRY=${TRAIN_ENTRY} conflicts with this wrapper (it sets its own trainer)"
    exit 1
fi
if [ -n "${BN_HIDDEN:-}" ]; then
    echo "[ablation3] BN_HIDDEN=${BN_HIDDEN} ignored: this ablation has no refiner"
fi

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
entry="${repo}/PI-JEPA/ablation/train_ablation3_no_refiner.py"
if [ ! -f "${entry}" ]; then
    echo "[ablation3] trainer not found: ${entry}"
    exit 1
fi
launcher="${repo}/scripts/train_pijepa.sh"
if [ ! -f "${launcher}" ]; then
    echo "[ablation3] launcher not found: ${launcher}"
    exit 1
fi

export TRAIN_ENTRY="${entry}"

echo "[ablation3] w/o Refiner | encoder = mlp -> softmax pool -> final_proj|state (no bottleneck)"
echo "[ablation3] TDT kept: input = pool_task(mlp(view)), target = pool_task(mlp(obs|future)) stop-grad"
echo "[ablation3] loss = ${L2_CLEAN_WEIGHT:-1.0}*L2_clean + ${L2_POLL_WEIGHT:-0.3}*L2_poll + ${LAMBDA_JEPA_CLEAN:-1}*L1_clean + ${LAMBDA_JEPA_POLL:-1}*L1_poll"
echo "[ablation3] task=${task_name} tag=${tag} seed=${seed} epochs=${epochs} gpu=${gpu} batch=${BATCH:-128}"
echo "[ablation3] ckpt is refiner-free (strict load by v5 fails) -> eval: bash PI-JEPA/ablation/eval_ablation3_no_refiner.sh ${task_name} ${tag} ${seed} ${gpu}"

exec bash "${launcher}" \
     "${task_name}" "${tag}" "${seed}" "${gpu}" "${epochs}" ${zarr_arg:+"${zarr_arg}"}
