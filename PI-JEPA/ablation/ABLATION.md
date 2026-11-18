## Ablation1
cd /root/3DP || exit 1
TASK=metaworld_assembly
SEED=2
LOGDIR=/hy-tmp/data/ablation1/logs
mkdir -p "${LOGDIR}"

ROLLOUT_EVERY=100 PYTHONUNBUFFERED=1 nohup bash PI-JEPA/ablation/train_ablation1_single_view.sh \
    ${TASK} "${SEED}" 0 ab1 1000 \
    > "${LOGDIR}/ablation1_${TASK}_seed${SEED}.log" 2>&1 &
echo $! > "${LOGDIR}/ablation1_${TASK}_seed${SEED}.pid"
cat "${LOGDIR}/ablation1_${TASK}_seed${SEED}.pid"


SEED=2
TASK=metaworld_assembly
for suite in false catgen posgen dynsphere dynhand; do
    echo "======${suite}======"
    CKPT=best bash PI-JEPA/ablation/eval_ablation1_single_view.sh ${TASK} ab1 ${SEED} 0 false 50 ${suite} 15
done


## Ablation2 (no JEPA L1, two-view L2 kept)
cd /root/3DP || exit 1
TASK=metaworld_assembly
SEED=2
LOGDIR=/hy-tmp/data/ablation2/logs
mkdir -p "${LOGDIR}"

ROLLOUT_EVERY=100 PYTHONUNBUFFERED=1 nohup bash PI-JEPA/ablation/train_ablation2_no_l1.sh \
    ${TASK} "${SEED}" 0 ab2 1000 \
    > "${LOGDIR}/ablation2_${TASK}_seed${SEED}.log" 2>&1 &
echo $! > "${LOGDIR}/ablation2_${TASK}_seed${SEED}.pid"
cat "${LOGDIR}/ablation2_${TASK}_seed${SEED}.pid"


SEED=2
TASK=metaworld_assembly
for suite in false catgen posgen dynsphere dynhand; do
    echo "======${suite}======"
    CKPT=best bash PI-JEPA/ablation/eval_ablation2_no_l1.sh ${TASK} ab2 ${SEED} 0 false 50 ${suite} 15
done
## Ablation3 (w/o Refiner: PIJEPA-BottleNeck removed, TDT kept)
# structural change -> the ckpt has no obs_encoder.bottleneck.* keys and is NOT
# loadable by v5; eval MUST go through eval_ablation3_no_refiner.sh (it sets V5_TARGET)
cd /root/3DP || exit 1
TASK=metaworld_assembly
SEED=2
LOGDIR=/hy-tmp/data/ablation3/logs
mkdir -p "${LOGDIR}"

ROLLOUT_EVERY=100 PYTHONUNBUFFERED=1 nohup bash PI-JEPA/ablation/train_ablation3_no_refiner.sh \
    ${TASK} "${SEED}" 0 ab3 1000 \
    > "${LOGDIR}/ablation3_${TASK}_seed${SEED}.log" 2>&1 &
echo $! > "${LOGDIR}/ablation3_${TASK}_seed${SEED}.pid"
cat "${LOGDIR}/ablation3_${TASK}_seed${SEED}.pid"


SEED=2
TASK=metaworld_assembly
for suite in false catgen posgen dynsphere dynhand; do
    echo "======${suite}======"
    CKPT=best bash PI-JEPA/ablation/eval_ablation3_no_refiner.sh ${TASK} ab3 ${SEED} 0 false 50 ${suite} 15
done


