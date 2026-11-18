# Baseline
## Collect Data
```
# Task: hammer door pen
TASK=door
bash scripts/gen_demonstration_adroit.sh ${TASK}          
```
## Train&Eval
default param: epochs=1000, batch_size=128.
### A policy trained with one seed can only be evaluated on that same seed.
#### Train
```
TASK=adroit_hammer
bash scripts/train_policy.sh dp3 ${TASK} 0112 <seed> 0 1000 /hy-tmp/3DP/data/${TASK}_expert.zarr
```
#### Eval
```
TASK=hammer
for SUITE in false catgen posgen dynsphere dynhand; do
    CKPT=best bash scripts/eval_policy.sh dp3 adroit_${TASK} 0112 0 0 false 50 "$SUITE" 15
done
```


