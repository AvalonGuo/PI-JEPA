# Collect Data
```
# Task: assembly, disassemble, stick-push
TASK=stick-push
bash scripts/gen_demonstration_metaworld.sh <task>
```
# Baseline
## Train&Eval
default param: epochs=1000, batch_size=128.
### A policy trained with one seed can only be evaluated on that same seed.
#### Train
```
TASK='assembly'
bash scripts/train_policy.sh dp3 metaworld_${TASK} 0112 <seed> 0 1000 /hy-tmp/3DP/data/metaworld_${TASK}_expert.zarr
```
#### Eval
```
# suite: false, catgen, posgen, dynsphere, dynhand
TASK=disassemble
for SUITE in false catgen posgen dynsphere dynhand; do
    bash scripts/eval_policy.sh dp3 metaworld_${TASK} 0112 0 0 false 50 "$SUITE" 15
done
```

