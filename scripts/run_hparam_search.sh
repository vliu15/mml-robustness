#!/bin/bash

# Sample usage:
# bash scripts/run_hparam_search.sh

for wd in 1 .1 .01 .001 .0001; do
    for lr in .001 .0001; do
        python ./train_erm.py \
            exp=erm \
            exp.train.total_epochs=10 \
            exp.optimizer.lr=$lr \
            exp.optimizer.weight_decay=$wd \
            exp.train.log_dir="./logs/mtl_task_1_task_2/wd_$lr/lr_$lr"
    done
done
