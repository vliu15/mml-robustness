#!/bin/bash
ckpts=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24" "last")
log_dir="./logs/mtl_attractive_smiling"
subgroup_attributes='{"Attractive":["Eyeglasses"],"Smiling":["High_Cheekbones"]}'

for ckpt in ${ckpts[@]}; do
    echo "ckpt:$ckpt"
    python ./test.py \
        --subgroup_labels \
        --ckpt_num $ckpt \
        --log_dir $log_dir \
        --subgroup_attributes $subgroup_attributes
done