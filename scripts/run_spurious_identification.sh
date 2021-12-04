#!/bin/bash

# Sample usage:
# bash scripts/run_spurious_identification.sh $TASK $LOG_DIR $CKPT_NUM

attributes=("5_o_Clock_Shadow" "Arched_Eyebrows" "Attractive" "Bags_Under_Eyes" "Bald" "Bangs" "Big_Lips" "Big_Nose" "Black_Hair" "Blond_Hair" "Blurry" "Brown_Hair" "Bushy_Eyebrows" "Chubby" "Double_Chin" "Eyeglasses" "Goatee" "Gray_Hair" "Heavy_Makeup" "High_Cheekbones" "Male" "Mouth_Slightly_Open" "Mustache" "Narrow_Eyes" "No_Beard" "Oval_Face" "Pale_Skin" "Pointy_Nose" "Receding_Hairline" "Rosy_Cheeks" "Sideburns" "Smiling" "Straight_Hair" "Wavy_Hair" "Wearing_Earrings" "Wearing_Hat" "Wearing_Lipstick" "Wearing_Necklace" "Wearing_Necktie" "Young")

TASK=$1
LOG_DIR=$2
CKPT_NUM=$3

## remove curr_task from attributes 
attributes=("${attributes[@]/$TASK}")

for correlate in ${attributes[@]}; do
    python ./test.py \
        --subgroup_labels \
        --ckpt_num $CKPT_NUM \
        --log_dir $LOG_DIR \
        --subgroup_attributes "[\"$TASK:$correlate\"]" \
        --json_name "$TASK:${correlate}_test_results.json"
done
