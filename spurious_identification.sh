#!/bin/bash
attributes=("5_o_Clock_Shadow" "Arched_Eyebrows" "Attractive" "Bags_Under_Eyes" "Bald" "Bangs" "Big_Lips" "Big_Nose" "Black_Hair" "Blond_Hair" "Blurry" "Brown_Hair" "Bushy_Eyebrows" "Chubby" "Double_Chin" "Eyeglasses" "Goatee" "Gray_Hair" "Heavy_Makeup" "High_Cheekbones" "Male" "Mouth_Slightly_Open" "Mustache" "Narrow_Eyes" "No_Beard" "Oval_Face" "Pale_Skin" "Pointy_Nose" "Receding_Hairline" "Rosy_Cheeks" "Sideburns" "Smiling" "Straight_Hair" "Wavy_Hair" "Wearing_Earrings" "Wearing_Hat" "Wearing_Lipstick" "Wearing_Necklace" "Wearing_Necktie" "Young")
curr_task="Bushy_Eyebrows" 
log_dir="./logs/erm_bushy_eyebrows"
ckpt=15

## reemove curr_task from attributes 
attributes=("${attributes[@]/$curr_task}")

for correlate in ${attributes[@]}; do
    python ./test.py \
        --subgroup_labels \
        --ckpt_num $ckpt \
        --log_dir $log_dir \
        --subgroup_attributes '{'"\"$curr_task\""': ['"\"$correlate\""']}'
done