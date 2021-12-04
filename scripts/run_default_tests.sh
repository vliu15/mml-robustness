#!/bin/bash

# #################################################
# For reference, here are groupings that belong
# together based on same task label during training
#
# Blond_Hair  : [Male]
# Attractive  : [Eyeglasses,Bald,Heavy_Makeup]
# Smiling     : [High_Cheekbones]
# Young       : [Attractive,Gray_Hair]
# Oval_Face   : [Rosy_Cheeks]
# Pointy_Nose : [Male,Rosy_Cheeks]
# #################################################

# Run on all default groupings
ckpts=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24" "last")
for ckpt in ${ckpts[@]}; do
    python test.py --log_dir logs/jtt-Blond_Hair/stage_2 --ckpt_num $ckpt --subgroup_attributes "[\"Blond_Hair:Male\"]" --json_name "Blond_Hair:Male.$ckpt.test_results.json"

    python test.py --log_dir logs/jtt-Attractive/stage_2 --ckpt_num $ckpt --subgroup_attributes "[\"Attractive:Eyeglasses\"]" --json_name "Attractive:Eyeglasses.$ckpt.test_results.json"
    python test.py --log_dir logs/jtt-Attractive/stage_2 --ckpt_num $ckpt --subgroup_attributes "[\"Attractive:Bald\"]" --json_name "Attractive:Bald.$ckpt.test_results.json"
    python test.py --log_dir logs/jtt-Attractive/stage_2 --ckpt_num $ckpt --subgroup_attributes "[\"Attractive:Heavy_Makeup\"]" --json_name "Attractive:Heavy_Makeup.$ckpt.test_results.json"

    python test.py --log_dir logs/jtt-Smiling/stage_2 --ckpt_num $ckpt --subgroup_attributes "[\"Smiling:High_Cheekbones\"]" --json_name "Smiling:High_Cheekbones.$ckpt.test_results.json"

    python test.py --log_dir logs/jtt-Young/stage_2 --ckpt_num $ckpt --subgroup_attributes "[\"Young:Attractive\"]" --json_name "Young:Attractive.$ckpt.test_results.json"
    python test.py --log_dir logs/jtt-Young/stage_2 --ckpt_num $ckpt --subgroup_attributes "[\"Young:Gray_Hair\"]" --json_name "Young:Gray_Hair.$ckpt.test_results.json"

    python test.py --log_dir logs/jtt-Oval_Face/stage_2 --ckpt_num $ckpt --subgroup_attributes "[\"Oval_Face:Rosy_Cheeks\"]" --json_name "Oval_Face:Rosy_Cheeks.$ckpt.test_results.json"

    python test.py --log_dir logs/jtt-Pointy_Nose/stage_2 --ckpt_num $ckpt --subgroup_attributes "[\"Pointy_Nose:Rosy_Cheeks\"]" --json_name "Pointy_Nose:Rosy_Cheeks.$ckpt.test_results.json"
    python test.py --log_dir logs/jtt-Pointy_Nose/stage_2 --ckpt_num $ckpt --subgroup_attributes "[\"Pointy_Nose:Rosy_Cheeks\"]" --json_name "Pointy_Nose:Rosy_Cheeks.$ckpt.test_results.json"
done
