"""
Script that runs spurious identification and saves the heatmap
All per-attribute JSON files are saved in {log_dir}/spurious/
and the heatmap is saved into {log_dir}/spurious_heatmap.png

Sample usage for running spurious idenfication inference:
python -m scripts.generate_spurious_matrix \
    --log_dir logs/erm/Blond_Hair:Male \
    --ckpt_num 20

Sample usage for loading intermediate jsons:
python -m scripts.generate_spurious_matrix \
    --json_dir logs/erm/Blond_Hair:Male/spurious
"""

import argparse
import json
import os
import re
import subprocess
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf

## for all files in the results of a given log dir, load them into dicts, put into pandas and then display it

attributes = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose",
    "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee",
    "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair",
    "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=False, default=None, help="Log directory from training")
    parser.add_argument(
        "--ckpt_num", type=int, required=False, default=None, help="Checkpoint number to run for spurious identification"
    )
    parser.add_argument(
        "--json_dir", type=str, required=False, default=None, help="Path to json directory dumped by this script"
    )
    args = parser.parse_args()

    assert (args.log_dir is not None and args.ckpt_num is not None) or (args.json_dir is not None)
    return args


def main():
    args = parse_args()

    # Load config
    config = OmegaConf.load(os.path.join(args.log_dir, "config.yaml"))
    assert len(config.dataset.groupings) == 1, \
        f"Use a single-task model for this script. Found groupings: {config.dataset.groupings}"

    # Prepare task x attr
    task_label = config.dataset.groupings[0].split(":")[0]
    attributes.remove(task_label)

    # Generate JSON files if not already present
    if args.json_dir is not None:
        # Set up save dir
        args.json_dir = os.path.join(args.log_dir, "spurious")
        os.makedirs(args.json_dir, exist_ok=True)

        # Call test.py 39 times
        for attr in attributes:
            save_json = os.path.join(args.json_dir, f"{task_label}:{attr}.json")
            command = (
                "python test.py "
                f"--log_dir {args.log_dir} "
                f"--ckpt_num {args.ckpt_num} "
                f"--groupings '[{task_label}:{attr}]' "
                f"--split test "
                f"--save_json {save_json}"
            )
            subprocess.run(command, shell=True, check=True)

    assert len(args.json_dir) == len(attributes), f"There should be {len(attributes)} JSON files"

    # Aggregate JSON files
    results_dict = defaultdict(list)
    for attr in attributes:
        with open(os.path.join(args.json_dir, f"{task_label}:{attr}.json"), "r") as f:
            data = json.load(f)
            for i in range(4):
                results_dict[f"Group {i} Acc"].append(round(100 * data[f"{task_label}_g{i}_acc"], 3))

    # Create PD dataframe
    results_df = pd.DataFrame.from_dict(results_dict)
    results_df = results_df.rename(index={ind: v for ind, v in enumerate(attributes)})

    # Create and save heatplot
    _, ax = plt.subplots(figsize=(39, 13))
    heatmap = sns.heatmap(results_df, annot=True, linewidths=.5, ax=ax)
    ax.set_title(f'Task Label: {task_label}')
    plt.xlabel("Subgroups")
    plt.ylabel("Potential Spurrious Correlates")
    heatmap.figure.savefig(os.path.join(args.log_dir, "spurious_heatmap.png"))


if __name__ == "__main__":
    main()
