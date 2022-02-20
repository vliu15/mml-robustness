"""
Script that runs spurious identification and saves the heatmap
All per-attribute JSON files are saved in {log_dir}/spurious/
and the heatmap is saved into {log_dir}/spurious_heatmap.png

Sample usage for running spurious idenfication inference:
python -m scripts.generate_spurious_matrix \
    --log_dir logs/erm/Blond_Hair:Male \
    --json_dir outputs/spurious_eval \
    --mode debug
"""

import argparse
import json
import logging
import logging.config
import os
import subprocess
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf

from scripts.find_best_ckpt import main as find_best_ckpt

logging.config.fileConfig("logger.conf")
logger = logging.getLogger(__name__)

sns.set(font="Arial")

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
    parser.add_argument("--log_dir", type=str, required=True, help="Log directory from training")
    parser.add_argument(
        "--json_dir",
        type=str,
        required=False,
        default="outputs/spurious_eval",
        help="Path to json directory dumped by this script"
    )

    parser.add_argument("--mode", type=str, choices=["debug", "shell"], default="debug", help="Spawn job mode")

    # No need to change any of these tbh
    parser.add_argument(
        "--template", type=str, default="scripts/sbatch_template.sh", required=False, help="SBATCH template file"
    )
    parser.add_argument("--slurm_logs", type=str, default="slurm_logs", required=False, help="Directory to output slurm logs")
    args = parser.parse_args()

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
    json_dir = os.path.join(args.json_dir, task_label)

    # Set up save dir
    os.makedirs(json_dir, exist_ok=True)

    # Find best checkpoint for average accuracy
    ckpt_num = find_best_ckpt(args.log_dir, run_test=False, test_groupings="", metric="avg")
    logger.info("Best validation checkpoint: %s", ckpt_num)

    # Call test.py 39 times. Generate JSON files if not already present
    for attr in attributes:
        save_json = os.path.join(json_dir, f"{task_label}:{attr}.json")
        if os.path.exists(save_json):
            continue
        command = (
            "python test.py "
            f"--log_dir {args.log_dir} "
            f"--ckpt_num {ckpt_num} "
            f"--groupings [\\\"{task_label}:{attr}\\\"] "
            f"--split val "
            f"--save_json {save_json}"
        )
        if args.mode == "debug":
            logger.info(command)
        else:
            subprocess.run(command, shell=True, check=True)

    # Some checks before we aggregate JSON files
    if args.mode == "debug":
        logger.info("Running in debug mode. Exiting program without creating heatmap")
        return
    assert len(list(filter(lambda f: f.endswith(".json"), os.listdir(json_dir)))) == len(attributes), \
        f"There should be {len(attributes)} JSON files"

    # Aggregate JSON files
    results_dict = defaultdict(list)
    for attr in attributes:
        with open(os.path.join(json_dir, f"{task_label}:{attr}.json"), "r") as f:
            data = json.load(f)
            for i in range(4):
                results_dict[f"Group {i}"].append(round(100 * data[f"{task_label}_g{i}_acc"], 4))

    # Create PD dataframe
    results_df = pd.DataFrame.from_dict(results_dict)
    results_df = results_df.rename(index={ind: v for ind, v in enumerate(attributes)})

    # Create and save heatplot
    _, ax = plt.subplots(figsize=(13, 13))
    heatmap = sns.heatmap(results_df, annot=True, fmt=".4f", linewidths=1.0, ax=ax)
    ax.set_title(f"Task Label: {task_label}", fontsize=18, pad=20)
    plt.xlabel("Subgroup Accuracy", labelpad=20, fontweight="bold")
    plt.ylabel("Potential Spurious Correlates", labelpad=20, fontweight="bold")
    plt.tight_layout()

    heatmap_file = os.path.join(json_dir, f"{task_label}_heatmap.png")
    heatmap.figure.savefig(heatmap_file)
    logger.info(f"Saved heatmap to {heatmap_file}")


if __name__ == "__main__":
    main()
