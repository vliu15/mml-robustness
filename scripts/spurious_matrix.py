"""
Script that runs spurious identification and saves the heatmap
All per-attribute JSON files and PNG heatmaps are saved in {json_dir}/$TASK

Sample usage for running spurious idenfication inference:
python -m scripts.spurious_matrix \
    --log_dir logs/erm/Blond_Hair:Male \
    --json_dir outputs/spurious_eval \
    --mode debug
"""

import argparse
import json
import logging
import logging.config
import os
import re
import subprocess
from collections import defaultdict
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf
from scipy import stats
from tqdm import tqdm

from datasets.celeba import CelebA
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

alpha = 0.05  ## Change this if something other than a 95% CI is desired
z = stats.norm.ppf(1 - alpha / 2)


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
    return parser.parse_args()


def get_group_sizes(config, task, attr):
    config = deepcopy(config)
    config.dataset.subgroup_labels = True
    config.dataset.groupings = [f"{task}:{attr}"]

    dataset = CelebA(config, split="val")
    counts = dataset.counts.squeeze(0).tolist()

    # In the event that there are subgroups with size 0, there are two cases:
    #   1. if len(counts) == 4: then these 0-count groups are not group 3
    #   2. if len(counts) < 4 : then these 0-count groups are at least group 3, 2, 1, 0, etc
    # Since we count subgroup sizes with torch.bincount, the only time we end up with <4
    # subgroups is when the instances are missing in reverse order (3, 2, 1, 0)
    counts = counts + [0] * (4 - len(counts))
    return counts


def create_group_acc_heatmap(group_acc_dict, json_dir, task_label, class_accuracies):
    # Create PD dataframe
    group_acc_df = pd.DataFrame.from_dict(group_acc_dict)
    group_acc_df = group_acc_df.rename(index={ind: v for ind, v in enumerate(attributes)})

    # Create and save heatplot
    fig, ax = plt.subplots(figsize=(13, 13))
    heatmap = sns.heatmap(group_acc_df, annot=True, fmt=".4f", linewidths=1.0, ax=ax, vmin=0, vmax=100)
    fig.suptitle(f"Task Label: {task_label}", fontsize=18)
    ax.set_title(
        f"Class 0 Acc: {round(100 * class_accuracies[0],4)}, Class 1 Acc: {round(100 * class_accuracies[1],4)}",
        fontsize=14,
        pad=15
    )
    plt.xlabel("Subgroup Accuracy", labelpad=20, fontweight="bold")
    plt.ylabel("Potential Spurious Correlates", labelpad=20, fontweight="bold")
    plt.tight_layout()
    heatmap_file = os.path.join(json_dir, f"{task_label}_heatmap_group_acc.png")
    heatmap.figure.savefig(heatmap_file)
    logger.info(f"Saved group accuracies heatmap to {heatmap_file}")


def create_group_size_heatmap(group_size_dict, json_dir, task_label):
    # Create PD dataframe
    group_size_df = pd.DataFrame.from_dict(group_size_dict)
    group_size_df = group_size_df.rename(index={ind: v for ind, v in enumerate(attributes)})

    # Create and save heatplot
    _, ax = plt.subplots(figsize=(13, 13))
    heatmap = sns.heatmap(group_size_df, annot=True, fmt="d", linewidths=1.0, ax=ax, vmin=0, vmax=19867)
    ax.set_title(f"Task Label: {task_label}", fontsize=18, pad=20)
    plt.xlabel("Subgroup Size", labelpad=20, fontweight="bold")
    plt.ylabel("Potential Spurious Correlates", labelpad=20, fontweight="bold")
    plt.tight_layout()
    heatmap_file = os.path.join(json_dir, f"{task_label}_heatmap_group_size.png")
    heatmap.figure.savefig(heatmap_file)
    logger.info(f"Saved group sizes heatmap to {heatmap_file}")


def create_spurious_eval_heatmap(group_acc_dict, group_size_dict, avg_task_acc, json_dir, task_label):
    # Compute spurious eval
    spurious_eval_list = []
    for i in range(len(attributes)):
        group_accs = np.array([group_acc_dict[f"Group {j}"][i] for j in range(4)])

        delta = np.nan if np.isnan(group_accs).any() else abs(group_accs[0] + group_accs[3] - group_accs[1] - group_accs[2])
        spurious_eval_list.append(delta)

    spurious_eval_df = pd.DataFrame({f"Average Accuracy: {avg_task_acc:.4f}": spurious_eval_list}, index=attributes)
    _, ax = plt.subplots(figsize=(6, 13))
    heatmap = sns.heatmap(spurious_eval_df, annot=True, fmt=".4f", linewidths=1.0, ax=ax, vmin=0, vmax=100)
    ax.set_title(f"Task Label: {task_label}", fontsize=18, pad=20)
    plt.xlabel("Spurious Correlate $\delta = |g_0 + g_3 - g_1 - g_2|$", labelpad=20, fontweight="bold")
    plt.ylabel("Potential Spurious Correlates", labelpad=20, fontweight="bold")
    plt.tight_layout()

    ### save spurious_eval_list with attribute information
    spurious_eval_list = [None if np.isnan(delta) else delta for delta in spurious_eval_list]
    spurious_eval_dict = {attr: delta for attr, delta in zip(attributes, spurious_eval_list)}
    spurious_eval_file = os.path.join(json_dir, f"{task_label}_spurious_eval.json")
    f = open(spurious_eval_file, "w")
    json.dump(spurious_eval_dict, f)
    f.close()

    heatmap_file = os.path.join(json_dir, f"{task_label}_heatmap_spurious_eval.png")
    heatmap.figure.savefig(heatmap_file)
    logger.info(f"Saved group sizes heatmap to {heatmap_file}")


def main():
    args = parse_args()

    # Load config
    config = OmegaConf.load(os.path.join(args.log_dir, "config.yaml"))
    assert len(config.dataset.groupings) == 1, \
        f"Use a single-task model for this script. Found groupings: {config.dataset.groupings}"

    # Prepare task x attr
    task_label = config.dataset.groupings[0].split(":")[0]
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
        subprocess.run(command, shell=True, check=True)

    # Some checks before we aggregate JSON files
    spurious_regex = re.compile(fr"{task_label}:.*\.json")
    assert len(list(filter(lambda f: spurious_regex.match(f), os.listdir(json_dir)))) == len(attributes), \
        f"There should be {len(attributes)} JSON files"

    class_accuracies = None

    # Aggregate JSON files and compute the Agresti-Coull Interval for a 95% CI
    group_acc_dict = defaultdict(list)
    group_size_dict = defaultdict(list)
    avg_task_acc = None
    for attr in tqdm(attributes, desc="Aggregating test.py JSON files", total=len(attributes)):
        group_sizes = get_group_sizes(config, task_label, attr)

        with open(os.path.join(json_dir, f"{task_label}:{attr}.json"), "r") as f:
            data = json.load(f)
            avg_task_acc = round(100 * data[f"{task_label}_avg_acc"], 4)  # should be the same for all attributes

            if class_accuracies is None:
                class_zero_total = float(group_sizes[0] + group_sizes[1])
                class_zero_correct = float(data[f"{task_label}_g0_correct_counts"] + data[f"{task_label}_g1_correct_counts"])

                class_one_total = float(group_sizes[2] + group_sizes[3])
                class_one_correct = float(data[f"{task_label}_g2_correct_counts"] + data[f"{task_label}_g3_correct_counts"])
                class_accuracies = [(class_zero_correct / class_zero_total), (class_one_correct / class_one_total)]

            for i in range(4):

                class_accuracy = class_accuracies[int(i / 2)]
                group_size = float(group_sizes[i])
                group_correct_counts = float(data[f"{task_label}_g{i}_correct_counts"])
                n_tilde = group_size + z**2
                p_tilde = (1 / n_tilde) * (group_correct_counts + ((z**2) / 2))
                ci_range = z * np.sqrt((p_tilde / n_tilde) * (1 - p_tilde))
                lower_ci = p_tilde - ci_range
                upper_ci = p_tilde + ci_range

                if class_accuracy >= lower_ci and class_accuracy <= upper_ci:
                    group_acc_dict[f"Group {i}"].append(100 * round(class_accuracy, 4))
                else:
                    ## upper is closer
                    if np.abs(class_accuracy - lower_ci) > np.abs(upper_ci - class_accuracy):
                        group_acc_dict[f"Group {i}"].append(100 * round(upper_ci, 4))
                    else:
                        group_acc_dict[f"Group {i}"].append(100 * round(lower_ci, 4))

                group_size_dict[f"Group {i}"].append(group_sizes[i])

    # Create and save heatmaps
    create_group_acc_heatmap(group_acc_dict, json_dir, task_label, class_accuracies)
    create_group_size_heatmap(group_size_dict, json_dir, task_label)
    create_spurious_eval_heatmap(group_acc_dict, group_size_dict, avg_task_acc, json_dir, task_label)


if __name__ == "__main__":
    main()
