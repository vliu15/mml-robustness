"""
Handy script to report results from evaluation on validation or test set across a set of seeds for either stl
or mtl

Sample usage:
python -m scripts.report_results \
    --log_dirs ./logs/erm_seed_0 ./logs/erm_seed_1 ./logs/erm_seed_2 \
    --split test \
    --checkpoint_type avg
"""

import argparse
import json
import logging
import logging.config
import os
import re
from collections import defaultdict
from copy import deepcopy

import numpy as np
from omegaconf import OmegaConf
from scipy import stats
from tqdm import tqdm

from datasets.celeba import CelebA

logging.config.fileConfig("logger.conf")
logger = logging.getLogger(__name__)

alpha = 0.05  ## Change this if something other than a 95% CI is desired
z = stats.norm.ppf(1 - alpha / 2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dirs", type=str, nargs='+', required=True, help="Path to log directories with results")
    parser.add_argument(
        "--split", type=str, required=True, choices=["val", "test"], help="Type of metric to report results on"
    )
    parser.add_argument(
        "--checkpoint_type",
        type=str,
        required=True,
        choices=["avg", "group"],
        help="Whether to choose avg or worst group checkpoint"
    )
    return parser.parse_args()


def get_group_sizes(config, task, attr, split):
    config = deepcopy(config)
    config.dataset.subgroup_labels = True
    config.dataset.groupings = [f"{task}:{attr}"]

    dataset = CelebA(config, split=split)
    counts = dataset.counts.squeeze(0).tolist()

    # In the event that there are subgroups with size 0, there are two cases:
    #   1. if len(counts) == 4: then these 0-count groups are not group 3
    #   2. if len(counts) < 4 : then these 0-count groups are at least group 3, 2, 1, 0, etc
    # Since we count subgroup sizes with torch.bincount, the only time we end up with <4
    # subgroups is when the instances are missing in reverse order (3, 2, 1, 0)
    counts = counts + [0] * (4 - len(counts))
    return counts


### print out average accuracy, print out worst group accuracy print out loss
def mean_std_results():
    args = parse_args()

    average_accuracy = defaultdict(list)
    worst_group_accuracy = defaultdict(list)
    for log_dir in args.log_dirs:

        file_name = f"{args.split}_stats_{args.checkpoint_type}_checkpoint.json"
        file_path = os.path.join(log_dir, 'results', file_name)

        with open(file_path, "r") as f:
            results = json.load(f)

        group_acc_key_regex = re.compile(r".*_g[0-9]+_acc")
        avg_acc_key_regex = re.compile(r".*_avg_acc")

        group_accuracies = {key: results[key] for key in results.keys() if group_acc_key_regex.match(key)}
        worst_group_accuracies = {}
        for key in group_accuracies.keys():
            group_acc_key_regex = r"_g[0-9]+_acc"
            sub_key = re.split(group_acc_key_regex, key)[0]

            if sub_key in worst_group_accuracies:
                curr_val = worst_group_accuracies[sub_key]
                worst_group_accuracies[sub_key] = min(curr_val, group_accuracies[key])
            else:
                worst_group_accuracies[sub_key] = group_accuracies[key]

        avg_accuracies = {key.split("_avg_acc")[0]: results[key] for key in results.keys() if avg_acc_key_regex.match(key)}

        for task, worst_group_acc in worst_group_accuracies.items():
            worst_group_accuracy[task].append(worst_group_acc)

        for task, avg_acc in avg_accuracies.items():
            average_accuracy[task].append(avg_acc)

    logger.info(f"For split: {args.split}, using checkpoints based on: {args.checkpoint_type} we obtain: \n")

    for task in average_accuracy.keys():

        logger.info(f"For TASK: {task}")
        logger.info(
            f"Mean average accuracy: {np.mean(average_accuracy[task])}, std:{np.std(average_accuracy[task])}, over {len(average_accuracy[task])} seeds \n"
        )
        logger.info(
            f"Mean worst-group accuracy: {np.mean(worst_group_accuracy[task])}, std:{np.std(worst_group_accuracy[task])}, over {len(worst_group_accuracy[task])} seeds \n"
        )


def mean_ci_results():
    args = parse_args()

    average_counts = defaultdict(
        lambda: defaultdict(list)
    )  ## for each task need a correct counts and a total counts of average
    worst_group_counts = defaultdict(
        lambda: defaultdict(list)
    )  ## for each task need a correct counts and a total counts of worst group

    for log_dir in tqdm(args.log_dirs):

        config = OmegaConf.load(os.path.join(log_dir, "config.yaml"))

        task_to_attributes = {}
        for grouping in config.dataset.groupings:
            task_attribute = grouping.split(":")
            task_to_attributes[task_attribute[0]] = task_attribute[1]

        file_name = f"{args.split}_stats_{args.checkpoint_type}_checkpoint.json"
        file_path = os.path.join(log_dir, 'results', file_name)

        with open(file_path, "r") as f:
            results = json.load(f)

        for task, attribute in task_to_attributes.items():
            ### get group total counts
            group_sizes = get_group_sizes(config, task, attribute, args.split)

            worst_group_acc = float("inf")
            worst_group_correct = 0
            worst_group_total = 0

            avg_correct = 0
            avg_total = 0

            for i in range(4):

                group_size = float(group_sizes[i])
                group_correct_counts = float(results[f"{task}_g{i}_correct_counts"])
                group_acc = float(results[f"{task}_g{i}_acc"])

                avg_correct += group_correct_counts
                avg_total += group_size

                if group_acc < worst_group_acc:
                    worst_group_acc = group_acc
                    worst_group_correct = group_correct_counts
                    worst_group_total = group_size

            average_counts[task][f"{task}_total_counts"].append(avg_total)
            average_counts[task][f"{task}_correct_counts"].append(avg_correct)

            worst_group_counts[task][f"{task}_total_counts"].append(worst_group_total)
            worst_group_counts[task][f"{task}_correct_counts"].append(worst_group_correct)

    logger.info(f"For split: {args.split}, using checkpoints based on: {args.checkpoint_type} we obtain: \n")

    for task in average_counts.keys():

        logger.info(f"For TASK: {task}")

        total_avg_counts = np.sum(average_counts[task][f"{task}_total_counts"])
        total_avg_correct_counts = np.sum(average_counts[task][f"{task}_correct_counts"])

        total_worst_group_counts = np.sum(worst_group_counts[task][f"{task}_total_counts"])
        total_worst_group_correct_counts = np.sum(worst_group_counts[task][f"{task}_correct_counts"])

        n_tilde_avg = total_avg_counts + z**2
        p_tilde_avg = (1 / n_tilde_avg) * (total_avg_correct_counts + ((z**2) / 2))
        ci_range_avg = z * np.sqrt((p_tilde_avg / n_tilde_avg) * (1 - p_tilde_avg))

        logger.info(
            f"Estimated mean average accuracy: {round(p_tilde_avg,4)}, with 95% CI:({round(p_tilde_avg - ci_range_avg, 4)},{round(p_tilde_avg + ci_range_avg, 4)}), over {len(average_counts[task][f'{task}_correct_counts'])} seeds \n"
        )

        n_tilde_group = total_worst_group_counts + z**2
        p_tilde_group = (1 / n_tilde_group) * (total_worst_group_correct_counts + ((z**2) / 2))
        ci_range_group = z * np.sqrt((p_tilde_group / n_tilde_group) * (1 - p_tilde_group))

        logger.info(
            f"Estimated mean worst-group accuracy: {round(p_tilde_group,4)}, with 95% CI:({round(p_tilde_group - ci_range_group, 4)},{round(p_tilde_group + ci_range_group, 4)}), over {len(worst_group_counts[task][f'{task}_correct_counts'])} seeds \n"
        )


if __name__ == "__main__":
    mean_ci_results()
