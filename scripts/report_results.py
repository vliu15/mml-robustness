"""
Handy script to report results from evaluation on validation or test set across a set of seeds for either stl
or mtl

Sample usage:
python -m scripts.report_results \
    --log_dirs logs/erm_.* \
    --split test \
    --checkpoint_metric_type avg
    --mtl_checkpoint_type average
"""

import argparse
import json
import logging
import logging.config
import os
import re
from collections import defaultdict
from copy import deepcopy
from typing import List

import numpy as np
from omegaconf import OmegaConf
from scipy import stats
from tqdm import tqdm

from datasets.celeba import CelebA

logging.config.fileConfig("logger.conf")
logger = logging.getLogger(__name__)

alpha = 0.05  ## Change this if something other than a 95% CI is desired
z = stats.norm.ppf(1 - alpha / 2)

###TODO:FIX THE PER-TASK ONE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dirs", type=str, required=True, help="Path to log directories with results")
    parser.add_argument(
        "--split", type=str, required=True, choices=["val", "test"], help="Type of metric to report results on"
    )
    parser.add_argument(
        "--checkpoint_metric_type",
        type=str,
        required=True,
        choices=["avg", "group"],
        help="Whether to choose avg or worst group checkpoint"
    )
    parser.add_argument(
        "--mtl_checkpoint_type",
        type=str,
        required=False,
        default=None,
        choices=["average", "best-worst", "per-task", None],
        help="Whether to choose checkpointing based on the average performance, best worst performance, or per-task"
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
def mean_std_results(
    *, exp_name: str, log_dirs: List[str], split: str, checkpoint_metric_type: str, mtl_checkpoint_type: str = None
):
    average_accuracy = defaultdict(list)
    worst_group_accuracy = defaultdict(list)

    for log_dir in tqdm(log_dirs, desc=f"EXPERIMENT [{exp_name}]"):

        file_name = f"{split}_stats_{checkpoint_metric_type}_checkpoint.json"
        if mtl_checkpoint_type is not None:
            file_name = f"{split}_stats_{checkpoint_metric_type}_checkpoint_{mtl_checkpoint_type}_mtl_type.json"
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

    if mtl_checkpoint_type is not None:
        logger.info(
            f"Split [{split}] | Checkpoint selection [{checkpoint_metric_type}] | MTL checkpoint type [{mtl_checkpoint_type}] :\n"
        )
    else:
        logger.info(f"Split [{split}] | Checkpoint selection [{checkpoint_metric_type}] :\n")

    for task in average_accuracy.keys():
        logger.info(f"\nTASK: {task}")
        logger.info(
            f"Mean average accuracy: {np.mean(average_accuracy[task])}, std:{np.std(average_accuracy[task])}, over {len(average_accuracy[task])} seeds"
        )
        logger.info(
            f"Mean worst-group accuracy: {np.mean(worst_group_accuracy[task])}, std:{np.std(worst_group_accuracy[task])}, over {len(worst_group_accuracy[task])} seeds\n"
        )


def mean_ci_results(
    *, exp_name: str, log_dirs: List[str], split: str, checkpoint_metric_type: str, mtl_checkpoint_type: str = None, dict_name: str=None
):
    average_counts = defaultdict(
        lambda: defaultdict(list)
    )  ## for each task need a correct counts and a total counts of average
    worst_group_counts = defaultdict(
        lambda: defaultdict(list)
    )  ## for each task need a correct counts and a total counts of worst group

    for log_dir in tqdm(log_dirs, desc=f"EXPERIMENT [{exp_name}]"):
        config = OmegaConf.load(os.path.join(log_dir, "config.yaml"))

        task_to_attributes = {}
        for grouping in config.dataset.groupings:
            task_attribute = grouping.split(":")
            task_to_attributes[task_attribute[0]] = task_attribute[1]

        file_name = f"{split}_stats_{checkpoint_metric_type}_checkpoint.json"
        if mtl_checkpoint_type is not None:
            file_name = f"{split}_stats_{checkpoint_metric_type}_checkpoint_{mtl_checkpoint_type}_mtl_type.json"
        file_path = os.path.join(log_dir, 'results', file_name)

        with open(file_path, "r") as f:
            results = json.load(f)

        for task, attribute in task_to_attributes.items():
            ### get group total counts
            group_sizes = get_group_sizes(config, task, attribute, split)

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

    if mtl_checkpoint_type is not None:
        logger.info(
            f"Split [{split}] | Checkpoint selection [{checkpoint_metric_type}] | MTL checkpoint type [{mtl_checkpoint_type}] :\n"
        )
    else:
        logger.info(f"Split [{split}] | Checkpoint selection [{checkpoint_metric_type}] :\n")

    save_dict = {}
    for task in average_counts.keys():

        logger.info(f"TASK [{task}]:")

        total_avg_counts = np.sum(average_counts[task][f"{task}_total_counts"])
        total_avg_correct_counts = np.sum(average_counts[task][f"{task}_correct_counts"])

        total_worst_group_counts = np.sum(worst_group_counts[task][f"{task}_total_counts"])
        total_worst_group_correct_counts = np.sum(worst_group_counts[task][f"{task}_correct_counts"])

        n_tilde_avg = total_avg_counts + z**2
        p_tilde_avg = (1 / n_tilde_avg) * (total_avg_correct_counts + ((z**2) / 2))
        ci_range_avg = z * np.sqrt((p_tilde_avg / n_tilde_avg) * (1 - p_tilde_avg))

        logger.info(
            f"Estimated mean average accuracy: {round(p_tilde_avg*100,2)}, with 95% CI:({round((p_tilde_avg - ci_range_avg)*100, 2)},{round( (p_tilde_avg + ci_range_avg)*100, 2)}), over {len(average_counts[task][f'{task}_correct_counts'])} seeds"
        )

        n_tilde_group = total_worst_group_counts + z**2
        p_tilde_group = (1 / n_tilde_group) * (total_worst_group_correct_counts + ((z**2) / 2))
        ci_range_group = z * np.sqrt((p_tilde_group / n_tilde_group) * (1 - p_tilde_group))

        logger.info(
            f"Estimated mean worst-group accuracy: {round(p_tilde_group*100,2)}, with 95% CI:({round((p_tilde_group - ci_range_group)*100, 2)},{round( (p_tilde_group + ci_range_group)*100, 2)}), over {len(worst_group_counts[task][f'{task}_correct_counts'])} seeds\n"
        )

        key_name = f"{task}:{task_to_attributes[task]}"
        save_dict[key_name] = [(round(p_tilde_avg*100,2),(round((p_tilde_avg - ci_range_avg)*100, 2), round( (p_tilde_avg + ci_range_avg)*100, 2))), (round(p_tilde_group*100,2), (round((p_tilde_group - ci_range_group)*100, 2),round( (p_tilde_group + ci_range_group)*100, 2) ))]

    dict_name = dict_name.split("/")[-1]
    save_name = f"{dict_name}_checkpoint_selection_{checkpoint_metric_type}_mtl_checkpoint_type_{mtl_checkpoint_type}.json"
    with open(os.path.join("./iclr_submission", "cached_results", save_name), "w") as f:
            json.dump(save_dict, f)


def main():
    args = parse_args()

    log_dir_regex = re.compile(args.log_dirs)
    log_dirs = []

    # [1] Allow for nested regex patterns (i.e. logs/mtl_.*/mtl_train:rwy.*)
    logger.info("Extracting all folders from $PWD that match the regex ...")
    for root, folders, _ in os.walk(os.getcwd()):
        for folder in folders:
            folder = os.path.join(root, folder)

            # We only want log directories, which are guaranteed to have a specific format (i.e. config.yaml)
            if "config.yaml" not in os.listdir(folder):
                continue

            # Once we get a valid log directory path, check for match
            if log_dir_regex.match(os.path.abspath(folder)) or log_dir_regex.match(os.path.relpath(folder)):
                log_dirs.append(folder)

    logger.info("Found %d log dirs that match the regex `%s`", len(log_dirs), args.log_dirs)
    exp_to_log_dirs = defaultdict(list)

    # [2] Group by identical folder names except seeds
    logger.info("Grouping folders by names across different seeds\n")
    for log_dir in log_dirs:
        key = re.sub("seed:[0-9]+", "seed:#", log_dir)  # substitute seed:# to create -> exp name
        key = os.path.relpath(key)
        exp_to_log_dirs[key].append(log_dir)

    # [3] Compute CI
    for exp_name, log_dirs in exp_to_log_dirs.items():
        mean_ci_results(
            exp_name=exp_name,
            log_dirs=log_dirs,
            split=args.split,
            checkpoint_metric_type=args.checkpoint_metric_type,
            mtl_checkpoint_type=args.mtl_checkpoint_type,
            dict_name = args.log_dirs,
        )


if __name__ == "__main__":
    main()
