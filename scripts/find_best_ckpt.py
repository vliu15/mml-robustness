"""
Iterates over the saved val stats saved from training and finds the best worst-group checkpoint

Loads the val_stats.json from training of the following format:
val_stats = {
    epoch: {
        "loss"               : ...,
        "${task}_loss"       : ...,
        "${task}_avg_acc"    : ...,
        "${task}_g${id}_acc" : ...,
    }
}

Sample usage:
python -m scripts.find_best_ckpt \
    --log_dir logs/erm/Blond_Hair:Male \
    --run_test --test_groupings [\"Blond_Hair:Male\"]
"""

import argparse
import json
import logging
import logging.config
import os
import re
import subprocess
import numpy as np

logging.config.fileConfig("logger.conf")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True, help="Path to log directory with val_stats.json")
    parser.add_argument(
        "--run_test", default=False, action="store_true", help="Whether to evaluate best checkpoint on test set"
    )
    parser.add_argument(
        "--test_groupings",
        type=str,
        required=False,
        default="",
        help="JSON-string list of {{task}}:{{subgroup}} to run additional evaluation on"
    )
    parser.add_argument(
        "--metric", type=str, required=True, choices=["group", "avg"], help="Type of metric/accuracy to compare checkpoints"
    )
    parser.add_argument(
        "--learning_type",
        type=str,
        required=True,
        choices=["stl", "mtl"],
        help="Whether we are evaluating a single or multi task learning appraoch"
    )

    parser.add_argument("--save_json", type=str, required=False, default="", help="Name for where file will be saved")
    return parser.parse_args()


def main(log_dir, run_test=False, test_groupings="", metric="avg", learning_type="stl", save_json=""):
    results_dir = os.path.join(log_dir, "results")

    val_stats_json_regex = re.compile(r"val_stats_[0-9]+\.json")
    val_stats = {}
    for results_file in sorted(os.listdir(results_dir)):
        if val_stats_json_regex.match(results_file):
            epoch = int(results_file.split(".")[0].split("_")[-1])
            if epoch in val_stats:
                logger.info(f"Warning: detected a duplicate val_stats json at epoch {epoch}")
            with open(os.path.join(results_dir, results_file), "r") as f:
                val_stats[epoch] = json.load(f)

    # Sort dict by epoch for readability
    val_stats = {k: val_stats[k] for k in sorted(val_stats.keys())}
    logger.info("Found the following epochs with validation results: %s", list(map(int, val_stats.keys())))

    # Find epoch with best worst-group accuracy
    best_epoch, best_acc = None, 0.0
    group_acc_key_regex = re.compile(r".*_g[0-9]+_acc")
    avg_acc_key_regex = re.compile(r".*_avg_acc")
    for epoch in val_stats.keys():
        if metric == "group":
            if learning_type == "stl":
                worst_group_acc = min(
                    val_stats[epoch][key] for key in val_stats[epoch].keys() if group_acc_key_regex.match(key)
                )
                if worst_group_acc > best_acc:
                    best_epoch = epoch
                    best_acc = worst_group_acc
            ## currently we define best checkpoint based on best average worst group accuracy across tasks
            elif learning_type == "mtl":
                group_accuracies = {
                    key: val_stats[epoch][key] for key in val_stats[epoch].keys() if group_acc_key_regex.match(key)
                }
                worst_group_accuracies = {}
                for key in group_accuracies.keys():
                    group_acc_key_regex_second = r"_g[0-9]+_acc"
                    sub_key = re.split(group_acc_key_regex_second, key)[0]

                    if sub_key in worst_group_accuracies:
                        curr_val = worst_group_accuracies[sub_key]
                        worst_group_accuracies[sub_key] = min(curr_val, group_accuracies[key])
                    else:
                        worst_group_accuracies[sub_key] = group_accuracies[key]

                worst_group_average_acc = sum(worst_group_accuracies.values()) / len(worst_group_accuracies)

                if worst_group_average_acc > best_acc:
                    best_epoch = epoch
                    best_acc = worst_group_average_acc

        elif metric == "avg":
            if learning_type == "stl":
                avg_group_acc = min(val_stats[epoch][key] for key in val_stats[epoch].keys() if avg_acc_key_regex.match(key))
                if avg_group_acc > best_acc:
                    best_epoch = epoch
                    best_acc = avg_group_acc
            ## currently we define best checkpoint based on best average average accuracy across tasks
            elif learning_type == "mtl":
                avg_task_acc = np.mean(
                    [val_stats[epoch][key] for key in val_stats[epoch].keys() if avg_acc_key_regex.match(key)]
                )
                if avg_task_acc > best_acc:
                    best_epoch = epoch
                    best_acc = avg_task_acc

        else:
            raise ValueError("Incorrect metric format. Only supports 'group' and 'acc'. ")

    best_epoch += 1
    logger.info("Best validation epoch: %s", best_epoch)
    logger.info("Best %s accuracy: %s", metric, best_acc)

    if best_epoch is None:
        logger.info("Detected an empty val_stats. Skipping")
        return

    for key in val_stats[best_epoch - 1].keys():
        if "acc" in key:
            logger.info("  %s: %s", key, val_stats[best_epoch - 1][key])

    # Actually run evaluation on test set with this checkpoint
    if run_test:
        logger.info(f"Running evaluation on test set with checkpoint {best_epoch}")
        command = (
            "python test.py "
            f"--log_dir {log_dir} "
            f"--ckpt_num {int(best_epoch)} "
            f"--split test "
            f"--save_json {save_json}"
        )
        if test_groupings:
            command = f"{command} --groupings {test_groupings}"

        subprocess.run(command, shell=True, check=True)

    return int(best_epoch)


if __name__ == "__main__":
    # In this script, call argparse outside of main so it can be imported by generate_spurious_matrix
    args = parse_args()
    main(
        log_dir=args.log_dir,
        run_test=args.run_test,
        test_groupings=args.test_groupings,
        metric=args.metric,
        learning_type=args.learning_type,
        save_json=args.save_json
    )
