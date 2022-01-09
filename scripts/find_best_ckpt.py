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
python -m scripts.find_best_val_ckpt \
    --log_dir logs/erm/Blond_Hair:Male \
    --run_test --test_groupings [Blond_Hair:Male]
"""

import argparse
import json
import logging
import logging.config
import os
import re
import subprocess

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
        required=True,
        help="JSON-string list of {{task}}:{{subgroup}} to run additional evaluation on"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with open(os.path.join(args.log_dir, "val_stats.json"), "r") as f:
        val_stats = json.load(f)

    logger.info("Found the following epochs with validation results: %s", list(map(int, val_stats.keys())))

    # Find epoch with best worst-group accuracy
    best_epoch, best_acc = None, 0.0
    group_acc_key_regex = re.compile(r".*_g[0-9]+_acc")
    for epoch in val_stats.keys():
        worst_group_acc = min(val_stats[epoch][key] for key in val_stats[epoch].keys() if group_acc_key_regex.match(key))
        if worst_group_acc > best_acc:
            best_epoch = epoch
            best_acc = worst_group_acc

    logger.info("Best validation epoch: %s", best_epoch)
    logger.info("Best worst-group accuracy: %s", best_acc)
    for key in val_stats[best_epoch].keys():
        if "acc" in key:
            logger.info("  %s: %s", key, val_stats[best_epoch][key])

    # Actually run evaluation on test set with this checkpoint
    if args.run_test:
        logger.info("Running evaluation on test set with this checkpoint")
        command = (
            f"python test.py "
            f"--log_dir {args.log_dir} "
            f"--ckpt_num {int(best_epoch)} "
            f"--groupings {args.test_groupings} "
            f"--split test "
        )
        subprocess.run(command, shell=True, check=True)


if __name__ == "__main__":
    main()
