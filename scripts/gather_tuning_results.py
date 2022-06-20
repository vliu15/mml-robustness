"""
Gathers all best average accuracies from hparam tuning
from log directories specified by the regex args.log_dirs
and prints out hparams for each task found, sorted by accuracy

Sample usage:
python -m scripts.gather_tuning_results \
    --log_dirs "logs/spurious_id_tune/.*" \
    --learning_type stl
"""

import argparse
import json
import logging
import logging.config
import os
import re
from collections import defaultdict

from omegaconf import OmegaConf
from tabulate import tabulate

from scripts.find_best_ckpt import main as find_best_ckpt

logging.config.fileConfig("logger.conf")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dirs", type=str, required=True, help="Regex pattern of log directories with results")
    parser.add_argument(
        "--mtl_checkpoint_type",
        type=str,
        default="average",
        choices=["average", "best-worst", "per-task", None],
        help="Whether to choose checkpointing based on the average performance, best worst performance, or per-task"
    )
    parser.add_argument(
        "--learning_type",
        type=str,
        default="stl",
        choices=["stl", "mtl"],
        help="Whether the model is trained with STL or MTL"
    )
    return parser.parse_args()


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

    # [2] Grab the hyperparameters and report sorted average accuracies per task that is present in args.log_dirs
    # NOTE: we should only expect to be tuning over WD, LR, and BATCH_SIZE (see GRIDS in run_hparam_grid_train)
    task_to_hparam_results = defaultdict(lambda: defaultdict(list))
    for log_dir in log_dirs:
        # Retrieve best checkpoint for this log_dir
        best_ckpt = find_best_ckpt(
            log_dir,
            mtl_checkpoint_type=args.mtl_checkpoint_type,
            learning_type=args.learning_type,
            # These below are all defaults, but specify just for clarity
            test_groupings="",
            run_test=False,
            metric="avg",
            save_json="",
        )

        # Load these hyperparameter fields
        config = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
        wd = config.optimizer.weight_decay
        lr = config.optimizer.lr
        batch_size = config.dataloader.batch_size
        tasks = [task.split(":")[0] for task in config.dataset.groupings]

        # Convert best_ckpt to dict format (compatible for MTL)
        if isinstance(best_ckpt, int):
            assert len(tasks) == 1, "find_best_ckpt returned a single integer, but we found multiple tasks in the config."
            best_ckpt = {tasks[0]: best_ckpt}

        # Update our aggregation of results per hyperparam combo per task
        for task in tasks:
            with open(os.path.join(log_dir, "results", f"val_stats_{best_ckpt[task] - 1}.json"), "r") as f:
                results = json.load(f)
                task_to_hparam_results[task]["wd"].append(wd)
                task_to_hparam_results[task]["lr"].append(lr)
                task_to_hparam_results[task]["batch_size"].append(batch_size)
                task_to_hparam_results[task]["avg_acc"].append(results[f"{task}_avg_acc"])

    # After aggregating, sort and print
    for task, hparam_results in task_to_hparam_results.items():
        sorted_results = list(
            sorted(
                zip(hparam_results["wd"], hparam_results["lr"], hparam_results["batch_size"], hparam_results["avg_acc"]),
                key=lambda *args: args[-1],  # only sort by average accuracy
                reverse=True,
            )
        )
        print(f"== TASK: {task.upper()} ==")
        print(tabulate(sorted_results, headers=["wd", "lr", "batch_size", "avg_acc"]))


if __name__ == "__main__":
    main()
