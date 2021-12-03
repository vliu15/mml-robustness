"""Generates PNG of confusion matrix of spurious correlations

Sample usage: MUST HAVE ALREADY RUN test.py ON ALL SPURIOUS CORRELATES
python -m scripts.generate_spurious_confusion_matrix \
    --log_dir logs/erm \
    --json_name test_results.json

Output PNG per task is saved into {args.log_dir}/results/heatmap_{task}.png
"""

import argparse
import json
import os
from copy import copy

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf

from datasets.grouping import ATTRIBUTES, get_grouping


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Log directory to load",
    )
    parser.add_argument(
        "--json_name",
        type=str,
        required=True,
        help="Name of JSON of results saved from test.py, should be filename in {{args.log_dir}}/results",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = OmegaConf.load(os.path.join(args.log_dir, "config.yaml"))
    grouping = get_grouping(config.dataset.grouping)

    # Load results from testing
    with open(os.path.join(args.log_dir, "results", args.json_name), "r") as f:
        test_results = json.load(f)

    # Loop through tasks
    for task in grouping.task_labels:
        attributes = copy(ATTRIBUTES)
        attributes.remove(task)

        results_dict = {"Group 0 Acc": [], "Group 1 Acc": [], "Group 2 Acc": [], "Group 3 Acc": []}

        # Loop through attributes
        for attr in attributes:
            j = ATTRIBUTES.index(attr)

            # Retrieve subgroup accuracy
            for k in range(4):
                results_dict[f'Group {k} Acc'] += [test_results[f"{task}_g{j},{k}"]]

        # Save as png
        results_df = pd.DataFrame.from_dict(results_dict)
        results_df = results_df.rename(index={ind: v for ind, v in enumerate(attributes)})

        results_dir = os.path.join(args.log_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        _, ax = plt.subplots(figsize=(16, 10))
        heatmap = sns.heatmap(results_df, annot=True, linewidths=.5, ax=ax)
        ax.set_title(f'Task: {task}')
        plt.xlabel("Subgroups: (Task x Potential Spurious Correlate)")
        plt.ylabel("Potential Spurious Correlates")
        heatmap.figure.savefig(os.path.join(results_dir, f"heatmap_{task}.png"))

        # Reset plt for next task
        plt.cla()
        plt.clf()


if __name__ == "__main__":
    main()
