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

logging.config.fileConfig("logger.conf")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dirs", type=str, nargs='+', required=True, help="Path to log directories with results")
    parser.add_argument(
        "--split", type=str, required=True, choices=["val", "test"], help="Type of metric to report results on"
    )
    parser.add_argument("--checkpoint_type", type=str, required=True, choices=["avg", "group"], help="Whether to choose avg or worst group checkpoint")
    return parser.parse_args()

### print out average accuracy, print out worst group accuracy print out loss
def main():
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

        group_accuracies = {key:results[key] for key in results.keys() if group_acc_key_regex.match(key)}
        worst_group_accuracies = {}
        for key in group_accuracies.keys():
            group_acc_key_regex = r"_g[0-9]+_acc"
            sub_key = re.split(group_acc_key_regex, key)[0]

            if sub_key in worst_group_accuracies:
                curr_val = worst_group_accuracies[sub_key]
                worst_group_accuracies[sub_key] = min(curr_val,group_accuracies[key])
            else:
                worst_group_accuracies[sub_key] = group_accuracies[key]

        avg_accuracies = {key.split("_avg_acc")[0]:results[key] for key in results.keys() if avg_acc_key_regex.match(key)}


        for task, worst_group_acc in worst_group_accuracies.items():
            worst_group_accuracy[task].append(worst_group_acc)

        for task, avg_acc in avg_accuracies.items():
            average_accuracy[task].append(avg_acc)

    
    logger.info(f"For split: {args.split}, using checkpoints based on: {args.checkpoint_type} we obtain: \n")

    for task in average_accuracy.keys():

        logger.info(f"For TASK: {task}")
        logger.info(f"Mean average accuracy: {np.mean(average_accuracy[task])}, std:{np.std(average_accuracy[task])}, over {len(average_accuracy[task])} seeds \n")
        logger.info(f"Mean worst-group accuracy: {np.mean(worst_group_accuracy[task])}, std:{np.std(worst_group_accuracy[task])}, over {len(worst_group_accuracy[task])} seeds \n")



if __name__ == "__main__":
    main()

