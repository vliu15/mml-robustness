"""Returns results dict with best worst-subgroup accuracy

Sample usage:
python -m scripts.find_best_test_result \
    --results_dir ./logs/jtt/stage_2/results/ \
    --task Attractive \
    --attr Smiling
"""

import argparse
import json
import os
from pprint import pprint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing all test results jsons")
    parser.add_argument("--task", type=str, required=True, help="Task that the model was trained on")
    parser.add_argument("--attr", type=str, required=True, help="Spurious correlate that the task was evaluated against")
    return parser.parse_args()


def main():
    args = parse_args()

    results_dicts = [
        os.path.join(args.results_dir, f) for f in os.listdir(args.results_dir) if f.endswith(".json") and args.attr in f
    ]
    best = {}
    best_worst_subgroup_acc = 0.0
    for results_dict in results_dicts:
        with open(results_dict, "r") as f:
            results_dict = json.load(f)

        subgroup_accs = [value for key, value in results_dict.items() if f"{args.task}_g" in key]
        if min(subgroup_accs) >= best_worst_subgroup_acc:
            best_worst_subgroup_acc = min(subgroup_accs)
            best = results_dict

    print(f"Best worst-subgroup accuracy: {best_worst_subgroup_acc}")
    pprint(best)


if __name__ == "__main__":
    main()
