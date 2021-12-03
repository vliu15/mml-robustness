"""Script to run test on all spurious correlations

Sample usage:
python -m scripts.run_all_spurious_test \
    --log_dir logs/erm \
    --ckpt_num 12
"""

import argparse
import os
import subprocess
from copy import copy

from omegaconf import OmegaConf

from datasets.groupings import ATTRIBUTES, get_grouping_object


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True, help="Log directory to load for test")
    parser.add_argument("--ckpt_num", type=int, required=True, help="Checkpoint number to load")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config to get training tasks
    config = OmegaConf.load(os.path.join(args.log_dir, "config.yaml"))
    grouping = get_grouping_object(config.dataset.groupings)
    tasks = grouping.task_labels

    # Append all extra attributes per task
    extra_attributes_flag = []
    for task in tasks:
        extra_attributes = copy(ATTRIBUTES)
        extra_attributes.remove(task)
        extra_attributes_flag += [f"{task}:{','.join(extra_attributes)}"]

    # Run test
    subprocess.run(
        f"python test.py "
        f"--log_dir {args.log_dir} "
        f"--ckpt_num {args.ckpt_num} "
        f"--extra_attributes {' '.join(extra_attributes_flag)}",
        check=True,
        shell=True,
    )


if __name__ == "__main__":
    main()
