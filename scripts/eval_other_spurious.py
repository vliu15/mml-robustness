"""
Runs validation on any new spurious correlates with checkpoints from training and
dumps the results in a new logdir inside the original

Sample usage:
python -m scripts.eval_other_spurious \
    --log_dir logs/erm/Blond_Hair:Male \
    --run_test --groupings [Blond_Hair:Male]
"""

import argparse
import os
import re
import subprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True, help="Path to log directory with val_stats.json")
    parser.add_argument(
        "--run_test", default=False, action="store_true", help="Whether to evaluate best checkpoint on test set"
    )
    parser.add_argument(
        "--groupings",
        type=str,
        required=True,
        help="JSON-string list of {{task}}:{{subgroup}} to run additional evaluation on"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_regex = re.compile(r"ckpt\.[0-9]+\.pt")
    ckpt_nums = [int(ckpt.split(".")[1]) for ckpt in os.listdir(os.path.join(args.log_dir, "ckpts")) if ckpt_regex.match(ckpt)]

    # Prepare new log dir to dump val stats
    new_log_dir = os.path.join(args.log_dir, ",".join(args.groupings))
    os.makedirs(new_log_dir, exist_ok=True)

    # Run validation for each checkpoint on these new groupings
    for ckpt_num in ckpt_nums:
        save_json = os.path.join(new_log_dir, "results", f"val_stats_{ckpt_num}.json")
        command = (
            "python test.py "
            f"--log_dir {args.log_dir} "
            f"--groupings {args.groupings} "
            f"--ckpt_num {ckpt_num} "
            f"--save_json {save_json} "
            f"--split val"
        )
        subprocess.run(command, shell=True, check=True)

    # Now find the best checkpoint
    command = (
        "python -m scripts.find_best_ckpt "
        f"--log_dir {new_log_dir} "
        f"--run_test {args.run_test} "
        f"--test_groupings {args.groupings}"
    )
    subprocess.run(command, shell=True, check=True)


if __name__ == "__main__":
    main()
