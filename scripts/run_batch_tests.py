"""
Runs and collects results on a batch of experiments.

Sample usage from repo root:
python -m scripts.run_batch_tests \
    logs/jtt-g0 logs/jtt-g1:5,10 logs/jtt-g2 logs/jtt-g3:6 logs/jtt-g4 logs/jtt-g8:9 \
    --json_file results.json
"""

import argparse
import json
import logging
import logging.config
import os
import subprocess
from collections import defaultdict

logging.config.fileConfig("logger.conf")
logger = logging.getLogger(__name__)

# #################################################
# For reference, here are groupings that belong
# together based on same task label during training
# Blond_Hair  : [0=7]
# Attractive  : [1,5,10]
# Smiling     : [2]
# Young       : [3,6]
# Oval_Face   : [4]
# Pointy_Nose : [8,9]
# #################################################


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "log_dirs_and_subgroups",
        metavar="log_dir",
        type=str,
        nargs="+",
        help="Sequence of log dirs and additional groupings to run tests on. Format: {{log_dir}}:g1,g2",
    )
    parser.add_argument(
        "--json_file",
        type=str,
        default="results",
        help="Json file to dump results",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    assert args.json_file.endswith(".json"), "Must specify a json file for --results_json"
    results_json = defaultdict(dict)

    # Loop over log_dirs specified
    for i, log_dir in enumerate(args.log_dirs_and_subgroups):
        log_dir, subgroup_attributes = log_dir.split(":")
        logger.info(f"== [{i + 1} / {len(args.log_dirs_and_subgroups)}] Running tests on {log_dir} ==")

        # Filter out ckpt.last.pt
        ckpts = sorted(
            [f for f in os.listdir(os.path.join(log_dir, "ckpts")) if "last" not in f],
            key=lambda f: int(f.split(".")[1]),
        )

        # Loop over checkpoints in each log_dir
        for j, ckpt in enumerate(ckpts):
            logger.info(f" - [{j + 1} / {len(ckpts)}] Running {ckpt} -")
            ckpt_num = int(ckpt.split(".")[1])

            # Run test.py, which saves results.json into {log_dir}/results
            extra_subgroups = "" if subgroup_attributes == "" else f"--subgroup_attributes {subgroup_attributes}"
            subprocess.run(
                f"python test.py "
                f"--log_dir {log_dir} "
                f"--ckpt_num {ckpt_num} "
                f"{extra_subgroups}",
                check=True,
                shell=True,
            )

            for json_name in os.listdir(os.path.join(log_dir, "results")):
                with open(os.path.join(log_dir, "results", json_name), "r") as f:
                    results_json[os.path.join(log_dir, json_name)][ckpt_num] = json.load(f)

    # Write cumulative results file
    with open(args.json_file, "w") as f:
        json.dump(results_json, f)


if __name__ == "__main__":
    main()
