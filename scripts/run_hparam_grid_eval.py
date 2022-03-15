"""
Handy script to run jobs for hyperparameter grid searches

Sample usage:
python -m scripts.run_hparam_grid_eval \
    --template ./scripts/sbatch_template.sh \
    --mode sbatch \
    --slurm_logs ./slurm_logs \
    --opt suby
"""

import argparse
import json
import os

from scripts.find_best_ckpt import main as find_best_ckpt
from scripts.job_manager import JobManager

# RICE MACROS
USER = os.environ["USER"]
LOG_DIR = f"/farmshare/user_data/{USER}/mml-robustness/logs"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template", type=str, default="scripts/sbatch_template.sh", required=False, help="SBATCH template file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="debug",
        choices=["debug", "shell", "sbatch"],
        help="Whether to run this script in debug mode, run in shell, or submit as sbatch"
    )
    parser.add_argument("--slurm_logs", type=str, default="slurm_logs", required=False, help="Directory to output slurm logs")
    parser.add_argument("--opt", type=str, required=True, help="The name of the submit_*_grid_jobs function to call.")
    args = parser.parse_args()

    # Convert relative papths to absolute paths to help slurm out
    args.slurm_logs = os.path.abspath(args.slurm_logs)
    return args


def submit_suby_eval_test(args):
    ## DECLARE MACROS HERE ##
    WD_GRID = [1e-2, 1e-1, 1]  # 10−4, 10−3, 10−2, 10−1, 1
    LR_GRID = [1e-5, 1e-4, 1e-3]  # 10−5, 10−4, 10−3
    BATCH_SIZE_GRID = [32, 64]  # 2, 4, 8, 16, 32, 64, 128
    TASK_GRID = [
        "Attractive:Eyeglasses",
    ]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)

    for task in TASK_GRID:
        for wd in WD_GRID:
            for lr in LR_GRID:
                for batch_size in BATCH_SIZE_GRID:
                    job_name = f"eval_task:{task},wd:{wd},lr:{lr},batch_size:{batch_size}"
                    log_file = os.path.join(args.slurm_logs, f"{job_name}.log")
                    command = f"python -m scripts.find_best_ckpt --run_test --log_dir ./logs/{job_name[5:]} --metric avg"
                    job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_suby_eval_val(args):
    ## DECLARE MACROS HERE ##
    WD_GRID = [1e-2, 1e-1, 1]  # 10−4, 10−3, 10−2, 10−1, 1
    LR_GRID = [1e-5, 1e-4, 1e-3]  # 10−5, 10−4, 10−3
    BATCH_SIZE_GRID = [32, 64]  # 2, 4, 8, 16, 32, 64, 128
    TASK_GRID = [
        "Attractive:Eyeglasses",
    ]

    for task in TASK_GRID:
        for wd in WD_GRID:
            for lr in LR_GRID:
                for batch_size in BATCH_SIZE_GRID:
                    job_name = f"eval_task:{task},wd:{wd},lr:{lr},batch_size:{batch_size}"

                    ckpt_num = find_best_ckpt(f'./logs/{job_name[5:]}', run_test=False, test_groupings="", metric="avg")

                    with open(os.path.join(f"./logs/{job_name[5:]}", "results", f"val_stats_{ckpt_num}.json"), "r") as f:
                        best_val_stats = json.load(f)

                    with open(os.path.join(f"./logs/{job_name[5:]}", "results", f"best_val_stats_{ckpt_num}.json"), "w") as fp:
                        json.dump(best_val_stats, fp)


def main():
    args = parse_args()
    if args.mode == "sbatch":
        os.makedirs(args.slurm_logs, exist_ok=True)

    if args.opt == "suby":
        submit_suby_eval_val(args)
    else:
        raise ValueError(f"Didn't recognize opt={args.opt}. Did you forget to add a check for this function?")


if __name__ == "__main__":
    main()
