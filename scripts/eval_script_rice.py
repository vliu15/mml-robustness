"""
Handy script to run jobs for hyperparameter grid searches

Sample usage:
python -m scripts.run_hparam_grid_search \
    --template ./scripts/sbatch_template.sh \
    --mode sbatch \
    --slurm_logs ./slurm_logs
"""

import argparse
import os
import subprocess
import uuid


class Color(object):
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


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
    args = parser.parse_args()

    # Convert relative papths to absolute paths to help slurm out
    args.slurm_logs = os.path.abspath(args.slurm_logs)
    return args

def eval_suby_blond_tuning_jobs(args):
    ## DECLARE MACROS HERE ##
    WD_GRID = [1e-2, 1e-1]
    LR_GRID = [1e-5, 1e-4]
    BATCH_SIZE_GRID = [8, 32, 64]
    TASK_GRID = ["Blond_Hair:Male"]
    LOG_DIR = "/mnt/data/mml_robustness_results"

    # Load SBATCH template if specified
    template = ""
    if args.mode == "sbatch":
        with open(args.template, "r") as f:
            template = f.read()

    total_commands = len(WD_GRID) * len(LR_GRID) * len(TASK_GRID) * len(BATCH_SIZE_GRID) 

    counter = 0

    for task in TASK_GRID:
        for wd in WD_GRID:
            for lr in LR_GRID:
                for batch_size in BATCH_SIZE_GRID:
                    job_name = f"eval_task:{task},wd:{wd},lr:{lr},batch_size:{batch_size}"

                    if job_name not in ["eval_task:Blond_Hair:Male,wd:0.01,lr:1e-05,batch_size:8", "eval_task:Blond_Hair:Male,wd:0.01,lr:1e-05,batch_size:32",
                                        "eval_task:Blond_Hair:Male,wd:0.01,lr:1e-05,batch_size:64", "eval_task:Blond_Hair:Male,wd:0.01,lr:0.0001,batch_size:8"]:
                        log_file = os.path.join(args.slurm_logs, f"{job_name}.log")
                        command = (
                            f"python scripts/find_best_ckpt.py --run_test --log_dir ./logs/{job_name[5:]}"
                        )
                        counter += 1

                        if args.mode == "debug":
                            print(command)
                        elif args.mode == "shell":
                            message = f"RUNNING COMMAND {counter} / {total_commands}"
                            print(f"{Color.BOLD}{Color.GREEN}{message}{Color.END}{Color.END}")
                            subprocess.run(command, shell=True, check=True)
                        elif args.mode == "sbatch":
                            sbatch = template.replace("$JOB_NAME", job_name).replace("$LOG_FILE",
                                                                                    log_file).replace("$COMMAND", command)
                            uniq_id = uuid.uuid4()
                            with open(f"{uniq_id}.sh", "w") as f:
                                f.write(sbatch)
                            subprocess.run(f"sbatch {uniq_id}.sh", shell=True, check=True)
                            os.remove(f"{uniq_id}.sh")



def main():
    args = parse_args()
    if args.mode == "sbatch":
        os.makedirs(args.slurm_logs, exist_ok=True)

    eval_suby_blond_tuning_jobs(args)


if __name__ == "__main__":
    main()