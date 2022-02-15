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

from submit_job import JobManager

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
    args = parser.parse_args()

    # Convert relative papths to absolute paths to help slurm out
    args.slurm_logs = os.path.abspath(args.slurm_logs)
    return args


def submit_erm_grid_jobs(args):
    ## DECLARE MACROS HERE ##
    WD_GRID = [1e-4, 1e-3, 1e-2, 1e-1]
    LR_GRID = [1e-5, 5e-5, 1e-4]
    TASK_GRID = [
        "Attractive:Eyeglasses",
        "Smiling:High_Cheekbones",
        "Young:Attractive",
        "Oval_Face:Rosy_Cheeks",
        "Pointy_Nose:Rosy_Cheeks",
    ]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)

    for wd in WD_GRID:
        for lr in LR_GRID:
            for task in TASK_GRID:
                job_name = f"task:{task},wd:{wd},lr:{lr}"
                log_file = os.path.join(args.slurm_logs, f"{job_name}.log")
                command = (
                    "python train_erm.py exp=erm "
                    f"exp.optimizer.weight_decay={wd} "
                    f"exp.optimizer.lr={lr} "
                    f"exp.dataset.groupings='[{task}]' "
                    f"exp.train.log_dir=\\'{os.path.join(LOG_DIR, job_name)}\\'"
                )
                job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_mtl_erm_grid_jobs(args):
    ## DECLARE MACROS HERE ##
    WD_GRID = [1e-4, 1e-3, 1e-2, 1e-1]
    LR_GRID = [1e-5, 5e-5, 1e-4]
    ALPHA_GRID = [0.1, 0.5, 1]
    TASK_GRID = [
        "Attractive:Eyeglasses,Smiling:High_Cheekbones", "Young:Attractive,Smiling:High_Cheekbones",
        "Blond_Hair:Male,Pointy_Nose:Male", "Pointy_Nose:Rosy_Cheeks,Attractive:Heavy_Makeup"
    ]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)

    for task in TASK_GRID:
        for wd in WD_GRID:
            for lr in LR_GRID:
                for alpha in ALPHA_GRID:
                    job_name = f"task:{task},wd:{wd},lr:{lr},alpha:{alpha}"
                    log_file = os.path.join(args.slurm_logs, f"{job_name}.log")
                    command = (
                        "python train_erm.py exp=erm "
                        f"exp.optimizer.weight_decay={wd} "
                        f"exp.optimizer.lr={lr} "
                        f"exp.dataset.groupings='[{task}]' "
                        f"exp.dataset.lbtw_alpha={alpha} "
                        f"exp.train.log_dir=\\'{os.path.join(LOG_DIR, job_name)}\\'"
                    )
                    job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_reweighted_subsampled_tuning_jobs(args):
    ## DECLARE MACROS HERE ##
    WD_GRID = [1e-2, 1e-1, 1] # 10−4, 10−3, 10−2, 10−1, 1
    LR_GRID = [1e-5, 1e-4, 1e-3] # 10−5, 10−4, 10−3
    BATCH_SIZE_GRID = [32, 64] # 2, 4, 8, 16, 32, 64, 128
    TASK_GRID = [
        "Attractive:Eyeglasses",
        "Smiling:High_Cheekbones",
        "Young:Attractive",
    ]

    ## "Oval_Face:Rosy_Cheeks", still need these to tune
    ##"Pointy_Nose:Rosy_Cheeks",
    method = "suby"

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)

    for task in TASK_GRID:
        for wd in WD_GRID:
            for lr in LR_GRID:
                for batch_size in BATCH_SIZE_GRID:
                    job_name = f"task:{task},wd:{wd},lr:{lr},batch_size:{batch_size}"

            
                    log_file = os.path.join(args.slurm_logs, f"{job_name}.log")
                    command = (
                        f"python train_erm.py exp={method} "
                        f"exp.optimizer.weight_decay={wd} "
                        f"exp.optimizer.lr={lr} "
                        f"exp.dataset.groupings='[{task}]' "
                        f"exp.dataloader.batch_size={batch_size} "
                        f"exp.train.log_dir=\\'{os.path.join(LOG_DIR, job_name)}\\'"
                    )
                    job_manager.submit(command, job_name=job_name, log_file=log_file)


def main():
    args = parse_args()
    if args.mode == "sbatch":
        os.makedirs(args.slurm_logs, exist_ok=True)

    submit_reweighted_subsampled_tuning_jobs(args)


if __name__ == "__main__":
    main()
