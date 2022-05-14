"""
Handy script to run jobs for hyperparameter grid searches

Sample usage:
python -m scripts.run_hparam_grid_train \
    --template ./scripts/sbatch_template.sh \
    --mode sbatch \
    --slurm_logs ./slurm_logs \
    --opt erm

For runs on any device that has a time limit per job, once the jobs for a given run finish please 
re-run the script with the --respawn flag to ensure that all jobs completed all checkpoints
"""

import argparse
import os
import re

from mtl.utils import get_mtl_task_weights
from scripts.job_manager import JobManager

# RICE MACROS
USER = os.environ["USER"]
#LOG_DIR = f"/farmshare/user_data/{USER}/mml-robustness/logs"
LOG_DIR = "./logs"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template", type=str, default="scripts/rice_sbatch_template.sh", required=False, help="SBATCH template file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="debug",
        choices=["debug", "shell", "sbatch"],
        help="Whether to run this script in debug mode, run in shell, or submit as sbatch"
    )
    parser.add_argument(
        "--slurm_logs", type=str, default="./slurm_logs", required=False, help="Directory to output slurm logs"
    )
    parser.add_argument("--opt", type=str, required=True, help="The name of the submit_*_grid_jobs function to call.")
    parser.add_argument("--respawn", action='store_true', help="Whether to respawn runs from last completed checkpoint")
    parser.add_argument(
        "--mtl_weighting",
        type=str,
        default="static_equal",
        choices=["static_equal", "static_delta", "dynamic"],
        help="For MTL tuning runs, what type of weighting to use across tasks"
    )
    args = parser.parse_args()

    # Convert relative paths to absolute paths to help slurm out
    args.slurm_logs = os.path.abspath(args.slurm_logs)
    return args


def find_last_checkpoint(ckpt_dir):

    ckpt_regex = re.compile(r"ckpt.[0-9]+\.pt")
    ckpt_epochs = []
    for ckpt_file in sorted(os.listdir(ckpt_dir)):
        if ckpt_regex.match(ckpt_file):
            epoch = int(ckpt_file.split(".")[1])
            ckpt_epochs.append(epoch)

    if len(ckpt_epochs) == 0:
        return None, None

    max_epoch = max(ckpt_epochs)
    max_epoch_path = os.path.join(ckpt_dir, f"ckpt.{max_epoch}.pt")

    return max_epoch_path, max_epoch


def submit_erm_train(args):
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


def submit_reweighted_subsampled_train(args):
    ## DECLARE MACROS HERE ##
    WD_GRID = [1e-2, 1e-1, 1]  # 10−4, 10−3, 10−2, 10−1, 1
    LR_GRID = [1e-5, 1e-4, 1e-3]  # 10−5, 10−4, 10−3
    BATCH_SIZE_GRID = [32, 64]  # 2, 4, 8, 16, 32, 64, 128
    TASK_GRID = [
        "Smiling:High_Cheekbones", "Pointy_Nose:Rosy_Cheeks", "Oval_Face:Rosy_Cheeks", "Young:Attractive",
        "Attractive:Eyeglasses"
    ]

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


def submit_erm_baseline_disjoint_tasks_train(args):
    ## DECLARE MACROS HERE ##
    WD = 1e-4
    LR = 1e-4
    BATCH_SIZE = 128
    EPOCHS = 50
    SEED_GRID = [0, 1, 2]

    TASK_GRID = ["Wearing_Earrings:Male", "Attractive:Male", "No_Beard:Heavy_Makeup", "Pointy_Nose:Heavy_Makeup",
    "Attractive:Gray_Hair", "Big_Nose:Gray_Hair", "Heavy_Makeup:Wearing_Lipstick", "No_Beard:Wearing_Lipstick",
    "Bangs:Wearing_Hat", "Blond_Hair:Wearing_Hat"]

    #TASK_GRID = [
    #    "Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair", "Goatee:No_Beard", "Gray_Hair:Young", "High_Cheekbones:Smiling",
    #    "Wavy_Hair:Straight_Hair", "Wearing_Lipstick:Male"
    #]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)
    method = "erm"
    for task in TASK_GRID:
        for seed in SEED_GRID:
            job_name = f"baseline:{method},task:{task},seed:{seed}"

            log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

            if args.respawn:
                ckpt_dir = os.path.join(LOG_DIR, job_name, "ckpts")
                ckpt_path, ckpt_num = find_last_checkpoint(ckpt_dir)

                if ckpt_num != EPOCHS:

                    command = (
                        f"python train_erm.py exp={method} "
                        f"exp.optimizer.weight_decay={WD} "
                        f"exp.optimizer.lr={LR} "
                        f"exp.seed={seed} "
                        f"exp.train.total_epochs={EPOCHS} "
                        f"exp.dataset.groupings='[{task}]' "
                        f"exp.dataloader.batch_size={BATCH_SIZE} "
                        f"exp.train.load_ckpt=\\'{ckpt_path}\\' "
                        f"exp.train.log_dir=\\'{os.path.join(LOG_DIR, job_name)}\\'"
                    )
                    job_manager.submit(command, job_name=job_name, log_file=log_file)

            else:
                command = (
                    f"python train_erm.py exp={method} "
                    f"exp.optimizer.weight_decay={WD} "
                    f"exp.optimizer.lr={LR} "
                    f"exp.seed={seed} "
                    f"exp.train.total_epochs={EPOCHS} "
                    f"exp.dataset.groupings='[{task}]' "
                    f"exp.dataloader.batch_size={BATCH_SIZE} "
                    f"exp.train.log_dir=\\'{os.path.join(LOG_DIR, job_name)}\\'"
                )
                job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_suby_baseline_disjoint_tasks_train(args):
    ## DECLARE MACROS HERE ##
    WD = 1e-2
    LR = 1e-3
    BATCH_SIZE = 128
    EPOCHS = 60
    SEED_GRID = [0, 1, 2]
    TASK_GRID = [
        "Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair", "Goatee:No_Beard", "Gray_Hair:Young", "High_Cheekbones:Smiling",
        "Wavy_Hair:Straight_Hair", "Wearing_Lipstick:Male"
    ]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)

    method = "suby"
    for task in TASK_GRID:
        for seed in SEED_GRID:
            job_name = f"baseline:{method},task:{task},seed:{seed}"

            log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

            if args.respawn:
                ckpt_dir = os.path.join(LOG_DIR, job_name, "ckpts")
                ckpt_path, ckpt_num = find_last_checkpoint(ckpt_dir)

                if ckpt_num != EPOCHS:

                    command = (
                        f"python train_erm.py exp={method} "
                        f"exp.optimizer.weight_decay={WD} "
                        f"exp.optimizer.lr={LR} "
                        f"exp.seed={seed} "
                        f"exp.train.total_epochs={EPOCHS} "
                        f"exp.dataset.groupings='[{task}]' "
                        f"exp.dataloader.batch_size={BATCH_SIZE} "
                        f"exp.train.load_ckpt=\\'{ckpt_path}\\' "
                        f"exp.train.log_dir=\\'{os.path.join(LOG_DIR, job_name)}\\'"
                    )
                    job_manager.submit(command, job_name=job_name, log_file=log_file)
            else:

                command = (
                    f"python train_erm.py exp={method} "
                    f"exp.optimizer.weight_decay={WD} "
                    f"exp.optimizer.lr={LR} "
                    f"exp.seed={seed} "
                    f"exp.train.total_epochs={EPOCHS} "
                    f"exp.dataset.groupings='[{task}]' "
                    f"exp.dataloader.batch_size=\\'{BATCH_SIZE}\\' "
                    f"exp.train.log_dir=\\'{os.path.join(LOG_DIR, job_name)}\\'"
                )
                job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_rwy_baseline_disjoint_tasks_train(args):
    ## DECLARE MACROS HERE ##
    WD = 1e-2
    LR = 1e-4
    BATCH_SIZE = 2
    EPOCHS = 60
    SEED_GRID = [0, 1, 2]
    TASK_GRID = [
        "Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair", "Goatee:No_Beard", "Gray_Hair:Young", "High_Cheekbones:Smiling",
        "Wavy_Hair:Straight_Hair", "Wearing_Lipstick:Male"
    ]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)

    method = "rwy"
    for task in TASK_GRID:
        for seed in SEED_GRID:
            job_name = f"baseline:{method},task:{task},seed:{seed}"

            log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

            if args.respawn:
                ckpt_dir = os.path.join(LOG_DIR, job_name, "ckpts")
                ckpt_path, ckpt_num = find_last_checkpoint(ckpt_dir)

                if ckpt_num != EPOCHS:
                    command = (
                        f"python train_erm.py exp={method} "
                        f"exp.optimizer.weight_decay={WD} "
                        f"exp.optimizer.lr={LR} "
                        f"exp.seed={seed} "
                        f"exp.train.total_epochs={EPOCHS} "
                        f"exp.dataset.groupings='[{task}]' "
                        f"exp.dataloader.batch_size={BATCH_SIZE} "
                        f"exp.train.load_ckpt=\\'{ckpt_path}\\' "
                        f"exp.train.log_dir=\\'{os.path.join(LOG_DIR, job_name)}\\'"
                    )

                    job_manager.submit(command, job_name=job_name, log_file=log_file)

            else:
                command = (
                    f"python train_erm.py exp={method} "
                    f"exp.optimizer.weight_decay={WD} "
                    f"exp.optimizer.lr={LR} "
                    f"exp.seed={seed} "
                    f"exp.train.total_epochs={EPOCHS} "
                    f"exp.dataset.groupings='[{task}]' "
                    f"exp.dataloader.batch_size={BATCH_SIZE} "
                    f"exp.train.log_dir=\\'{os.path.join(LOG_DIR, job_name)}\\'"
                )

                job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_jtt_baseline_disjoint_tasks_train(args):
    ## DECLARE MACROS HERE ##
    T = 1
    LAM_UP = 50
    WD = 1e-1
    LR = 1e-5
    BATCH_SIZE = 128
    EPOCHS = 50
    SEED_GRID = [0, 1, 2]
    TASK_GRID = [
        "Bushy_Eyebrows:Blond_Hair", "Goatee:No_Beard", "Gray_Hair:Young", "High_Cheekbones:Smiling",
        "Wavy_Hair:Straight_Hair", "Wearing_Lipstick:Male"
    ]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)

    method = "jtt"
    for task in TASK_GRID:
        for seed in SEED_GRID:
            job_name = f"baseline:{method},task:{task},seed:{seed}"

            log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

            if args.respawn:
                stage_1_ckpt_dir = os.path.join(LOG_DIR, job_name, 'stage_1', "ckpts")
                stage_1_ckpt_path, stage_1_ckpt_num = find_last_checkpoint(stage_1_ckpt_dir)

                if stage_1_ckpt_num == T:
                    stage_2_ckpt_dir = os.path.join(LOG_DIR, job_name, 'stage_2', "ckpts")
                    stage_2_ckpt_path, stage_2_ckpt_num = find_last_checkpoint(stage_2_ckpt_dir)
                else:
                    stage_2_ckpt_path, stage_2_ckpt_num = "null", "null"

                if stage_1_ckpt_num != T:
                    command = (
                        f"python train_jtt.py exp={method} "
                        f"exp.weight_decay={WD} "
                        f"exp.lr={LR} "
                        f"exp.seed={seed} "
                        f"exp.epochs_stage_1={T} "
                        f"exp.epochs_stage_2={EPOCHS} "
                        f"exp.groupings='[{task}]' "
                        f"exp.lambda_up={LAM_UP} "
                        f"exp.load_stage_1_ckpt=\\'{stage_1_ckpt_path}\\' "
                        f"exp.load_stage_2_ckpt={stage_2_ckpt_path} "
                        f"exp.batch_size={BATCH_SIZE} "
                        f"exp.log_dir=\\'{os.path.join(LOG_DIR, job_name)}\\'"
                    )
                    job_manager.submit(command, job_name=job_name, log_file=log_file)

                elif stage_1_ckpt_num == T and stage_2_ckpt_num == None:
                    load_up_pkl_path = os.path.join(LOG_DIR, job_name, f"jtt_error_set_inv.pkl")
                    command = (
                        f"python train_jtt.py exp={method} "
                        f"exp.weight_decay={WD} "
                        f"exp.lr={LR} "
                        f"exp.seed={seed} "
                        f"exp.epochs_stage_1={T} "
                        f"exp.epochs_stage_2={EPOCHS} "
                        f"exp.groupings='[{task}]' "
                        f"exp.lambda_up={LAM_UP} "
                        f"exp.load_stage_1_ckpt=\\'{stage_1_ckpt_path}\\' "
                        f"exp.load_stage_2_ckpt={'null'} "
                        f"exp.load_up_pkl=\\'{load_up_pkl_path}\\' "
                        f"exp.batch_size={BATCH_SIZE} "
                        f"exp.log_dir=\\'{os.path.join(LOG_DIR, job_name)}\\'"
                    )
                    job_manager.submit(command, job_name=job_name, log_file=log_file)

                elif stage_1_ckpt_num == T and stage_2_ckpt_num != EPOCHS:
                    load_up_pkl_path = os.path.join(LOG_DIR, job_name, f"jtt_error_set_inv.pkl")
                    command = (
                        f"python train_jtt.py exp={method} "
                        f"exp.weight_decay={WD} "
                        f"exp.lr={LR} "
                        f"exp.seed={seed} "
                        f"exp.epochs_stage_1={T} "
                        f"exp.epochs_stage_2={EPOCHS} "
                        f"exp.groupings='[{task}]' "
                        f"exp.lambda_up={LAM_UP} "
                        f"exp.load_stage_1_ckpt=\\'{stage_1_ckpt_path}\\' "
                        f"exp.load_stage_2_ckpt=\\'{stage_2_ckpt_path}\\' "
                        f"exp.load_up_pkl=\\'{load_up_pkl_path}\\' "
                        f"exp.batch_size={BATCH_SIZE} "
                        f"exp.log_dir=\\'{os.path.join(LOG_DIR, job_name)}\\'"
                    )
                    job_manager.submit(command, job_name=job_name, log_file=log_file)

            else:
                command = (
                    f"python train_jtt.py exp={method} "
                    f"exp.weight_decay={WD} "
                    f"exp.lr={LR} "
                    f"exp.seed={seed} "
                    f"exp.epochs_stage_1={T} "
                    f"exp.epochs_stage_2={EPOCHS} "
                    f"exp.groupings='[{task}]' "
                    f"exp.lambda_up={LAM_UP} "
                    f"exp.batch_size={BATCH_SIZE} "
                    f"exp.log_dir=\\'{os.path.join(LOG_DIR, job_name)}\\'"
                )
                job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_mtl_disjoint_tasks_tune(args):
    ## DECLARE MACROS HERE ##
    WD_GRID = [1e-4, 1e-3, 1e-2, 1e-1]  #1e-4, 1e-3, 1e-2
    LR_GRID = [1e-5, 1e-4, 1e-3]
    BATCH_SIZE_GRID = [32, 64, 128]
    EPOCHS = 50
    SEED_GRID = [0]
    TASK = ["Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair"]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)
    method = "erm"

    task_weights, use_loss_balanced, lbtw_alpha = get_mtl_task_weights(args.mtl_weighting, TASK)

    for wd in WD_GRID:
        for lr in LR_GRID:
            for batch_size in BATCH_SIZE_GRID:
                for seed in SEED_GRID:

                    job_name = f"mtl_tuning:{method},task:{len(TASK)}_tasks_{args.mtl_weighting}_task_weighting,seed:{seed},wd:{wd},lr:{lr},batch_size:{batch_size}"
                    log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

                    if args.respawn:
                        ckpt_dir = os.path.join(LOG_DIR, job_name, "ckpts")
                        ckpt_path, ckpt_num = find_last_checkpoint(ckpt_dir)

                        if ckpt_num != EPOCHS:

                            command = (
                                f"python train_erm.py exp={method} "
                                f"exp.optimizer.weight_decay={wd} "
                                f"exp.optimizer.lr={lr} "
                                f"exp.seed={seed} "
                                f"exp.train.total_epochs={EPOCHS} "
                                f"exp.dataset.groupings='{TASK}' "
                                f"exp.dataloader.batch_size={batch_size} "
                                f"exp.dataset.task_weights='{task_weights}' "
                                f"exp.dataset.loss_based_task_weighting={use_loss_balanced} "
                                f"exp.dataset.lbtw_alpha={lbtw_alpha} "
                                f"exp.train.load_ckpt=\\'{ckpt_path}\\' "
                                f"exp.train.log_dir=\\'{os.path.join(LOG_DIR, job_name)}\\'"
                            )
                            job_manager.submit(command, job_name=job_name, log_file=log_file)
                    else:
                        command = (
                            f"python train_erm.py exp={method} "
                            f"exp.optimizer.weight_decay={wd} "
                            f"exp.optimizer.lr={lr} "
                            f"exp.seed={seed} "
                            f"exp.train.total_epochs={EPOCHS} "
                            f"exp.dataset.groupings='{TASK}' "
                            f"exp.dataloader.batch_size={batch_size} "
                            f"exp.dataset.task_weights='{task_weights}' "
                            f"exp.dataset.loss_based_task_weighting={use_loss_balanced} "
                            f"exp.dataset.lbtw_alpha={lbtw_alpha} "
                            f"exp.train.log_dir=\\'{os.path.join(LOG_DIR, job_name)}\\'"
                        )
                        job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_mtl_disjoint_tasks_train_avg(args):
    ## DECLARE MACROS HERE ##
    WD = 1e-1
    LR = 1e-4
    BATCH_SIZE = 64
    EPOCHS = 50
    SEED_GRID = [0, 1, 2]
    TASK = ["Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair", "Wearing_Lipstick:Male", "Gray_Hair:Young", "High_Cheekbones:Smiling"]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)
    method = "erm"

    task_weights, use_loss_balanced, lbtw_alpha = get_mtl_task_weights(args.mtl_weighting, TASK)

    for seed in SEED_GRID:
        job_name = f"mtl_train:{method},task:{len(TASK)}_tasks_{args.mtl_weighting}_task_weighting,seed:{seed},ckpt:avg"
        log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

        if args.respawn:
            ckpt_dir = os.path.join(LOG_DIR, job_name, "ckpts")
            ckpt_path, ckpt_num = find_last_checkpoint(ckpt_dir)

            if ckpt_num != EPOCHS:

                command = (
                    f"python train_erm.py exp={method} "
                    f"exp.optimizer.weight_decay={WD} "
                    f"exp.optimizer.lr={LR} "
                    f"exp.seed={seed} "
                    f"exp.train.total_epochs={EPOCHS} "
                    f"exp.dataset.groupings='{TASK}' "
                    f"exp.dataloader.batch_size={BATCH_SIZE} "
                    f"exp.dataset.task_weights='{task_weights}' "
                    f"exp.dataset.loss_based_task_weighting={use_loss_balanced} "
                    f"exp.dataset.lbtw_alpha={lbtw_alpha} "
                    f"exp.train.load_ckpt=\\'{ckpt_path}\\' "
                    f"exp.train.log_dir=\\'{os.path.join(LOG_DIR, job_name)}\\'"
                )
                job_manager.submit(command, job_name=job_name, log_file=log_file)
        else:
            command = (
                f"python train_erm.py exp={method} "
                f"exp.optimizer.weight_decay={WD} "
                f"exp.optimizer.lr={LR} "
                f"exp.seed={seed} "
                f"exp.train.total_epochs={EPOCHS} "
                f"exp.dataset.groupings='{TASK}' "
                f"exp.dataloader.batch_size={BATCH_SIZE} "
                f"exp.dataset.task_weights='{task_weights}' "
                f"exp.dataset.loss_based_task_weighting={use_loss_balanced} "
                f"exp.dataset.lbtw_alpha={lbtw_alpha} "
                f"exp.train.log_dir=\\'{os.path.join(LOG_DIR, job_name)}\\'"
            )
            job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_mtl_disjoint_tasks_train_group(args):
    ## DECLARE MACROS HERE ##

    if args.mtl_weighting == "static_equal":
        WD = 1e-2
        LR = 1e-4
        BATCH_SIZE = 32
    elif args.mtl_weighting == "static_delta":
        WD = 1e-3
        LR = 1e-3
        BATCH_SIZE = 32
    elif args.mtl_weighting == "dynamic":
        WD = 1e-2
        LR = 1e-4
        BATCH_SIZE = 32

    EPOCHS = 50
    SEED_GRID = [0, 1, 2]
    TASK = ["Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair", "Wearing_Lipstick:Male", "Gray_Hair:Young", "High_Cheekbones:Smiling"]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)
    method = "erm"

    task_weights, use_loss_balanced, lbtw_alpha = get_mtl_task_weights(args.mtl_weighting, TASK)

    for seed in SEED_GRID:
        job_name = f"mtl_train:{method},task:{len(TASK)}_tasks_{args.mtl_weighting}_task_weighting,seed:{seed},ckpt:group"
        log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

        if args.respawn:
            ckpt_dir = os.path.join(LOG_DIR, job_name, "ckpts")
            ckpt_path, ckpt_num = find_last_checkpoint(ckpt_dir)

            if ckpt_num != EPOCHS:

                command = (
                    f"python train_erm.py exp={method} "
                    f"exp.optimizer.weight_decay={WD} "
                    f"exp.optimizer.lr={LR} "
                    f"exp.seed={seed} "
                    f"exp.train.total_epochs={EPOCHS} "
                    f"exp.dataset.groupings='{TASK}' "
                    f"exp.dataloader.batch_size={BATCH_SIZE} "
                    f"exp.dataset.task_weights='{task_weights}' "
                    f"exp.dataset.loss_based_task_weighting={use_loss_balanced} "
                    f"exp.dataset.lbtw_alpha={lbtw_alpha} "
                    f"exp.train.load_ckpt=\\'{ckpt_path}\\' "
                    f"exp.train.log_dir=\\'{os.path.join(LOG_DIR, job_name)}\\'"
                )
                job_manager.submit(command, job_name=job_name, log_file=log_file)
                
        else:
            command = (
                f"python train_erm.py exp={method} "
                f"exp.optimizer.weight_decay={WD} "
                f"exp.optimizer.lr={LR} "
                f"exp.seed={seed} "
                f"exp.train.total_epochs={EPOCHS} "
                f"exp.dataset.groupings='{TASK}' "
                f"exp.dataloader.batch_size={BATCH_SIZE} "
                f"exp.dataset.task_weights='{task_weights}' "
                f"exp.dataset.loss_based_task_weighting={use_loss_balanced} "
                f"exp.dataset.lbtw_alpha={lbtw_alpha} "
                f"exp.train.log_dir=\\'{os.path.join(LOG_DIR, job_name)}\\'"
            )
            job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_mtl_suby_disjoint_tasks_train(args):
    WD = 1e-2
    LR = 1e-3
    BATCH_SIZE = 128
    EPOCHS = 60
    SEED_GRID = [0, 1, 2]
    TASK = ["Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair"]
    CVX_GRID = ["qp", "maxent"]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)
    task_weights, use_loss_balanced, lbtw_alpha = get_mtl_task_weights(args.mtl_weighting, TASK)

    for seed in SEED_GRID:
        for cvx in CVX_GRID:
            job_name = f"mtl_train:suby,task:{len(TASK)}_tasks_{args.mtl_weighting}_task_weighting,seed:{seed},cvx:{cvx}"
            log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

            if args.respawn:
                ckpt_dir = os.path.join(LOG_DIR, job_name, "ckpts")
                ckpt_path, ckpt_num = find_last_checkpoint(ckpt_dir)

                if ckpt_num != EPOCHS:
                    command = (
                        f"python train_erm.py exp=suby "
                        f"exp.optimizer.weight_decay={WD} "
                        f"exp.optimizer.lr={LR} "
                        f"exp.seed={seed} "
                        f"exp.train.total_epochs={EPOCHS} "
                        f"exp.dataset.groupings='{TASK}' "
                        f"exp.dataloader.batch_size={BATCH_SIZE} "
                        f"exp.dataset.cvx={cvx} "
                        f"exp.dataset.task_weights='{task_weights}' "
                        f"exp.dataset.loss_based_task_weighting={use_loss_balanced} "
                        f"exp.dataset.lbtw_alpha={lbtw_alpha} "
                        f"exp.train.load_ckpt=\\'{ckpt_path}\\' "
                        f"exp.train.log_dir=\\'{os.path.join(LOG_DIR, job_name)}\\'"
                    )
                    job_manager.submit(command, job_name=job_name, log_file=log_file)

            else:
                command = (
                    f"python train_erm.py exp=suby "
                    f"exp.optimizer.weight_decay={WD} "
                    f"exp.optimizer.lr={LR} "
                    f"exp.seed={seed} "
                    f"exp.train.total_epochs={EPOCHS} "
                    f"exp.dataset.groupings='{TASK}' "
                    f"exp.dataloader.batch_size={BATCH_SIZE} "
                    f"exp.dataset.cvx={cvx} "
                    f"exp.dataset.task_weights='{task_weights}' "
                    f"exp.dataset.loss_based_task_weighting={use_loss_balanced} "
                    f"exp.dataset.lbtw_alpha={lbtw_alpha} "
                    f"exp.train.log_dir=\\'{os.path.join(LOG_DIR, job_name)}\\'"
                )
                job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_mtl_rwy_disjoint_tasks_train(args):
    WD = 1e-2
    LR = 1e-4
    BATCH_SIZE = 2
    EPOCHS = 60
    SEED_GRID = [0, 1, 2]
    TASK = ["Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair"]
    CVX_GRID = ["qp", "maxent"]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)
    task_weights, use_loss_balanced, lbtw_alpha = get_mtl_task_weights(args.mtl_weighting, TASK)

    for seed in SEED_GRID:
        for cvx in CVX_GRID:
            job_name = f"mtl_train:rwy,task:{len(TASK)}_tasks_{args.mtl_weighting}_task_weighting,seed:{seed},cvx:{cvx}"
            log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

            if args.respawn:
                ckpt_dir = os.path.join(LOG_DIR, job_name, "ckpts")
                ckpt_path, ckpt_num = find_last_checkpoint(ckpt_dir)

                if ckpt_num != EPOCHS:
                    command = (
                        f"python train_erm.py exp=rwy "
                        f"exp.optimizer.weight_decay={WD} "
                        f"exp.optimizer.lr={LR} "
                        f"exp.seed={seed} "
                        f"exp.train.total_epochs={EPOCHS} "
                        f"exp.dataset.groupings='{TASK}' "
                        f"exp.dataloader.batch_size={BATCH_SIZE} "
                        f"exp.dataset.cvx={cvx} "
                        f"exp.dataset.task_weights='{task_weights}' "
                        f"exp.dataset.loss_based_task_weighting={use_loss_balanced} "
                        f"exp.dataset.lbtw_alpha={lbtw_alpha} "
                        f"exp.train.load_ckpt=\\'{ckpt_path}\\' "
                        f"exp.train.log_dir=\\'{os.path.join(LOG_DIR, job_name)}\\'"
                    )
                    job_manager.submit(command, job_name=job_name, log_file=log_file)

            else:
                command = (
                    f"python train_erm.py exp=rwy "
                    f"exp.optimizer.weight_decay={WD} "
                    f"exp.optimizer.lr={LR} "
                    f"exp.seed={seed} "
                    f"exp.train.total_epochs={EPOCHS} "
                    f"exp.dataset.groupings='{TASK}' "
                    f"exp.dataloader.batch_size={BATCH_SIZE} "
                    f"exp.dataset.cvx={cvx} "
                    f"exp.dataset.task_weights='{task_weights}' "
                    f"exp.dataset.loss_based_task_weighting={use_loss_balanced} "
                    f"exp.dataset.lbtw_alpha={lbtw_alpha} "
                    f"exp.train.log_dir=\\'{os.path.join(LOG_DIR, job_name)}\\'"
                )
                job_manager.submit(command, job_name=job_name, log_file=log_file)

def submit_mtl_disjoint_size_tasks_train_group(args):
    ## DECLARE MACROS HERE ##
    WD = 1e-2
    LR = 1e-4
    BATCH_SIZE = 32
    EPOCHS = 50
    SEED_GRID = [0, 1, 2]
    TASKS = [["Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair", "Wearing_Lipstick:Male"], ["Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair", "Wearing_Lipstick:Male", "Gray_Hair:Young"], ["Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair", "Wearing_Lipstick:Male", "Gray_Hair:Young", "High_Cheekbones:Smiling", "Brown_Hair:Wearing_Hat"]]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)
    method = "erm"

    for seed in SEED_GRID:
        for idx, task in enumerate(TASKS):
            task_weights, use_loss_balanced, lbtw_alpha = get_mtl_task_weights(args.mtl_weighting, task)
            job_name = f"mtl_train:{method},task:{len(task)}_tasks,disjoint_idx:{idx},{args.mtl_weighting}_task_weighting,seed:{seed}"
            log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

            if args.respawn:
                ckpt_dir = os.path.join(LOG_DIR, job_name, "ckpts")
                ckpt_path, ckpt_num = find_last_checkpoint(ckpt_dir)

                if ckpt_num != EPOCHS:

                    command = (
                        f"python train_erm.py exp={method} "
                        f"exp.optimizer.weight_decay={WD} "
                        f"exp.optimizer.lr={LR} "
                        f"exp.seed={seed} "
                        f"exp.train.total_epochs={EPOCHS} "
                        f"exp.dataset.groupings='{task}' "
                        f"exp.dataloader.batch_size={BATCH_SIZE} "
                        f"exp.dataset.task_weights='{task_weights}' "
                        f"exp.dataset.loss_based_task_weighting={use_loss_balanced} "
                        f"exp.dataset.lbtw_alpha={lbtw_alpha} "
                        f"exp.train.load_ckpt=\\'{ckpt_path}\\' "
                        f"exp.train.log_dir=\\'{os.path.join(LOG_DIR, job_name)}\\'"
                    )
                    job_manager.submit(command, job_name=job_name, log_file=log_file)
            else:
                command = (
                    f"python train_erm.py exp={method} "
                    f"exp.optimizer.weight_decay={WD} "
                    f"exp.optimizer.lr={LR} "
                    f"exp.seed={seed} "
                    f"exp.train.total_epochs={EPOCHS} "
                    f"exp.dataset.groupings='{task}' "
                    f"exp.dataloader.batch_size={BATCH_SIZE} "
                    f"exp.dataset.task_weights='{task_weights}' "
                    f"exp.dataset.loss_based_task_weighting={use_loss_balanced} "
                    f"exp.dataset.lbtw_alpha={lbtw_alpha} "
                    f"exp.train.log_dir=\\'{os.path.join(LOG_DIR, job_name)}\\'"
                )
                job_manager.submit(command, job_name=job_name, log_file=log_file)

def submit_mtl_nondisjoint_tasks_train_group(args):
    ## DECLARE MACROS HERE ##
    WD = 1e-2
    LR = 1e-4
    BATCH_SIZE = 32
    EPOCHS = 50
    SEED_GRID = [0, 1, 2]
    TASKS = [["Wearing_Earrings:Male", "Attractive:Male"], ["No_Beard:Heavy_Makeup", "Pointy_Nose:Heavy_Makeup"],
    ["Attractive:Gray_Hair", "Big_Nose:Gray_Hair"], ["Heavy_Makeup:Wearing_Lipstick", "No_Beard:Wearing_Lipstick"],
    ["Bangs:Wearing_Hat", "Blond_Hair:Wearing_Hat"]]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)
    method = "erm"

    for seed in SEED_GRID:
        for idx, task in enumerate(TASKS):
            task_weights, use_loss_balanced, lbtw_alpha = get_mtl_task_weights(args.mtl_weighting, task)
            job_name = f"mtl_train:{method},task:{len(task)}_tasks,nondisjoint_idx:{idx},{args.mtl_weighting}_task_weighting,seed:{seed}"
            log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

            if args.respawn:
                ckpt_dir = os.path.join(LOG_DIR, job_name, "ckpts")
                ckpt_path, ckpt_num = find_last_checkpoint(ckpt_dir)

                if ckpt_num != EPOCHS:

                    command = (
                        f"python train_erm.py exp={method} "
                        f"exp.optimizer.weight_decay={WD} "
                        f"exp.optimizer.lr={LR} "
                        f"exp.seed={seed} "
                        f"exp.train.total_epochs={EPOCHS} "
                        f"exp.dataset.groupings='{task}' "
                        f"exp.dataloader.batch_size={BATCH_SIZE} "
                        f"exp.dataset.task_weights='{task_weights}' "
                        f"exp.dataset.loss_based_task_weighting={use_loss_balanced} "
                        f"exp.dataset.lbtw_alpha={lbtw_alpha} "
                        f"exp.train.load_ckpt=\\'{ckpt_path}\\' "
                        f"exp.train.log_dir=\\'{os.path.join(LOG_DIR, job_name)}\\'"
                    )
                    job_manager.submit(command, job_name=job_name, log_file=log_file)
            else:
                command = (
                    f"python train_erm.py exp={method} "
                    f"exp.optimizer.weight_decay={WD} "
                    f"exp.optimizer.lr={LR} "
                    f"exp.seed={seed} "
                    f"exp.train.total_epochs={EPOCHS} "
                    f"exp.dataset.groupings='{task}' "
                    f"exp.dataloader.batch_size={BATCH_SIZE} "
                    f"exp.dataset.task_weights='{task_weights}' "
                    f"exp.dataset.loss_based_task_weighting={use_loss_balanced} "
                    f"exp.dataset.lbtw_alpha={lbtw_alpha} "
                    f"exp.train.log_dir=\\'{os.path.join(LOG_DIR, job_name)}\\'"
                )
                job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_mtl_semantic_similar_tasks_train_group(args):
    ## DECLARE MACROS HERE ##
    WD = 1e-2
    LR = 1e-4
    BATCH_SIZE = 32
    EPOCHS = 50
    SEED_GRID = [0, 1, 2]
    TASKS = [["Big_Nose:Wearing_Lipstick", "High_Cheekbones:Smiling"], ["Big_Lips:Goatee", "Wearing_Lipstick:Male"],
    ["Bags_Under_Eyes:Double_Chin", "High_Cheekbones:Rosy_Cheeks"], ["Blond_Hair:Wearing_Hat", "Brown_Hair:Wearing_Hat"]]


    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)
    method = "erm"

    for seed in SEED_GRID:
        for idx, task in enumerate(TASKS):
            task_weights, use_loss_balanced, lbtw_alpha = get_mtl_task_weights(args.mtl_weighting, task)
            job_name = f"mtl_train:{method},task:{len(task)}_tasks,semantic_similar:{idx + 1},{args.mtl_weighting}_task_weighting,seed:{seed}"
            log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

            if args.respawn:
                ckpt_dir = os.path.join(LOG_DIR, job_name, "ckpts")
                ckpt_path, ckpt_num = find_last_checkpoint(ckpt_dir)

                if ckpt_num != EPOCHS:

                    command = (
                        f"python train_erm.py exp={method} "
                        f"exp.optimizer.weight_decay={WD} "
                        f"exp.optimizer.lr={LR} "
                        f"exp.seed={seed} "
                        f"exp.train.total_epochs={EPOCHS} "
                        f"exp.dataset.groupings='{task}' "
                        f"exp.dataloader.batch_size={BATCH_SIZE} "
                        f"exp.dataset.task_weights='{task_weights}' "
                        f"exp.dataset.loss_based_task_weighting={use_loss_balanced} "
                        f"exp.dataset.lbtw_alpha={lbtw_alpha} "
                        f"exp.train.load_ckpt=\\'{ckpt_path}\\' "
                        f"exp.train.log_dir=\\'{os.path.join(LOG_DIR, job_name)}\\'"
                    )
                    job_manager.submit(command, job_name=job_name, log_file=log_file)
            else:
                command = (
                    f"python train_erm.py exp={method} "
                    f"exp.optimizer.weight_decay={WD} "
                    f"exp.optimizer.lr={LR} "
                    f"exp.seed={seed} "
                    f"exp.train.total_epochs={EPOCHS} "
                    f"exp.dataset.groupings='{task}' "
                    f"exp.dataloader.batch_size={BATCH_SIZE} "
                    f"exp.dataset.task_weights='{task_weights}' "
                    f"exp.dataset.loss_based_task_weighting={use_loss_balanced} "
                    f"exp.dataset.lbtw_alpha={lbtw_alpha} "
                    f"exp.train.log_dir=\\'{os.path.join(LOG_DIR, job_name)}\\'"
                )
                job_manager.submit(command, job_name=job_name, log_file=log_file)

def submit_mtl_strong_spurious_correlations_tasks_train_group(args):
    ## DECLARE MACROS HERE ##
    WD = 1e-2
    LR = 1e-4
    BATCH_SIZE = 32
    EPOCHS = 50
    SEED_GRID = [0, 1, 2]
    TASKS = [["Wearing_Lipstick:Male", "High_Cheekbones:Smiling"], ["Heavy_Makeup:Male", "Wearing_Earrings:Male"]]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)
    method = "erm"

    for seed in SEED_GRID:
        for idx, task in enumerate(TASKS):
            task_weights, use_loss_balanced, lbtw_alpha = get_mtl_task_weights(args.mtl_weighting, task)
            job_name = f"mtl_train:{method},task:{len(task)}_tasks,strong_spurious_correlation_idx:{idx},{args.mtl_weighting}_task_weighting,seed:{seed}"
            log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

            if args.respawn:
                ckpt_dir = os.path.join(LOG_DIR, job_name, "ckpts")
                ckpt_path, ckpt_num = find_last_checkpoint(ckpt_dir)

                if ckpt_num != EPOCHS:

                    command = (
                        f"python train_erm.py exp={method} "
                        f"exp.optimizer.weight_decay={WD} "
                        f"exp.optimizer.lr={LR} "
                        f"exp.seed={seed} "
                        f"exp.train.total_epochs={EPOCHS} "
                        f"exp.dataset.groupings='{task}' "
                        f"exp.dataloader.batch_size={BATCH_SIZE} "
                        f"exp.dataset.task_weights='{task_weights}' "
                        f"exp.dataset.loss_based_task_weighting={use_loss_balanced} "
                        f"exp.dataset.lbtw_alpha={lbtw_alpha} "
                        f"exp.train.load_ckpt=\\'{ckpt_path}\\' "
                        f"exp.train.log_dir=\\'{os.path.join(LOG_DIR, job_name)}\\'"
                    )
                    job_manager.submit(command, job_name=job_name, log_file=log_file)
            else:
                command = (
                    f"python train_erm.py exp={method} "
                    f"exp.optimizer.weight_decay={WD} "
                    f"exp.optimizer.lr={LR} "
                    f"exp.seed={seed} "
                    f"exp.train.total_epochs={EPOCHS} "
                    f"exp.dataset.groupings='{task}' "
                    f"exp.dataloader.batch_size={BATCH_SIZE} "
                    f"exp.dataset.task_weights='{task_weights}' "
                    f"exp.dataset.loss_based_task_weighting={use_loss_balanced} "
                    f"exp.dataset.lbtw_alpha={lbtw_alpha} "
                    f"exp.train.log_dir=\\'{os.path.join(LOG_DIR, job_name)}\\'"
                )
                job_manager.submit(command, job_name=job_name, log_file=log_file)

def submit_mtl_weak_spurious_correlations_tasks_train_group(args):
    ## DECLARE MACROS HERE ##
    WD = 1e-2
    LR = 1e-4
    BATCH_SIZE = 32
    EPOCHS = 50
    SEED_GRID = [0, 1, 2]
    TASKS = [["Big_Lips:Chubby", "Young:Chubby"], ["High_Cheekbones:Rosy_Cheeks", "Brown_Hair:Wearing_Hat"]]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)
    method = "erm"

    for seed in SEED_GRID:
        for idx, task in enumerate(TASKS):
            task_weights, use_loss_balanced, lbtw_alpha = get_mtl_task_weights(args.mtl_weighting, task)
            job_name = f"mtl_train:{method},task:{len(task)}_tasks,weak_spurious_correlation_idx:{idx},{args.mtl_weighting}_task_weighting,seed:{seed}"
            log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

            if args.respawn:
                ckpt_dir = os.path.join(LOG_DIR, job_name, "ckpts")
                ckpt_path, ckpt_num = find_last_checkpoint(ckpt_dir)

                if ckpt_num != EPOCHS:

                    command = (
                        f"python train_erm.py exp={method} "
                        f"exp.optimizer.weight_decay={WD} "
                        f"exp.optimizer.lr={LR} "
                        f"exp.seed={seed} "
                        f"exp.train.total_epochs={EPOCHS} "
                        f"exp.dataset.groupings='{task}' "
                        f"exp.dataloader.batch_size={BATCH_SIZE} "
                        f"exp.dataset.task_weights='{task_weights}' "
                        f"exp.dataset.loss_based_task_weighting={use_loss_balanced} "
                        f"exp.dataset.lbtw_alpha={lbtw_alpha} "
                        f"exp.train.load_ckpt=\\'{ckpt_path}\\' "
                        f"exp.train.log_dir=\\'{os.path.join(LOG_DIR, job_name)}\\'"
                    )
                    job_manager.submit(command, job_name=job_name, log_file=log_file)
            else:
                command = (
                    f"python train_erm.py exp={method} "
                    f"exp.optimizer.weight_decay={WD} "
                    f"exp.optimizer.lr={LR} "
                    f"exp.seed={seed} "
                    f"exp.train.total_epochs={EPOCHS} "
                    f"exp.dataset.groupings='{task}' "
                    f"exp.dataloader.batch_size={BATCH_SIZE} "
                    f"exp.dataset.task_weights='{task_weights}' "
                    f"exp.dataset.loss_based_task_weighting={use_loss_balanced} "
                    f"exp.dataset.lbtw_alpha={lbtw_alpha} "
                    f"exp.train.log_dir=\\'{os.path.join(LOG_DIR, job_name)}\\'"
                )
                job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_mtl_jtt_train(args):
    EPOCHS = 50
    SEED_GRID = [0, 1, 2]
    TASK = ["Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair"]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)

    for seed in SEED_GRID:
        for mtl_weighting in ["static_equal", "static_delta", "dynamic"]:
            task_weights, use_loss_balanced, lbtw_alpha = get_mtl_task_weights(mtl_weighting, TASK)

            job_name = f"mtl_train:jtt,task:{len(TASK)}_tasks_{mtl_weighting}_task_weighting,seed:{seed}"
            log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

            ckpt_dir = os.path.join(LOG_DIR, "mtl_jtt", job_name, "stage_2", "ckpts")
            mtl_jtt_dir = os.path.join(LOG_DIR, "mtl_jtt")
            if os.path.exists(ckpt_dir):
                if os.path.exists(os.path.join(ckpt_dir, "ckpt.last.pt")):
                    print(f"{ckpt_dir} exists. Skipping.")
                    continue

                ckpt_path, ckpt_num = find_last_checkpoint(ckpt_dir)
                load_up_pkl = os.path.join(LOG_DIR, "mtl_jtt", job_name, "jtt_error_set_inv.pkl")

                if ckpt_num != EPOCHS:
                    command = (
                        "python train_jtt.py exp=jtt "
                        f"exp.groupings='{TASK}' "
                        f"exp.seed={seed} "
                        f"exp.task_weights='{task_weights}' "
                        f"exp.loss_based_task_weighting={use_loss_balanced} "
                        f"exp.lbtw_alpha={lbtw_alpha} "
                        f"exp.load_stage_2_ckpt=\\'{ckpt_path}\\' "
                        f"exp.load_up_pkl=\\'{load_up_pkl}\\' "
                        f"exp.log_dir=\\'{os.path.join(mtl_jtt_dir, job_name)}\\'"
                    )
            else:
                command = (
                    "python train_jtt.py exp=jtt "
                    f"exp.groupings='{TASK}' "
                    f"exp.seed={seed} "
                    f"exp.task_weights='{task_weights}' "
                    f"exp.loss_based_task_weighting={use_loss_balanced} "
                    f"exp.lbtw_alpha={lbtw_alpha} "
                    f"exp.log_dir=\\'{os.path.join(mtl_jtt_dir, job_name)}\\'"
                )

            job_manager.submit(command, job_name=job_name, log_file=log_file)


def main():
    args = parse_args()
    if args.mode == "sbatch":
        os.makedirs(args.slurm_logs, exist_ok=True)

    if args.opt == "erm":
        submit_erm_train(args)
    elif args.opt == "rw_sub":
        submit_reweighted_subsampled_train(args)
    elif args.opt == "erm_baseline":
        submit_erm_baseline_disjoint_tasks_train(args)
    elif args.opt == "suby_baseline":
        submit_suby_baseline_disjoint_tasks_train(args)
    elif args.opt == "rwy_baseline":
        submit_rwy_baseline_disjoint_tasks_train(args)
    elif args.opt == "jtt_baseline":
        submit_jtt_baseline_disjoint_tasks_train(args)
    elif args.opt == "mtl_disjoint_tuning":
        submit_mtl_disjoint_tasks_tune(args)
    elif args.opt == "mtl_disjoint_avg_train":
        submit_mtl_disjoint_tasks_train_avg(args)
    elif args.opt == "mtl_disjoint_group_train":
        submit_mtl_disjoint_tasks_train_group(args)
    elif args.opt == "mtl_suby":
        submit_mtl_suby_disjoint_tasks_train(args)
    elif args.opt == "mtl_rwy":
        submit_mtl_rwy_disjoint_tasks_train(args)
    elif args.opt == "mtl_nondisjoint_group_train":
        submit_mtl_nondisjoint_tasks_train_group(args)
    elif args.opt == "mtl_disjoint_group_size_train":
        submit_mtl_disjoint_size_tasks_train_group(args)
    elif args.opt == "mtl_semantic_similar_train":
        submit_mtl_semantic_similar_tasks_train_group(args)
    elif args.opt == "mtl_strong_spurious_correlation_train":
        submit_mtl_strong_spurious_correlations_tasks_train_group(args)
    elif args.opt == "mtl_weak_spurious_correlation_train":
        submit_mtl_weak_spurious_correlations_tasks_train_group(args)
    elif args.opt == "mtl_jtt_train":
        submit_mtl_jtt_train(args)
    else:
        raise ValueError(f"Didn't recognize opt={args.opt}. Did you forget to add a check for this function?")


if __name__ == "__main__":
    main()
