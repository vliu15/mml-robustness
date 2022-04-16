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
import json
import os
import re

import numpy as np

from scripts.job_manager import JobManager

# RICE MACROS
USER = os.environ["USER"]
#LOG_DIR = f"/farmshare/user_data/{USER}/mml-robustness/logs"
LOG_DIR = "./logs"


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
    parser.add_argument(
        "--spurious_eval_dir",
        type=str,
        default="./",
        required=False,
        help="The folder which contains all results from spurious id"
    )
    args = parser.parse_args()

    # Convert relative paths to absolute paths to help slurm out
    args.slurm_logs = os.path.abspath(args.slurm_logs)
    return args


def get_mtl_task_weights(args, task_pairing, alpha=0.5):

    if args.mtl_weighting == "static_equal":
        task_weights = [1] * len(task_pairing)
        use_loss_balanced = "false"
        lbtw_alpha = 0

        return task_weights, use_loss_balanced, lbtw_alpha

    elif args.mtl_weighting == "static_delta":
        use_loss_balanced = "false"
        lbtw_alpha = 0

        ## get delta value for each task:attribute pair
        deltas = []
        for grouping in task_pairing:
            task = grouping.split(":")[0]
            attribute = grouping.split(":")[1]
            with open(os.path.join(args.spurious_eval_dir, task, f"{task}_spurious_eval.json"), "r") as f:
                data = json.load(f)
                deltas.append(data[attribute])

        ## compute delta heuristic, scale by 100 to get back into a range that won't blow up in softmax
        deltas_np = np.array(deltas) / 100
        task_weights_np = np.exp(deltas_np) / np.sum(np.exp(deltas_np))
        task_weights = list(task_weights_np)

        return task_weights, use_loss_balanced, lbtw_alpha
    elif args.mtl_weighting == "dynamic":
        task_weights = [1] * len(task_pairing)
        use_loss_balanced = "true"
        lbtw_alpha = alpha

        return task_weights, use_loss_balanced, lbtw_alpha


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
    TASK_GRID = [
        "Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair", "Goatee:No_Beard", "Gray_Hair:Young", "High_Cheekbones:Smiling",
        "Wavy_Hair:Straight_Hair", "Wearing_Lipstick:Male"
    ]

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
    WD_GRID = [1e-4, 1e-3, 1e-2, 1e-1] #1e-4, 1e-3, 1e-2
    LR_GRID = [1e-5, 1e-4, 1e-3]
    BATCH_SIZE_GRID = [32, 64, 128]
    EPOCHS = 50
    SEED_GRID = [0]
    TASK = ["Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair"]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)
    method = "erm"

    task_weights, use_loss_balanced, lbtw_alpha = get_mtl_task_weights(args, TASK)

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
    TASK = ["Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair"]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)
    method = "erm"

    task_weights, use_loss_balanced, lbtw_alpha = get_mtl_task_weights(args, TASK)

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
                ob_manager.submit(command, job_name=job_name, log_file=log_file)
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
    TASK = ["Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair"]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)
    method = "erm"

    task_weights, use_loss_balanced, lbtw_alpha = get_mtl_task_weights(args, TASK)

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
                ob_manager.submit(command, job_name=job_name, log_file=log_file)
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
    else:
        raise ValueError(f"Didn't recognize opt={args.opt}. Did you forget to add a check for this function?")


if __name__ == "__main__":
    main()
