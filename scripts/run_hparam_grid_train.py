"""
Handy script to run jobs for hyperparameter grid searches

Sample usage:
python -m scripts.run_hparam_grid_train \
    --template ./scripts/sbatch_template_rice.sh \
    --mode sbatch \
    --slurm_logs ./slurm_logs \
    --opt erm

For runs on any device that has a time limit per job, once the jobs for a given run finish please 
re-run the script with the --respawn flag to ensure that all jobs completed all checkpoints
"""

import argparse
import os
import re
import warnings

from mtl.utils import get_mtl_task_weights
from scripts.const import GRIDS, PARAMS, TASKS
from scripts.job_manager import JobManager

USER = os.environ["USER"]
LOG_DIR = "./new_logs"
SEED_GRID = [0, 1, 2] #, 1, 2]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template", type=str, default="scripts/sbatch_template_rice.sh", required=False, help="SBATCH template file"
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

    parser.add_argument(
        "--model", type=str, default="resnet50", choices=["resnet50", "clip_resnet50"], help="Name of model to use"
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

    # Raise warnings here in case some flags aren't implemented for some methods
    warnings.warn(
        "The --model arg is only currently used for --opt=[erm_tune,suby_tune,erm_id,suby_id] options.",
        category=FutureWarning
    )
    warnings.warn(
        "The `clip_erm` hparams in `PARAMS` is not implemented into any of the functions in this file yet.",
        category=FutureWarning
    )

    return args


def find_last_checkpoint(ckpt_dir):
    """Finds the most recent checkpoint in the checkpoint directory."""
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


def flatten(t):
    """Flattens a nested list into a flat list"""
    return [item for sublist in t for item in sublist]


def append_ckpt_for_respawn(command, job_name, epochs):
    ckpt_dir = os.path.join(LOG_DIR, job_name, "ckpts")
    if not os.path.exists(ckpt_dir):
        warnings.warn(f"Skipping respawn ... No existing checkpoint found for: {job_name}")
        return command
    ckpt_path, ckpt_num = find_last_checkpoint(ckpt_dir)
    if ckpt_num is None:
        return command

    if ckpt_num < epochs:
        return f"{command} exp.train.load_ckpt=\\'{ckpt_path}\\'"
    else:
        warnings.warn(f"All checkpoints already completed for: {job_name}")
        return command


def append_ckpt_for_jtt_respawn(command, job_name, epochs1, epochs2):
    stage_1_ckpt_dir = os.path.join(LOG_DIR, job_name, "stage_1", "ckpts")
    if not os.path.exists(stage_1_ckpt_dir):  # no stage 1 ckpts mean we haven't trained at all yet
        warnings.warn(f"Skipping respawn ... No existing checkpoint found for: {job_name}")
        return command
    stage_1_ckpt_path, stage_1_ckpt_num = find_last_checkpoint(stage_1_ckpt_dir)

    stage_2_ckpt_dir = os.path.join(LOG_DIR, job_name, "stage_2", "ckpts")
    if stage_1_ckpt_num == epochs1 and os.path.exists(stage_2_ckpt_dir):
        stage_2_ckpt_path, stage_2_ckpt_num = find_last_checkpoint(stage_2_ckpt_dir)
    else:
        stage_2_ckpt_path, stage_2_ckpt_num = "null", "null"

    load_up_pkl_path = os.path.join(LOG_DIR, job_name, f"jtt_error_set_uniform.pkl")  # default is inv merging, even for stl

    if stage_1_ckpt_num != epochs1 or (stage_1_ckpt_num == epochs1 and not os.path.exists(load_up_pkl_path)):
        return f"{command} exp.load_stage_1_ckpt=\\'{stage_1_ckpt_path}\\'"
    elif stage_1_ckpt_num == epochs1 and stage_2_ckpt_num is None:
        return f"{command} exp.load_up_pkl=\\'{load_up_pkl_path}\\'"
    elif stage_1_ckpt_num == epochs1 and stage_2_ckpt_num != epochs2:
        return f"{command} exp.load_up_pkl=\\'{load_up_pkl_path}\\' exp.load_stage_2_ckpt=\\'{stage_2_ckpt_path}\\'"
    else:
        warnings.warn(f"All checkpoints already completed for: {job_name}")
        return command


def submit_stl_tune_train(args):
    """Tunes STL methods on tasks (for spurious ID). Otherwise, we can use default hparams from previous papers."""

    TASK_GRID = TASKS["SPURIOUS_ID_TUNE"]
    EPOCHS = 25

    assert args.opt.endswith("_tune"), "This method should only be called with --opt=.*_tune"
    method = args.opt.replace("_tune", "")
    assert method in ["erm", "suby"]  # just to double check

    WD_GRID = GRIDS[method]["WD"]
    LR_GRID = GRIDS[method]["LR"]
    BATCH_SIZE_GRID = GRIDS[method]["BATCH_SIZE"]

    # Support different naming conventions from earlier in the codebase
    # NOTE that the old naming conventions don't include the method in the name ...
    if method == "erm":
        job_name_generator = lambda task, wd, lr, batch_size: f"task:{task},wd:{wd},lr:{lr}"
    elif method == "suby":
        job_name_generator = lambda task, wd, lr, batch_size: f"task:{task},wd:{wd},lr:{lr},batch_size:{batch_size}"
    else:
        raise ValueError("Only --opt=erm and --opt=suby are supported in this function.")

    if args.respawn:
        warnings.warn("--respawn is not implemented for this function.", category=FutureWarning)

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)

    for task in TASK_GRID:
        for wd in WD_GRID:
            for lr in LR_GRID:
                for batch_size in BATCH_SIZE_GRID:
                    job_name = job_name_generator(task, wd, lr, batch_size)
                    log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

                    log_dir = os.path.join(LOG_DIR, f"{args.model}_tune", job_name)
                    command = (
                        f"python train_erm.py exp={method} "
                        f"exp.model.name={args.model} "
                        f"exp.optimizer.weight_decay={wd} "
                        f"exp.optimizer.lr={lr} "
                        f"exp.dataloader.batch_size={batch_size} "
                        f"exp.train.total_epochs={EPOCHS} "
                        f"exp.dataset.groupings='[{task}]' "
                        f"exp.train.log_dir=\\'{log_dir}\\'"
                    )
                    job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_spurious_id_train(args):
    ATTRIBUTES = TASKS["SPURIOUS_ID_ALL"]

    assert args.opt.endswith("_id"), "This method should only be called with --opt=.*_id"
    method = args.opt.replace("_id", "")
    assert method in ["erm", "suby"]  # just to double check

    # HACK(vliu): we only use CLIP models for spurious ID, so it's not worth refactoring PARAMS dict right now
    if args.model == "resnet50":
        wd = PARAMS[method]["WD"]
        lr = PARAMS[method]["LR"]
        batch_size = PARAMS[method]["BATCH_SIZE"]
        epochs = PARAMS[method]["EPOCHS"]
    elif args.model == "clip_resnet50":
        wd = PARAMS[f"clip_{method}"]["WD"]
        lr = PARAMS[f"clip_{method}"]["LR"]
        batch_size = PARAMS[f"clip_{method}"]["BATCH_SIZE"]
        epochs = PARAMS[f"clip_{method}"]["EPOCHS"]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)
    for attribute in ATTRIBUTES:
        job_name = f"task:{attribute},wd:{wd},lr:{lr}"
        log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

        log_dir = os.path.join(LOG_DIR, f"{args.model}_id", job_name)
        command = (
            "python train_erm.py exp.dataset.subgroup_labels=False "
            f"exp={method} "
            f"exp.model.name={args.model} "
            f"exp.dataset.groupings='[{attribute}:Blond_Hair]' "
            f"exp.dataloader.batch_size={batch_size} "
            f"exp.optimizer.lr={lr} "
            f"exp.optimizer.weight_decay={wd} "
            f"exp.train.total_epochs={epochs} "
            f"exp.train.log_dir=\\'{log_dir}\\'"
        )
        if args.respawn:
            command = append_ckpt_for_respawn(command, os.path.join(f"{args.model}_id", job_name), epochs)

        job_manager.submit(command, job_name=job_name, log_file=log_file)

def submit_spurious_id_train_cxr(args):
    CXR_ATTRIBUTES = TASKS["SPURIOUS_ID_ALL_CXR"]

    assert args.opt.endswith("_cxr_id"), "This method should only be called with --opt=.*_id"
    method = args.opt.replace("_cxr_id", "")
    assert method in ["erm", "suby"]  # just to double check

    # HACK(vliu): we only use CLIP models for spurious ID, so it's not worth refactoring PARAMS dict right now
    if args.model == "resnet50":
        wd = PARAMS[method]["WD"]
        lr = PARAMS[method]["LR"]
        batch_size = PARAMS[method]["BATCH_SIZE"]
        epochs = PARAMS[method]["EPOCHS"]
    elif args.model == "clip_resnet50":
        wd = PARAMS[f"clip_{method}"]["WD"]
        lr = PARAMS[f"clip_{method}"]["LR"]
        batch_size = PARAMS[f"clip_{method}"]["BATCH_SIZE"]
        epochs = PARAMS[f"clip_{method}"]["EPOCHS"]
    dataset_name = "chestxray8"
    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)
    for attribute in CXR_ATTRIBUTES:
        job_name = f"method:{method},task:{attribute},wd:{wd},lr:{lr}"
        log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

        log_dir = os.path.join(LOG_DIR, f"{args.model}_id", job_name)
        command = (
            "python train_erm.py exp.dataset.subgroup_labels=False "
            f"exp={method} "
            f"exp.model.name={args.model} "
            f"exp.dataset.groupings='[{attribute}:Old]' "
            f"exp.dataloader.batch_size={batch_size} "
            f"exp.optimizer.lr={lr} "
            f"exp.dataset.name='{dataset_name}' "
            f"exp.optimizer.weight_decay={wd} "
            f"exp.train.total_epochs={epochs} "
            f"exp.train.log_dir=\\'{log_dir}\\'"
        )
        if args.respawn:
            command = append_ckpt_for_respawn(command, os.path.join(f"{args.model}_id", job_name), epochs)

        job_manager.submit(command, job_name=job_name, log_file=log_file)



def submit_stl_train(args):
    method = args.opt
    assert method in ["erm", "suby", "rwy", "jtt"]  # just to double check

    if method in ["suby", "rwy", "jtt"]:
        TASK_GRID = ["Blond_Hair:Male"] #flatten(TASKS["MTL_DISJOINT"])
        #TASK_GRID = ["High_Cheekbones:Smiling"]

    else:  # we want to run erm on everything
        TASK_GRID = set(flatten(TASKS["MTL_DISJOINT"] + TASKS["MTL_NONDISJOINT"] + TASKS["MTL_SIMILAR"] + TASKS["MTL_STRONG"]))

    WD = PARAMS[method]["WD"]
    LR = PARAMS[method]["LR"]
    BATCH_SIZE = PARAMS[method]["BATCH_SIZE"]
    EPOCHS = PARAMS[method]["EPOCHS"]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)

    for task in TASK_GRID:
        for seed in SEED_GRID:
            job_name = f"baseline:{method},task:{task},seed:{seed}"
            log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

            if method != "jtt":
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
                if args.respawn:
                    command = append_ckpt_for_respawn(command, job_name, EPOCHS)

            # Different train script for JTT
            else:
                T = PARAMS[method]["T"]
                LAM_UP = PARAMS[method]["LAM_UP"]
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
                if args.respawn:
                    command = append_ckpt_for_jtt_respawn(command, job_name, T, EPOCHS)

            job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_mtl_tune_train(args):
    TASK = TASKS["MTL_DISJOINT"][0]  # just tune on the first pair of disjoint tasks

    assert args.opt.startswith("mtl_") and args.opt.endswith("_tune"), \
        "This method should only be called with --opt=mtl_.*_tune"
    mtl_method = args.opt.replace("_tune", "")
    assert mtl_method in ["mtl_erm", "mtl_suby"]  # just to double check
    method = mtl_method.replace("mtl_", "")

    WD_GRID = GRIDS[mtl_method]["WD"]
    LR_GRID = GRIDS[mtl_method]["LR"]
    BATCH_SIZE_GRID = GRIDS[mtl_method]["BATCH_SIZE"]

    EPOCHS = 50

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)
    task_weights, use_loss_balanced, lbtw_alpha = get_mtl_task_weights(args.mtl_weighting, TASK)

    for wd in WD_GRID:
        for lr in LR_GRID:
            for batch_size in BATCH_SIZE_GRID:
                for seed in SEED_GRID:
                    job_name = f"mtl_tuning:{method},task:{len(TASK)}_tasks,{args.mtl_weighting}_task_weighting,seed:{seed},wd:{wd},lr:{lr},batch_size:{batch_size}"
                    log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

                    if os.path.exists(os.path.join(LOG_DIR, job_name, "ckpts", "ckpt.last.pt")):
                        continue

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
                    if args.respawn:
                        command = append_ckpt_for_respawn(command, job_name, EPOCHS)

                    job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_mtl_stl_erm_train(args):
    TASK_GRID = TASKS["MTL_STL_COMPARISON"]

    assert args.opt in ["mtl_erm_mtl_stl"]
    mtl_method = args.opt.replace("_mtl_stl", "")
    method = mtl_method.replace("mtl_", "")

    key = f"{mtl_method}_group_ckpt_{args.mtl_weighting}_mtl_weighting"
    WD = PARAMS[key]["WD"]
    LR = PARAMS[key]["LR"]
    BATCH_SIZE = PARAMS[key]["BATCH_SIZE"]
    EPOCHS = PARAMS[key]["EPOCHS"]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)

    for seed in SEED_GRID:
        for idx, task in enumerate(TASK_GRID):
            task_weights, use_loss_balanced, lbtw_alpha = get_mtl_task_weights(args.mtl_weighting, task)

            job_name = f"mtl_train:{method},task:{len(task)}_tasks,task_mtl_stl_idx:{idx + 1},{args.mtl_weighting}_task_weighting,seed:{seed}"
            log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

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
            if args.respawn:
                command = append_ckpt_for_respawn(command, job_name, EPOCHS)

            job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_mtl_erm_ablate_disjoint_tasks_train(args):
    TASK_GRID = TASKS["MTL_ABLATE_DISJOINT"]

    assert args.opt in ["mtl_erm_ablate_disjoint"]
    mtl_method = args.opt.replace("_ablate_disjoint", "")
    method = mtl_method.replace("mtl_", "")

    key = f"{mtl_method}_group_ckpt_{args.mtl_weighting}_mtl_weighting"
    WD = PARAMS[key]["WD"]
    LR = PARAMS[key]["LR"]
    BATCH_SIZE = PARAMS[key]["BATCH_SIZE"]
    EPOCHS = PARAMS[key]["EPOCHS"]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)

    for seed in SEED_GRID:
        for idx, task in enumerate(TASK_GRID):
            task_weights, use_loss_balanced, lbtw_alpha = get_mtl_task_weights(args.mtl_weighting, task)

            job_name = f"mtl_train:{method},task:{len(task)}_tasks,task_ablation_disjoint_idx:{idx + 1},{args.mtl_weighting}_task_weighting,seed:{seed}"
            log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

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
            if args.respawn:
                command = append_ckpt_for_respawn(command, job_name, EPOCHS)

            job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_mtl_erm_ablate_nondisjoint_tasks_train(args):
    TASK_GRID = TASKS["MTL_ABLATE_NONDISJOINT"][1:]  # MULTIPLE PAIRS

    assert args.opt in ["mtl_erm_ablate_nondisjoint"]

    mtl_method = args.opt.replace("_ablate_nondisjoint", "")
    method = mtl_method.replace("mtl_", "")

    key = f"{mtl_method}_group_ckpt_{args.mtl_weighting}_mtl_weighting"
    WD = PARAMS[key]["WD"]
    LR = PARAMS[key]["LR"]
    BATCH_SIZE = PARAMS[key]["BATCH_SIZE"]
    EPOCHS = PARAMS[key]["EPOCHS"]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)

    for seed in SEED_GRID:
        for idx, task in enumerate(TASK_GRID):
            task_weights, use_loss_balanced, lbtw_alpha = get_mtl_task_weights(args.mtl_weighting, task)

            job_name = f"mtl_train:{method},task:{len(task)}_tasks,task_ablation_nondisjoint_idx:{idx},{args.mtl_weighting}_task_weighting,seed:{seed}"
            log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

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
            if args.respawn:
                command = append_ckpt_for_respawn(command, job_name, EPOCHS)

            job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_mtl_cvx_disjoint_tasks_train(args):
    TASK_GRID = TASKS["MTL_STL_COMPARISON"]  # SINGLE PAIR
    CVX_GRID = ["qp", "maxent"]

    assert args.opt in ["mtl_rwy", "mtl_suby"], "This method only supports --opt=mtl_rwy and --opt=mtl_suby"
    method = args.opt.replace("mtl_", "")

    WD = PARAMS[args.opt]["WD"]
    LR = PARAMS[args.opt]["LR"]
    BATCH_SIZE = PARAMS[args.opt]["BATCH_SIZE"]
    EPOCHS = PARAMS[args.opt]["EPOCHS"]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)

    for idx, task in enumerate(TASK_GRID):
        for seed in SEED_GRID:
            for cvx in CVX_GRID:
                task_weights, use_loss_balanced, lbtw_alpha = get_mtl_task_weights(args.mtl_weighting, task)
                job_name = f"mtl_train:{method},task:{len(task)}_tasks_idx:{idx+1},{args.mtl_weighting}_task_weighting,seed:{seed},cvx:{cvx}"
                log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

                command = (
                    f"python train_erm.py exp={method} "
                    f"exp.optimizer.weight_decay={WD} "
                    f"exp.optimizer.lr={LR} "
                    f"exp.seed={seed} "
                    f"exp.train.total_epochs={EPOCHS} "
                    f"exp.dataset.groupings='{task}' "
                    f"exp.dataloader.batch_size={BATCH_SIZE} "
                    f"exp.dataset.cvx={cvx} "
                    f"exp.dataset.task_weights='{task_weights}' "
                    f"exp.dataset.loss_based_task_weighting={use_loss_balanced} "
                    f"exp.dataset.lbtw_alpha={lbtw_alpha} "
                    f"exp.train.log_dir=\\'{os.path.join(LOG_DIR, job_name)}\\'"
                )
                if args.respawn:
                    command = append_ckpt_for_respawn(command, job_name, EPOCHS)

                job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_mtl_jtt_disjoint_tasks_train(args):
    TASK_GRID = TASKS["MTL_STL_COMPARISON"]  # SINGLE PAIR

    assert args.opt in ["mtl_jtt"], "This method only supports --opt=mtl_jtt"

    WD = PARAMS[args.opt]["WD"]
    LR = PARAMS[args.opt]["LR"]
    BATCH_SIZE = PARAMS[args.opt]["BATCH_SIZE"]
    EPOCHS = PARAMS[args.opt]["EPOCHS"]

    T = PARAMS["jtt"]["T"]
    LAM_UP = PARAMS["jtt"]["LAM_UP"]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)

    for idx, task in enumerate(TASK_GRID):
        for seed in SEED_GRID:
            task_weights, use_loss_balanced, lbtw_alpha = get_mtl_task_weights(args.mtl_weighting, task)
            job_name = f"mtl_train:jtt,task:{len(task)}_tasks_idx:{idx + 1},{args.mtl_weighting}_task_weighting,seed:{seed},uniform"
            log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

            command = (
                f"python train_jtt.py exp=jtt "
                f"exp.weight_decay={WD} "
                f"exp.lr={LR} "
                f"exp.seed={seed} "
                f"exp.task_weights='{task_weights}' "
                f"exp.loss_based_task_weighting={use_loss_balanced} "
                f"exp.lbtw_alpha={lbtw_alpha} "
                f"exp.epochs_stage_1={T} "
                f"exp.epochs_stage_2={EPOCHS} "
                f"exp.groupings='{task}' "
                f"exp.lambda_up={LAM_UP} "
                f"exp.batch_size={BATCH_SIZE} "
                f"exp.log_dir=\\'{os.path.join(LOG_DIR, job_name)}\\'"
            )
            if args.respawn:
                command = append_ckpt_for_jtt_respawn(command, job_name, T, EPOCHS)

            job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_mtl_erm_disjoint_tasks_train(args):
    TASK_GRID = TASKS["MTL_DISJOINT"]  # all, but we need the pair indices, so we just skip the 0th one

    assert args.opt in ["mtl_erm_disjoint"]
    mtl_method = args.opt.replace("_disjoint", "")
    method = mtl_method.replace("mtl_", "")

    key = f"{mtl_method}_group_ckpt_{args.mtl_weighting}_mtl_weighting"
    WD = PARAMS[key]["WD"]
    LR = PARAMS[key]["LR"]
    BATCH_SIZE = PARAMS[key]["BATCH_SIZE"]
    EPOCHS = PARAMS[key]["EPOCHS"]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)

    for seed in SEED_GRID:
        for idx, task in enumerate(TASK_GRID):
            if idx == 0:
                continue  # skip the one we tuned on

            task_weights, use_loss_balanced, lbtw_alpha = get_mtl_task_weights(args.mtl_weighting, task)

            # Appending of ,ckpt:{ckpt} in the job_name might be new, deprecated naming doesn't have this ...?
            job_name = f"mtl_train:{method},task:{len(task)}_tasks,disjoint_idx_{idx + 1},{args.mtl_weighting}_task_weighting,seed:{seed},ckpt:group"
            log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

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
            if args.respawn:
                command = append_ckpt_for_respawn(command, job_name, EPOCHS)

            job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_mtl_erm_nondisjoint_tasks_train(args):
    TASK_GRID = TASKS["MTL_NONDISJOINT"]  # MULTIPLE PAIRS

    assert args.opt in ["mtl_erm_nondisjoint"]
    mtl_method = args.opt.replace("_nondisjoint", "")
    method = mtl_method.replace("mtl_", "")

    key = f"{mtl_method}_group_ckpt_{args.mtl_weighting}_mtl_weighting"
    WD = PARAMS[key]["WD"]
    LR = PARAMS[key]["LR"]
    BATCH_SIZE = PARAMS[key]["BATCH_SIZE"]
    EPOCHS = PARAMS[key]["EPOCHS"]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)

    for seed in SEED_GRID:
        for idx, task in enumerate(TASK_GRID):
            task_weights, use_loss_balanced, lbtw_alpha = get_mtl_task_weights(args.mtl_weighting, task)

            job_name = f"mtl_train:{method},task:{len(task)}_tasks,nondisjoint_idx:{idx},{args.mtl_weighting}_task_weighting,seed:{seed}"
            log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

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
            if args.respawn:
                command = append_ckpt_for_respawn(command, job_name, EPOCHS)

            job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_mtl_erm_similar_tasks_train(args):
    TASK_GRID = TASKS["MTL_SIMILAR"]

    assert args.opt in ["mtl_erm_similar"]
    mtl_method = args.opt.replace("_similar", "")
    method = mtl_method.replace("mtl_", "")

    key = f"{mtl_method}_group_ckpt_{args.mtl_weighting}_mtl_weighting"
    WD = PARAMS[key]["WD"]
    LR = PARAMS[key]["LR"]
    BATCH_SIZE = PARAMS[key]["BATCH_SIZE"]
    EPOCHS = PARAMS[key]["EPOCHS"]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)

    for seed in SEED_GRID:
        for idx, task in enumerate(TASK_GRID):
            task_weights, use_loss_balanced, lbtw_alpha = get_mtl_task_weights(args.mtl_weighting, task)
            job_name = f"mtl_train:{method},task:{len(task)}_tasks,semantic_similar:{idx+1},{args.mtl_weighting}_task_weighting,seed:{seed}"
            log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

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
            if args.respawn:
                command = append_ckpt_for_respawn(command, job_name, EPOCHS)

            job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_mtl_erm_strong_tasks_train(args):
    TASK_GRID = TASKS["MTL_STRONG"]

    assert args.opt in ["mtl_erm_strong"]
    mtl_method = args.opt.replace("_strong", "")
    method = mtl_method.replace("mtl_", "")

    key = f"{mtl_method}_group_ckpt_{args.mtl_weighting}_mtl_weighting"
    WD = PARAMS[key]["WD"]
    LR = PARAMS[key]["LR"]
    BATCH_SIZE = PARAMS[key]["BATCH_SIZE"]
    EPOCHS = PARAMS[key]["EPOCHS"]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)

    for seed in SEED_GRID:
        for idx, task in enumerate(TASK_GRID):
            task_weights, use_loss_balanced, lbtw_alpha = get_mtl_task_weights(args.mtl_weighting, task)
            job_name = f"mtl_train:{method},task:{len(task)}_tasks,strong_spurious_correlation_idx:{idx},{args.mtl_weighting}_task_weighting,seed:{seed}"
            log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

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
            if args.respawn:
                command = append_ckpt_for_respawn(command, job_name, EPOCHS)

            job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_mtl_erm_weak_tasks_train(args):
    TASK_GRID = TASKS["MTL_WEAK"]

    assert args.opt in ["mtl_erm_weak"]
    mtl_method = args.opt.replace("_weak", "")
    method = mtl_method.replace("mtl_", "")

    key = f"{mtl_method}_group_ckpt_{args.mtl_weighting}_mtl_weighting"
    WD = PARAMS[key]["WD"]
    LR = PARAMS[key]["LR"]
    BATCH_SIZE = PARAMS[key]["BATCH_SIZE"]
    EPOCHS = PARAMS[key]["EPOCHS"]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)

    for seed in SEED_GRID:
        for idx, task in enumerate(TASK_GRID):
            task_weights, use_loss_balanced, lbtw_alpha = get_mtl_task_weights(args.mtl_weighting, task)
            job_name = f"mtl_train:{method},task:{len(task)}_tasks,weak_spurious_correlation_idx:{idx},{args.mtl_weighting}_task_weighting,seed:{seed}"
            log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

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
            if args.respawn:
                command = append_ckpt_for_respawn(command, job_name, EPOCHS)

            job_manager.submit(command, job_name=job_name, log_file=log_file)

def submit_stl_erm_cxr_train(args):

    assert args.opt in ["stl_erm_cxr"]
    stl_method = args.opt.replace("_cxr", "")
    method = stl_method.replace("stl_", "")

    TASK_GRID = ["Pneumothorax:Old", "Pneumonia:Male", "Pneumonia:Old", "Pneumothorax:Male"]

    WD = PARAMS[method]["WD"]
    LR = PARAMS[method]["LR"]
    BATCH_SIZE = PARAMS[method]["BATCH_SIZE"]
    EPOCHS = 5
    SEED_GRID=[0]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)
    dataset_name = "chestxray8"

    for task in TASK_GRID:
        for seed in SEED_GRID:
            job_name = f"baseline:{method},task:{task},{dataset_name},seed:{seed}"
            log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

            
            command = (
                f"python train_erm.py exp={method} "
                f"exp.optimizer.weight_decay={WD} "
                f"exp.optimizer.lr={LR} "
                f"exp.seed={seed} "
                f"exp.train.total_epochs={EPOCHS} "
                f"exp.dataset.groupings='[{task}]' "
                f"exp.dataset.name='{dataset_name}' "
                f"exp.dataloader.batch_size={BATCH_SIZE} "
                f"exp.train.log_dir=\\'{os.path.join(LOG_DIR, job_name)}\\'"
            )
            if args.respawn:
                command = append_ckpt_for_respawn(command, job_name, EPOCHS)


            job_manager.submit(command, job_name=job_name, log_file=log_file)

def submit_mtl_erm_cxr_train(args):
    TASK_GRID = [["Pneumothorax:Old", "Pneumonia:Male"],["Pneumothorax:Old", "Pneumonia:Old"], ["Pneumothorax:Male", "Pneumonia:Male"]]

    assert args.opt in ["mtl_erm_cxr"]
    mtl_method = args.opt.replace("_cxr", "")
    method = mtl_method.replace("mtl_", "")

    key = f"{mtl_method}_group_ckpt_{args.mtl_weighting}_mtl_weighting"
    WD = PARAMS[key]["WD"]
    LR = PARAMS[key]["LR"]
    BATCH_SIZE = PARAMS[key]["BATCH_SIZE"]
    EPOCHS = 5
    SEED_GRID=[0]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)
    dataset_name = "chestxray8"
    for seed in SEED_GRID:
        for idx, task in enumerate(TASK_GRID):
            task_weights, use_loss_balanced, lbtw_alpha = get_mtl_task_weights(args.mtl_weighting, task)
            job_name = f"mtl_train:{method},task:{len(task)}_tasks,{dataset_name}:{idx},{args.mtl_weighting}_task_weighting,seed:{seed}"
            log_file = os.path.join(args.slurm_logs, f"{job_name}.log")


            command = (
                f"python train_erm.py exp={method} "
                f"exp.optimizer.weight_decay={WD} "
                f"exp.optimizer.lr={LR} "
                f"exp.seed={seed} "
                f"exp.train.total_epochs={EPOCHS} "
                f"exp.dataset.name='{dataset_name}' "
                f"exp.dataset.groupings='{task}' "
                f"exp.dataloader.batch_size={BATCH_SIZE} "
                f"exp.dataset.task_weights='{task_weights}' "
                f"exp.dataset.loss_based_task_weighting={use_loss_balanced} "
                f"exp.dataset.lbtw_alpha={lbtw_alpha} "
                f"exp.train.log_dir=\\'{os.path.join(LOG_DIR, job_name)}\\'"
            )
            if args.respawn:
                command = append_ckpt_for_respawn(command, job_name, EPOCHS)

            job_manager.submit(command, job_name=job_name, log_file=log_file)


def main():
    args = parse_args()

    #######################
    # [0] STL SPURIOUS ID #
    #######################

    # Tunes STL methods on the list of spurious ID pairs
    if args.opt in ["erm_tune", "suby_tune"]:
        submit_stl_tune_train(args)

    # Runs STL methods on all attributes for spurious ID training
    elif args.opt in ["erm_id", "suby_id"]:
        submit_spurious_id_train(args)

    # Runs STL methods on all attributes for spurious ID training
    elif args.opt in ["erm_cxr_id", "suby_cxr_id"]:
        submit_spurious_id_train_cxr(args)

    ####################
    # [1] TUNE MTL ERM #
    ####################

    # Tunes MTL methods on the first disjoint pair
    elif args.opt in ["mtl_erm_tune", "mtl_suby_tune"]:
        submit_mtl_tune_train(args)

    ##################
    # [2] MTL VS STL #
    ##################

    # Trains STL methods on all tasks in flattened disjoint pairs
    elif args.opt in ["erm", "suby", "rwy", "jtt"]:
        submit_stl_train(args)

    # Trains MTL methods with CVX optimization on the first disjoint pair
    elif args.opt in ["mtl_rwy", "mtl_suby"]:
        submit_mtl_cvx_disjoint_tasks_train(args)

    # Trains MTL JTT on the first disjoint pair
    elif args.opt in ["mtl_jtt"]:
        submit_mtl_jtt_disjoint_tasks_train(args)

    # Trains MTL ERM on the mtl stl comparison pairs
    elif args.opt in ["mtl_erm_mtl_stl"]:
        submit_mtl_stl_erm_train(args)

    #########################
    # [3] MTL TASK ABLATION #
    #########################

    # Trains MTL methods on task sizes from 2-4
    elif args.opt in ["mtl_erm_ablate_disjoint"]:
        submit_mtl_erm_ablate_disjoint_tasks_train(args)

    elif args.opt in ["mtl_erm_ablate_nondisjoint"]:
        submit_mtl_erm_ablate_nondisjoint_tasks_train(args)

    ###############################
    # [4] DISJOINT VS NONDISJOINT #
    ###############################

    # Trains MTL methods on the last 4 disjoint task pairs
    elif args.opt in ["mtl_erm_disjoint"]:
        submit_mtl_erm_disjoint_tasks_train(args)

    # Trains MTL methods on all 5 nondisjoint task pairs
    elif args.opt in ["mtl_erm_nondisjoint"]:
        submit_mtl_erm_nondisjoint_tasks_train(args)

    #############################
    # [5] SIMILAR VS NONSIMILAR #
    #############################

    # Trains MTL methods on 4 additional semantically similar task pairs
    elif args.opt in ["mtl_erm_similar"]:
        submit_mtl_erm_similar_tasks_train(args)

    ###############################
    # [6] STRONG VS WEAK SPURIOUS #
    ###############################

    # Trains MTL methods on 2 additional strongly spuriously correlated task pairs
    elif args.opt in ["mtl_erm_strong"]:
        submit_mtl_erm_strong_tasks_train(args)

    # Trains MTL methods on 2 additional strongly spuriously correlated task pairs
    elif args.opt in ["mtl_erm_weak"]:
        submit_mtl_erm_weak_tasks_train(args)

    ###############################
    # [7] ChestXray Training #
    ###############################

    # Trains MTL methods on 2 additional strongly spuriously correlated task pairs
    elif args.opt in ["mtl_erm_cxr"]:
        submit_mtl_erm_cxr_train(args)

    # Trains MTL methods on 2 additional strongly spuriously correlated task pairs
    elif args.opt in ["stl_erm_cxr"]:
        submit_stl_erm_cxr_train(args)

    else:
        raise ValueError(f"Didn't recognize opt={args.opt}. Did you forget to add a check for this function?")


if __name__ == "__main__":
    main()
