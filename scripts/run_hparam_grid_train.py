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
from scripts.job_manager import JobManager

USER = os.environ["USER"]
LOG_DIR = "./logs"
SEED_GRID = [0, 1, 2]

# Dictionary of task lists based on the type of ablation experiments we want to run
TASKS = {

    ##########################
    # COMPLETE MTL TASK SETS #
    ##########################

    # 5 tasks to tune STL methods on before running spurious ID on them
    "SPURIOUS_ID_TUNE":
        [
            "Attractive:Eyeglasses",
            "Smiling:High_Cheekbones",
            "Pointy_Nose:Rosy_Cheeks",
            "Oval_Face:Rosy_Cheeks",
            "Young:Attractive",
        ],

    # 2 sets of ablations over disjoint tasks, for each: 1x MTL(2), 3x MTL(3), 3x MTL(4)
    "MTL_ABLATE_DISJOINT":
        [
            # PREVIOUS RUNS

            # MTL(2)
            ["Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair"],
            ["Bushy_Eyebrows:Blond_Hair", "High_Cheekbones:Smiling"],

            # MTL(3)
            ["Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair", "Wearing_Lipstick:Male"],
            ["Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair", "Gray_Hair:Young"],
            ["Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair", "High_Cheekbones:Smiling"],
            ["Bushy_Eyebrows:Blond_Hair", "High_Cheekbones:Smiling", "Wearing_Lipstick:Male"],
            ["Bushy_Eyebrows:Blond_Hair", "High_Cheekbones:Smiling", "Brown_Hair:Wearing_Hat"],
            ["Bushy_Eyebrows:Blond_Hair", "High_Cheekbones:Smiling", "Gray_Hair:Young"],

            # MTL(4)
            ["Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair", "Wearing_Lipstick:Male", "High_Cheekbones:Smiling"],
            ["Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair", "Gray_Hair:Young", "High_Cheekbones:Smiling"],
            ["Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair", "High_Cheekbones:Smiling", "Wearing_Lipstick:Male"],
            ["Bushy_Eyebrows:Blond_Hair", "High_Cheekbones:Smiling", "Wearing_Lipstick:Male", "Gray_Hair:Young"],
            ["Bushy_Eyebrows:Blond_Hair", "High_Cheekbones:Smiling", "Brown_Hair:Wearing_Hat", "Big_Lips:Chubby"],
            ["Bushy_Eyebrows:Blond_Hair", "High_Cheekbones:Smiling", "Gray_Hair:Young", "Brown_Hair:Wearing_Hat"],
        ],

    # 2 sets of ablations over nondisjoint tasks, for each: 1x MTL(2), 3x MTL(3), 3x MTL(4)
    "MTL_ABLATE_NONDISJOINT":
        [
            # MTL(2)
            ["Arched_Eyebrows:Male", "Big_Nose:Male"],
            ["Blond_Hair:Male", "Wearing_Earrings:Male"],

            # MTL(3)
            ["Arched_Eyebrows:Male", "Big_Nose:Male", "Wearing_Earrings:Male"],
            ["Arched_Eyebrows:Male", "Big_Nose:Male", "Wearing_Lipstick:Male"],
            ["Arched_Eyebrows:Male", "Big_Nose:Male", "Attractive:Male"],
            ["Blond_Hair:Male", "Wearing_Earrings:Male", "Wearing_Lipstick:Male"],
            ["Blond_Hair:Male", "Wearing_Earrings:Male", "Big_Nose:Male"],
            ["Blond_Hair:Male", "Wearing_Earrings:Male", "Arched_Eyebrows:Male"],

            # MTL(4)
            ["Arched_Eyebrows:Male", "Big_Nose:Male", "Wearing_Earrings:Male", "Blond_Hair:Male"],
            ["Arched_Eyebrows:Male", "Big_Nose:Male", "Wearing_Lipstick:Male", "Wearing_Earrings:Male"],
            ["Arched_Eyebrows:Male", "Big_Nose:Male", "Attractive:Male", "Blond_Hair:Male"],
            ["Blond_Hair:Male", "Wearing_Earrings:Male", "Wearing_Lipstick:Male", "Arched_Eyebrows:Male"],
            ["Blond_Hair:Male", "Wearing_Earrings:Male", "Big_Nose:Male", "Attractive:Male"],
            ["Blond_Hair:Male", "Wearing_Earrings:Male", "Arched_Eyebrows:Male", "Big_Nose:Male"],
        ],

    # 5 pairs of pairwise disjoint tasks
    "MTL_DISJOINT":
        [
            ["Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair"],
            ["Wearing_Lipstick:Male", "Gray_Hair:Young"],
            ["High_Cheekbones:Smiling", "Brown_Hair:Wearing_Hat"],
            ["No_Beard:Wearing_Lipstick", "Young:Chubby"],
            ["Bangs:Wearing_Hat", "Pointy_Nose:Heavy_Makeup"],
        ],

    # 5 pairs of pairwise nondisjoint tasks
    "MTL_NONDISJOINT":
        [
            ["Wearing_Earrings:Male", "Attractive:Male"],
            ["No_Beard:Heavy_Makeup", "Pointy_Nose:Heavy_Makeup"],
            ["Attractive:Gray_Hair", "Big_Nose:Gray_Hair"],
            ["Heavy_Makeup:Wearing_Lipstick", "No_Beard:Wearing_Lipstick"],
            ["Bangs:Wearing_Hat", "Blond_Hair:Wearing_Hat"],
        ],

    ############################
    # INCOMPLETE MTL TASK SETS #
    ############################
    # exclude repeats from above

    # 4 NEW pairs of semantically similar task pairs
    "MTL_SIMILAR":
        [
            ["Big_Nose:Wearing_Lipstick", "High_Cheekbones:Smiling"],
            ["Big_Lips:Goatee", "Wearing_Lipstick:Male"],
            ["Bags_Under_Eyes:Double_Chin", "High_Cheekbones:Rosy_Cheeks"],
            ["Blond_Hair:Wearing_Hat", "Brown_Hair:Wearing_Hat"],
        ],

    # 2 NEW pairs of strongly spuriously correlated task pairs
    "MTL_STRONG": [
        ["Wearing_Lipstick:Male", "High_Cheekbones:Smiling"],
        ["Heavy_Makeup:Male", "Wearing_Earrings:Male"],
    ],

    # 2 NEW pairs of weakly spuriously correlated task pairs
    "MTL_WEAK": [
        ["Big_Lips:Chubby", "Young:Chubby"],
        ["High_Cheekbones:Rosy_Cheeks", "Brown_Hair:Wearing_Hat"],
    ],
}

# Defines param grids for tuning methods
GRIDS = {
    "erm": {
        "WD": [1e-4, 1e-3, 1e-2, 1e-1],
        "LR": [1e-5, 5e-5, 1e-4],
        "BATCH_SIZE": [128],
    },
    "suby": {
        "WD": [1e-2, 1e-1, 1],
        "LR": [1e-5, 1e-4, 1e-3],
        "BATCH_SIZE": [32, 64],
    },
    "mtl_erm": {
        "WD": [1e-4, 1e-3, 1e-2, 1e-1],
        "LR": [1e-5, 1e-4, 1e-3],
        "BATCH_SIZE": [32, 64, 128],
    },
    "mtl_suby": {
        "WD": [1e-2, 1e-1, 1],
        "LR": [1e-5, 1e-4, 1e-3],
        "BATCH_SIZE": [32, 64],
    },
}

# Defines params for established methods
PARAMS = {
    # STL baseline methods
    "erm": {
        "WD": 1e-4,
        "LR": 1e-4,
        "BATCH_SIZE": 128,
        "EPOCHS": 50,
    },
    "suby": {
        "WD": 1e-2,
        "LR": 1e-3,
        "BATCH_SIZE": 128,
        "EPOCHS": 60,
    },
    "rwy": {
        "WD": 1e-2,
        "LR": 1e-4,
        "BATCH_SIZE": 2,
        "EPOCHS": 60,
    },
    "jtt":
        {
            "WD": 1e-1,
            "LR": 1e-5,
            "BATCH_SIZE": 128,
            "EPOCHS": 50,
            "T": 1,
            "LAM_UP": 50,  # some extras here for jtt, handled with an if statement
        },

    # MTL ERM methods, tuned with different task weightings and checkpointing
    "mtl_erm_avg_ckpt_static_equal_mtl_weighting": {
        "WD": 1e-1,
        "LR": 1e-4,
        "BATCH_SIZE": 64,
        "EPOCHS": 50,
    },
    "mtl_erm_avg_ckpt_static_delta_mtl_weighting": {
        "WD": 1e-1,
        "LR": 1e-4,
        "BATCH_SIZE": 64,
        "EPOCHS": 50,
    },
    "mtl_erm_avg_ckpt_dynamic_mtl_weighting": {
        "WD": 1e-1,
        "LR": 1e-4,
        "BATCH_SIZE": 64,
        "EPOCHS": 50,
    },
    "mtl_erm_group_ckpt_static_equal_mtl_weighting": {
        "WD": 1e-2,
        "LR": 1e-4,
        "BATCH_SIZE": 32,
        "EPOCHS": 50,
    },
    "mtl_erm_group_ckpt_static_delta_mtl_weighting": {
        "WD": 1e-3,
        "LR": 1e-3,
        "BATCH_SIZE": 32,
        "EPOCHS": 50,
    },
    "mtl_erm_group_ckpt_dynamic_mtl_weighting": {
        "WD": 1e-2,
        "LR": 1e-4,
        "BATCH_SIZE": 32,
        "EPOCHS": 50,
    },

    # THESE AREN'T TUNED FOR MTL
    "mtl_suby": {
        "WD": 1e-2,
        "LR": 1e-3,
        "BATCH_SIZE": 128,
        "EPOCHS": 60,
    },
    "mtl_rwy": {
        "WD": 1e-2,
        "LR": 1e-4,
        "BATCH_SIZE": 2,
        "EPOCHS": 60,
    },
    "mtl_jtt":
        {
            "WD": 1e-1,
            "LR": 1e-5,
            "BATCH_SIZE": 128,
            "EPOCHS": 50,
            "T": 1,
            "LAM_UP": 50,  # some extras here for jtt, handled with an if statement
        },
}


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
    args = parser.parse_args()

    # Convert relative paths to absolute paths to help slurm out
    args.slurm_logs = os.path.abspath(args.slurm_logs)
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
    if ckpt_num < epochs:
        return f"{command} exp.train.load_ckpt=\\'{ckpt_path}\\'"
    else:
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

    load_up_pkl_path = os.path.join(LOG_DIR, job_name, f"jtt_error_set_inv.pkl")  # default is inv merging, even for stl

    if stage_1_ckpt_num != epochs1 or (stage_1_ckpt_num == epochs1 and not os.path.exists(load_up_pkl_path)):
        return f"{command} exp.load_stage_1_ckpt=\\'{stage_1_ckpt_path}\\'"
    elif stage_1_ckpt_num == epochs1 and stage_2_ckpt_num is None:
        return f"{command} exp.load_up_pkl={load_up_pkl_path}"
    elif stage_1_ckpt_num == epochs1 and stage_2_ckpt_num != epochs2:
        return f"{command} exp.load_stage_2_ckpt={stage_2_ckpt_path}"
    else:
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
        warnings.warn("--respawn is not implemented for this function.", category=warnings.FutureWarning)

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)

    for task in TASK_GRID:
        for wd in WD_GRID:
            for lr in LR_GRID:
                for batch_size in BATCH_SIZE_GRID:
                    job_name = job_name_generator(task, wd, lr, batch_size)
                    log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

                    command = (
                        f"python train_erm.py exp={method} "
                        f"exp.optimizer.weight_decay={wd} "
                        f"exp.optimizer.lr={lr} "
                        f"exp.dataloader.batch_size={batch_size} "
                        f"exp.train.total_epochs={EPOCHS} "
                        f"exp.dataset.groupings='[{task}]' "
                        f"exp.train.log_dir=\\'{os.path.join(LOG_DIR, job_name)}\\'"
                    )
                    job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_stl_train(args):
    method = args.opt
    assert method in ["erm", "suby", "rwy", "jtt"]  # just to double check

    if method in ["suby", "rwy", "jtt"]:
        TASK_GRID = flatten(TASKS["MTL_DISJOINT"])
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
    TASK = TASKS["MTL_DISJOINT"][0]  # SINGLE PAIR
    CVX_GRID = ["qp", "maxent"]

    assert args.opt in ["mtl_rwy", "mtl_suby"], "This method only supports --opt=mtl_rwy and --opt=mtl_suby"
    method = args.opt.replace("mtl_", "")

    WD = PARAMS[args.opt]["WD"]
    LR = PARAMS[args.opt]["LR"]
    BATCH_SIZE = PARAMS[args.opt]["BATCH_SIZE"]
    EPOCHS = PARAMS[args.opt]["EPOCHS"]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)
    task_weights, use_loss_balanced, lbtw_alpha = get_mtl_task_weights(args.mtl_weighting, TASK)

    for seed in SEED_GRID:
        for cvx in CVX_GRID:
            job_name = f"mtl_train:{method},task:{len(TASK)}_tasks,{args.mtl_weighting}_task_weighting,seed:{seed},cvx:{cvx}"
            log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

            command = (
                f"python train_erm.py exp={method} "
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
            if args.respawn:
                command = append_ckpt_for_respawn(command, job_name, EPOCHS)

            job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_mtl_jtt_disjoint_tasks_train(args):
    TASK = TASKS["MTL_DISJOINT"][0]  # SINGLE PAIR

    assert args.opt in ["mtl_jtt"], "This method only supports --opt=mtl_jtt"

    WD = PARAMS[args.opt]["WD"]
    LR = PARAMS[args.opt]["LR"]
    BATCH_SIZE = PARAMS[args.opt]["BATCH_SIZE"]
    EPOCHS = PARAMS[args.opt]["EPOCHS"]

    T = PARAMS["jtt"]["T"]
    LAM_UP = PARAMS["jtt"]["LAM_UP"]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)
    task_weights, use_loss_balanced, lbtw_alpha = get_mtl_task_weights(args.mtl_weighting, TASK)

    for seed in SEED_GRID:
        job_name = f"mtl_train:jtt,task:{len(TASK)}_tasks,{args.mtl_weighting}_task_weighting,seed:{seed}"
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
            f"exp.groupings='[{TASK}]' "
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
            job_name = f"mtl_train:{method},task:{len(task)}_tasks,disjoint_idx_{idx},{args.mtl_weighting}_task_weighting,seed:{seed},ckpt:group"
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
            job_name = f"mtl_train:{method},task:{len(task)}_tasks,semantic_similar:{idx + 1},{args.mtl_weighting}_task_weighting,seed:{seed}"
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

    key = f"{mtl_method}_strong_ckpt_{args.mtl_weighting}_mtl_weighting"
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


def main():
    args = parse_args()

    #######################
    # [0] STL SPURIOUS ID #
    #######################

    # Tunes STL methods on the list of spurious ID pairs
    if args.opt in ["erm_tune", "suby_tune"]:
        submit_stl_tune_train(args)

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

    else:
        raise ValueError(f"Didn't recognize opt={args.opt}. Did you forget to add a check for this function?")


if __name__ == "__main__":
    main()
