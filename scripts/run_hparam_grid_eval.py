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
    parser.add_argument("--slurm_logs", type=str, default="slurm_logs", required=False, help="Directory to output slurm logs")
    parser.add_argument("--opt", type=str, required=True, help="The name of the submit_*_grid_jobs function to call.")
    parser.add_argument(
        "--mtl_weighting",
        type=str,
        default="static_equal",
        choices=["static_equal", "static_delta", "dynamic"],
        help="For MTL tuning runs, what type of weighting to eval"
    )
    parser.add_argument(
        "--mtl_checkpoint_type",
        type=str,
        required=False,
        default=None,
        choices=["average", "best-worst", "per-task", None],
        help="Whether to choose checkpointing based on the average performance, best worst performance, or per-task"
    )
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


def submit_erm_baseline_disjoint_eval_test(args):
    ## DECLARE MACROS HERE ##
    SEED_GRID = [0, 1, 2]
    TASK_GRID = ["Heavy_Makeup:Male"]
    
    #[
    #    "Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair", "Goatee:No_Beard", "Gray_Hair:Young", "High_Cheekbones:Smiling",
    #    "Wavy_Hair:Straight_Hair", "Wearing_Lipstick:Male"
    #]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)
    method = "erm"
    for task in TASK_GRID:
        for seed in SEED_GRID:
            for checkpoint_type in ["avg", "group"]:
                job_name = f"eval_baseline:{method},task:{task},seed:{seed}"
                log_file = os.path.join(args.slurm_logs, f"{job_name}.log")
                save_json = f"test_stats_{checkpoint_type}_checkpoint.json"

                log_dir = os.path.join(LOG_DIR, job_name[5:])
                results_dir = os.path.join(log_dir, "results")
                save_json = os.path.join(results_dir, save_json)

                command = f"python -m scripts.find_best_ckpt --run_test --log_dir {log_dir} --metric {checkpoint_type} --learning_type stl --save_json {save_json}"
                job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_suby_baseline_disjoint_eval_test(args):
    ## DECLARE MACROS HERE ##
    SEED_GRID = [0, 1, 2]
    TASK_GRID = [
        "Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair", "Goatee:No_Beard", "Gray_Hair:Young", "High_Cheekbones:Smiling",
        "Wavy_Hair:Straight_Hair", "Wearing_Lipstick:Male"
    ]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)

    method = "suby"
    for task in TASK_GRID:
        for seed in SEED_GRID:
            for checkpoint_type in ["avg", "group"]:
                job_name = f"eval_baseline:{method},task:{task},seed:{seed}"
                log_file = os.path.join(args.slurm_logs, f"{job_name}.log")
                save_json = f"test_stats_{checkpoint_type}_checkpoint.json"

                log_dir = os.path.join(LOG_DIR, job_name[5:])
                results_dir = os.path.join(log_dir, "results")
                save_json = os.path.join(results_dir, save_json)

                command = f"python -m scripts.find_best_ckpt --run_test --log_dir {log_dir} --metric {checkpoint_type} --learning_type stl --save_json {save_json}"
                job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_rwy_baseline_disjoint_eval_test(args):
    SEED_GRID = [0, 1, 2]
    TASK_GRID = [
        "Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair", "Goatee:No_Beard", "Gray_Hair:Young", "High_Cheekbones:Smiling",
        "Wavy_Hair:Straight_Hair", "Wearing_Lipstick:Male"
    ]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)

    method = "rwy"
    for task in TASK_GRID:
        for seed in SEED_GRID:
            for metric in ["avg", "group"]:
                job_name = f"baseline:{method},task:{task},seed:{seed}"
                log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

                save_json = f"test_stats_{metric}_checkpoint.json"

                log_dir = os.path.join(LOG_DIR, job_name)
                results_dir = os.path.join(log_dir, "results")
                save_json = os.path.join(results_dir, save_json)

                command = f"python -m scripts.find_best_ckpt --run_test --log_dir {log_dir} --metric {metric} --learning_type stl --save_json {save_json}"
                job_manager.submit(command, job_name=job_name, log_file=log_file)

### CHANGE AND UPDATE
def submit_mtl_disjoint_tasks_eval_val(args):
    WD_GRID = [1e-4, 1e-3, 1e-2, 1e-1]
    LR_GRID = [1e-5, 1e-4, 1e-3]
    BATCH_SIZE_GRID = [32, 64, 128]
    SEED_GRID = [0]
    TASK = ["Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair"]

    if args.mtl_checkpoint_type is None:
        raise ValueError(f"Please specify an option for --{args.mtl_checkpoint_type}")

    method = "erm"
    for wd in WD_GRID:
        for lr in LR_GRID:
            for batch_size in BATCH_SIZE_GRID:
                for seed in SEED_GRID:
                    for metric in ["avg", "group"]:

                        job_name = f"eval_mtl_tuning:{method},task:{len(TASK)}_tasks_{args.mtl_weighting}_task_weighting,seed:{seed},wd:{wd},lr:{lr},batch_size:{batch_size}"
                        save_json = f"val_stats_{metric}_checkpoint_{args.mtl_checkpoint_type}_mtl_type.json"

                        ckpt_num = find_best_ckpt(
                            f'./logs/{job_name[5:]}', run_test=False, test_groupings="", metric=metric, learning_type="mtl", mtl_checkpoint_type=f"{args.mtl_checkpoint_type}"
                        ) - 1
                        with open(os.path.join(f"./logs/{job_name[5:]}", "results", f"val_stats_{ckpt_num}.json"), "r") as f:
                            best_val_stats = json.load(f)

                        with open(os.path.join(f"./logs/{job_name[5:]}", "results", save_json), "w") as fp:
                            json.dump(best_val_stats, fp)


def submit_jtt_baseline_disjoint_eval_test(args):
    ## DECLARE MACROS HERE ##
    SEED_GRID = [0, 1, 2]
    TASK_GRID = ["High_Cheekbones:Smiling", "Wavy_Hair:Straight_Hair"]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)
    method = "jtt"
    for task in TASK_GRID:
        for seed in SEED_GRID:
            for checkpoint_type in ["avg", "group"]:
                job_name = f"eval_baseline:{method},task:{task},seed:{seed}"
                log_file = os.path.join(args.slurm_logs, f"{job_name}.log")
                save_json = f"test_stats_{checkpoint_type}_checkpoint.json"

                log_dir = os.path.join(LOG_DIR, job_name[5:], 'stage_2')
                results_dir = os.path.join(log_dir, "results")
                save_json = os.path.join(results_dir, save_json)

                command = f"python -m scripts.find_best_ckpt --run_test --log_dir {log_dir} --metric {checkpoint_type} --learning_type stl --save_json {save_json}"
                job_manager.submit(command, job_name=job_name, log_file=log_file)

### CHANGE AND UPDATE
def submit_mtl_disjoint_tasks_eval_test(args):
    ## DECLARE MACROS HERE ##
    SEED_GRID = [0, 1, 2]
    TASK = ["Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair"]

    if args.mtl_checkpoint_type is None:
        raise ValueError(f"Please specify an option for --{args.mtl_checkpoint_type}")

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)
    method = "erm"
    for seed in SEED_GRID:
        for checkpoint_type in ["avg", "group"]:
            for mtl_hparam_tpye in ['avg', 'group']:

                job_name = f"eval_mtl_train:{method},task:{len(TASK)}_tasks_{args.mtl_weighting}_task_weighting,seed:{seed},ckpt:{mtl_hparam_tpye}"
                log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

                save_json = f"test_stats_{checkpoint_type}_checkpoint_{args.mtl_checkpoint_type}_mtl_type.json"

                log_dir = os.path.join(LOG_DIR, job_name[5:])
                results_dir = os.path.join(log_dir, "results")
                save_json = os.path.join(results_dir, save_json)

                command = f"python -m scripts.find_best_ckpt --run_test --log_dir {log_dir} --metric {checkpoint_type} --learning_type mtl --save_json {save_json} --mtl_checkpoint_type {args.mtl_checkpoint_type}"
                job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_mtl_rwy_disjoint_tasks_eval_test(args):
    SEED_GRID = [0, 1, 2]
    TASK = ["Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair"]
    CVX_GRID = ["qp", "maxent"]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)

    for seed in SEED_GRID:
        for cvx in CVX_GRID:
            for metric in ["group", "avg"]:
                job_name = f"eval_mtl_train:rwy,task:{len(TASK)}_tasks_{args.mtl_weighting}_task_weighting,seed:{seed},cvx:{cvx}"
                log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

                save_json = f"test_stats_{metric}_checkpoint_{args.mtl_checkpoint_type}_mtl_type.json"
                log_dir = os.path.join(LOG_DIR, job_name[5:])
                results_dir = os.path.join(log_dir, "results")
                save_json = os.path.join(results_dir, save_json)

                command = f"python -m scripts.find_best_ckpt --run_test --log_dir {log_dir} --metric {metric} --mtl_checkpoint_type {args.mtl_checkpoint_type} --learning_type mtl --save_json {save_json}"
                job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_mtl_suby_disjoint_tasks_eval_test(args):
    SEED_GRID = [0, 1, 2]
    TASK = ["Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair"]
    CVX_GRID = ["qp", "maxent"]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)

    for seed in SEED_GRID:
        for cvx in CVX_GRID:
            for metric in ["group", "avg"]:
                job_name = f"eval_mtl_train:suby,task:{len(TASK)}_tasks_{args.mtl_weighting}_task_weighting,seed:{seed},cvx:{cvx}"
                log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

                save_json = f"test_stats_{metric}_checkpoint_{args.mtl_checkpoint_type}_mtl_type.json"
                log_dir = os.path.join(LOG_DIR, job_name[5:])
                results_dir = os.path.join(log_dir, "results")
                save_json = os.path.join(results_dir, save_json)

                command = f"python -m scripts.find_best_ckpt --run_test --log_dir {log_dir} --metric {metric} --mtl_checkpoint_type {args.mtl_checkpoint_type} --learning_type mtl --save_json {save_json}"
                job_manager.submit(command, job_name=job_name, log_file=log_file)

def submit_erm_baseline_nondisjoint_eval_test(args):
    ## DECLARE MACROS HERE ##
    SEED_GRID = [0, 1, 2]
    TASK_GRID = ["Wearing_Earrings:Male", "Attractive:Male", "No_Beard:Heavy_Makeup", "Pointy_Nose:Heavy_Makeup", "Attractive:Gray_Hair", "Big_Nose:Gray_Hair", "Heavy_Makeup:Wearing_Lipstick", "No_Beard:Wearing_Lipstick", "Bangs:Wearing_Hat", "Blond_Hair:Wearing_Hat"]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)
    method = "erm"
    for task in TASK_GRID:
        for seed in SEED_GRID:
            for checkpoint_type in ["avg", "group"]:
                job_name = f"eval_baseline:{method},task:{task},seed:{seed}"
                log_file = os.path.join(args.slurm_logs, f"{job_name}.log")
                save_json = f"test_stats_{checkpoint_type}_checkpoint.json"

                log_dir = os.path.join(LOG_DIR, job_name[5:])
                results_dir = os.path.join(log_dir, "results")
                save_json = os.path.join(results_dir, save_json)

                command = f"python -m scripts.find_best_ckpt --run_test --log_dir {log_dir} --metric {checkpoint_type} --learning_type stl --save_json {save_json}"
                job_manager.submit(command, job_name=job_name, log_file=log_file)

def submit_mtl_nondisjoint_tasks_eval_test(args):
    SEED_GRID = [0, 1, 2]
    TASKS = [["Wearing_Earrings:Male", "Attractive:Male"], ["No_Beard:Heavy_Makeup", "Pointy_Nose:Heavy_Makeup"],
    ["Attractive:Gray_Hair", "Big_Nose:Gray_Hair"], ["Heavy_Makeup:Wearing_Lipstick", "No_Beard:Wearing_Lipstick"],
    ["Bangs:Wearing_Hat", "Blond_Hair:Wearing_Hat"]]

    if args.mtl_checkpoint_type is None:
        raise ValueError(f"Please specify an option for --{args.mtl_checkpoint_type}")

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)
    method = "erm"

    for seed in SEED_GRID:
        for idx, task in enumerate(TASKS):
            for checkpoint_type in ["avg", "group"]:
                job_name = f"eval_mtl_train:{method},task:{len(task)}_tasks,nondisjoint_idx:{idx},{args.mtl_weighting}_task_weighting,seed:{seed}"
                log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

                save_json = f"test_stats_{checkpoint_type}_checkpoint_{args.mtl_checkpoint_type}_mtl_type.json"

                log_dir = os.path.join(LOG_DIR, job_name[5:])
                results_dir = os.path.join(log_dir, "results")
                save_json = os.path.join(results_dir, save_json)

                command = f"python -m scripts.find_best_ckpt --run_test --log_dir {log_dir} --metric {checkpoint_type} --learning_type mtl --save_json {save_json} --mtl_checkpoint_type {args.mtl_checkpoint_type}"
                job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_mtl_semantic_similar_tasks_eval_test(args):
    SEED_GRID = [0, 1, 2]
    TASKS = [["Big_Nose:Wearing_Lipstick", "High_Cheekbones:Smiling"], ["Big_Lips:Goatee", "Wearing_Lipstick:Male"],
    ["Bags_Under_Eyes:Double_Chin", "High_Cheekbones:Rosy_Cheeks"], ["Blond_Hair:Wearing_Hat", "Brown_Hair:Wearing_Hat"]]
    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)
    method = "erm"

    for seed in SEED_GRID:
        for idx, task in enumerate(TASKS):
            for checkpoint_type in ["avg", "group"]:
                job_name = f"eval_mtl_train:{method},task:{len(task)}_tasks,semantic_similar:{idx + 1},{args.mtl_weighting}_task_weighting,seed:{seed}"
                log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

                save_json = f"test_stats_{checkpoint_type}_checkpoint_{args.mtl_checkpoint_type}_mtl_type.json"

                log_dir = os.path.join(LOG_DIR, job_name[5:])
                results_dir = os.path.join(log_dir, "results")
                save_json = os.path.join(results_dir, save_json)

                command = f"python -m scripts.find_best_ckpt --run_test --log_dir {log_dir} --metric {checkpoint_type} --learning_type mtl --save_json {save_json} --mtl_checkpoint_type {args.mtl_checkpoint_type}"
                job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_mtl_strong_spurious_correlations_tasks_eval_test(args):
    SEED_GRID = [0, 1, 2]
    TASKS = [["Wearing_Lipstick:Male", "High_Cheekbones:Smiling"], ["Heavy_Makeup:Male", "Wearing_Earrings:Male"]]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)
    method = "erm"

    for seed in SEED_GRID:
        for idx, task in enumerate(TASKS):
            for checkpoint_type in ["avg", "group"]:
                job_name = f"eval_mtl_train:{method},task:{len(task)}_tasks,strong_spurious_correlation_idx:{idx},{args.mtl_weighting}_task_weighting,seed:{seed}"
                log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

                save_json = f"test_stats_{checkpoint_type}_checkpoint_{args.mtl_checkpoint_type}_mtl_type.json"

                log_dir = os.path.join(LOG_DIR, job_name[5:])
                results_dir = os.path.join(log_dir, "results")
                save_json = os.path.join(results_dir, save_json)

                command = f"python -m scripts.find_best_ckpt --run_test --log_dir {log_dir} --metric {checkpoint_type} --learning_type mtl --save_json {save_json} --mtl_checkpoint_type {args.mtl_checkpoint_type}"
                job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_mtl_weak_spurious_correlations_tasks_eval_test(args):
    SEED_GRID = [0, 1, 2]
    TASKS = [["Big_Lips:Chubby", "Young:Chubby"], ["High_Cheekbones:Rosy_Cheeks", "Brown_Hair:Wearing_Hat"]]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)
    method = "erm"

    for seed in SEED_GRID:
        for idx, task in enumerate(TASKS):
            for checkpoint_type in ["avg", "group"]:
                job_name = f"eval_mtl_train:{method},task:{len(task)}_tasks,weak_spurious_correlation_idx:{idx},{args.mtl_weighting}_task_weighting,seed:{seed}"
                log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

                save_json = f"test_stats_{checkpoint_type}_checkpoint_{args.mtl_checkpoint_type}_mtl_type.json"

                log_dir = os.path.join(LOG_DIR, job_name[5:])
                results_dir = os.path.join(log_dir, "results")
                save_json = os.path.join(results_dir, save_json)

                command = f"python -m scripts.find_best_ckpt --run_test --log_dir {log_dir} --metric {checkpoint_type} --learning_type mtl --save_json {save_json} --mtl_checkpoint_type {args.mtl_checkpoint_type}"
                job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_mtl_jtt_eval_test(args):
    SEED_GRID = [0, 1, 2]
    TASK = ["Big_Lips:Chubby", "Bushy_Eyebrows:Blond_Hair"]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)

    for seed in SEED_GRID:
        for checkpoint_type in ["avg", "group"]:
            for mtl_weighting in ["static_equal", "static_delta", "dynamic"]: #"static_delta"

                job_name = f"eval_mtl_train:jtt,task:{len(TASK)}_tasks_{mtl_weighting}_task_weighting,seed:{seed}"
                log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

                save_json = f"test_stats_{checkpoint_type}_checkpoint_{args.mtl_checkpoint_type}_mtl_type.json"

                log_dir = os.path.join(LOG_DIR, "mtl_jtt", job_name[5:], "stage_2")
                results_dir = os.path.join(log_dir, "results")
                save_json = os.path.join(results_dir, save_json)

                command = f"python -m scripts.find_best_ckpt --run_test --log_dir {log_dir} --metric {checkpoint_type} --learning_type mtl --save_json {save_json} --mtl_checkpoint_type {args.mtl_checkpoint_type}"
                job_manager.submit(command, job_name=job_name, log_file=log_file)



def main():
    args = parse_args()
    if args.mode == "sbatch":
        os.makedirs(args.slurm_logs, exist_ok=True)
    if args.opt == "suby_val":
        submit_suby_eval_val(args)
    elif args.opt == "suby_test":
        submit_suby_eval_test(args)
    elif args.opt == "rwy_disjoint_test":
        submit_rwy_baseline_disjoint_eval_test(args)
    elif args.opt == "suby_disjoint_test":
        submit_suby_baseline_disjoint_eval_test(args)
    elif args.opt == "erm_disjoint_test":
        submit_erm_baseline_disjoint_eval_test(args)
    elif args.opt == "mtl_disjoint_val":
        submit_mtl_disjoint_tasks_eval_val(args)
    elif args.opt == "jtt_disjoint_test":
        submit_jtt_baseline_disjoint_eval_test(args)
    elif args.opt == "mtl_disjoint_test":
        submit_mtl_disjoint_tasks_eval_test(args)
    elif args.opt == "mtl_rwy":
        submit_mtl_rwy_disjoint_tasks_eval_test(args)
    elif args.opt == "mtl_suby":
        submit_mtl_suby_disjoint_tasks_eval_test(args)
    elif args.opt == "erm_nondisjoint":
        submit_erm_baseline_nondisjoint_eval_test(args)
    elif args.opt == "mtl_nondisjoint":
        submit_mtl_nondisjoint_tasks_eval_test(args)
    elif args.opt == "mtl_semantic_similar":
        submit_mtl_semantic_similar_tasks_eval_test(args)
    elif args.opt == "mtl_strong_spurious":
        submit_mtl_strong_spurious_correlations_tasks_eval_test(args)
    elif args.opt == "mtl_weak_spurious":
        submit_mtl_weak_spurious_correlations_tasks_eval_test(args)
    elif args.opt == "mtl_jtt":
        submit_mtl_jtt_eval_test(args)
    else:
        raise ValueError(f"Didn't recognize opt={args.opt}. Did you forget to add a check for this function?")


if __name__ == "__main__":
    main()
