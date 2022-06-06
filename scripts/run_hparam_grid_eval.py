"""
Handy script to run jobs for hyperparameter grid searches

Sample usage:
python -m scripts.run_hparam_grid_eval \
    --template ./scripts/sbatch_template_rice.sh \
    --mode sbatch \
    --slurm_logs ./slurm_logs \
    --opt suby
"""

import argparse
import json
import os

from scripts.find_best_ckpt import main as find_best_ckpt
from scripts.job_manager import JobManager
from scripts.run_hparam_grid_train import flatten, GRIDS, TASKS

USER = os.environ["USER"]
LOG_DIR = "./logs"
SEED_GRID = [0, 1, 2]


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


def submit_stl_tune_val(args):
    """Aggregates validation JSONs for (ERM, SUBY) STL runs"""

    TASK_GRID = TASKS["SPURIOUS_ID_TUNE"][0]  # first pair?

    assert args.opt.endswith("_tune"), "This method should only be called with --opt=.*_tune"
    method = args.opt.replace("_tune", "")
    assert method in ["erm", "suby"]  # just to double check

    WD_GRID = GRIDS[method]["WD"]
    LR_GRID = GRIDS[method]["LR"]
    BATCH_SIZE_GRID = GRIDS[method]["BATCH_SIZE"]

    # Support different naming conventions from earlier in the codebase
    # NOTE that the old naming conventions don't include the method in the name ...
    if method == "erm":
        job_name_generator = lambda task, wd, lr, batch_size: f"eval_task:{task},wd:{wd},lr:{lr}"
    elif method == "suby":
        job_name_generator = lambda task, wd, lr, batch_size: f"eval_task:{task},wd:{wd},lr:{lr},batch_size:{batch_size}"
    else:
        raise ValueError("Only --opt=erm and --opt=suby are supported in this function.")

    for task in TASK_GRID:
        for wd in WD_GRID:
            for lr in LR_GRID:
                for batch_size in BATCH_SIZE_GRID:
                    job_name = job_name = job_name_generator(task, wd, lr, batch_size)
                    ckpt_num = find_best_ckpt(f'./logs/{job_name[5:]}', run_test=False, test_groupings="", metric="avg")

                    with open(os.path.join(f"./logs/{job_name[5:]}", "results", f"val_stats_{ckpt_num}.json"), "r") as f:
                        best_val_stats = json.load(f)

                    with open(os.path.join(f"./logs/{job_name[5:]}", "results", f"best_val_stats_{ckpt_num}.json"), "w") as fp:
                        json.dump(best_val_stats, fp)


def submit_mtl_tune_val(args):
    TASK = TASKS["MTL_ABLATE_DISJOINT"][0]  # first pair ?

    assert args.opt.startswith("mtl_") and args.opt.endswith("_tune"), \
        "This method should only be called with --opt=mtl_.*_tune"
    mtl_method = args.opt.replace("_tune", "")
    method = mtl_method.replace("mtl_", "")

    WD_GRID = GRIDS[method]["WD"]
    LR_GRID = GRIDS[method]["LR"]
    BATCH_SIZE_GRID = GRIDS[method]["BATCH_SIZE"]

    if args.mtl_checkpoint_type is None:
        raise ValueError(f"Please specify an option for --{args.mtl_checkpoint_type}")

    for wd in WD_GRID:
        for lr in LR_GRID:
            for batch_size in BATCH_SIZE_GRID:
                for seed in SEED_GRID:
                    for metric in ["avg", "group"]:
                        job_name = f"eval_mtl_tuning:{method},task:{len(TASK)}_tasks,{args.mtl_weighting}_task_weighting,seed:{seed},wd:{wd},lr:{lr},batch_size:{batch_size}"
                        save_json = f"val_stats_{metric}_checkpoint_{args.mtl_checkpoint_type}_mtl_type.json"

                        ckpt_num = find_best_ckpt(
                            os.path.join(LOG_DIR, job_name[5:]),
                            run_test=False,
                            test_groupings="",
                            metric=metric,
                            learning_type="mtl",
                            mtl_checkpoint_type=f"{args.mtl_checkpoint_type}"
                        ) - 1
                        with open(os.path.join(f"./logs/{job_name[5:]}", "results", f"val_stats_{ckpt_num}.json"), "r") as f:
                            best_val_stats = json.load(f)

                        with open(os.path.join(f"./logs/{job_name[5:]}", "results", save_json), "w") as fp:
                            json.dump(best_val_stats, fp)


def submit_stl_test(args):
    method = args.opt
    assert method in ["erm", "suby", "rwy", "jtt"]  # just to double check

    if method in ["suby", "rwy", "jtt"]:
        TASK_GRID = flatten(TASKS["MTL_DISJOINT"])
    else:  # we want to run erm on everything
        TASK_GRID = set(flatten(TASKS["MTL_DISJOINT"] + TASKS["MTL_NONDISJOINT"] + TASKS["MTL_SIMILAR"] + TASKS["MTL_STRONG"]))
    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)

    for task in TASK_GRID:
        for seed in SEED_GRID:
            for checkpoint_type in ["avg", "group"]:
                job_name = f"eval_baseline:{method},task:{task},seed:{seed}"
                log_file = os.path.join(args.slurm_logs, f"{job_name}.log")
                save_json = f"test_stats_{checkpoint_type}_checkpoint.json"

                log_dir = os.path.join(LOG_DIR, job_name[5:])
                if not os.path.exists(log_dir):
                    continue
                if method == "jtt":
                    log_dir = os.path.join(log_dir, "stage_2")
                results_dir = os.path.join(log_dir, "results")
                save_json = os.path.join(results_dir, save_json)

                command = f"python -m scripts.find_best_ckpt --run_test --log_dir {log_dir} --metric {checkpoint_type} --learning_type stl --save_json {save_json}"
                job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_mtl_cvx_disjoint_tasks_test(args):
    TASK = TASKS["MTL_DISJOINT"][0]  # SINGLE PAIR
    CVX_GRID = ["qp", "maxent"]

    assert args.opt in ["mtl_rwy", "mtl_suby"], "This method only supports --opt=mtl_rwy and --opt=mtl_suby"
    method = args.opt.replace("mtl_", "")

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)

    for seed in SEED_GRID:
        for cvx in CVX_GRID:
            for checkpoint_type in ["group", "avg"]:
                job_name = f"eval_mtl_train:{method},task:{len(TASK)}_tasks,{args.mtl_weighting}_task_weighting,seed:{seed},cvx:{cvx}"
                log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

                save_json = f"test_stats_{checkpoint_type}_checkpoint_{args.mtl_checkpoint_type}_mtl_type.json"
                log_dir = os.path.join(LOG_DIR, job_name[5:])
                results_dir = os.path.join(log_dir, "results")
                save_json = os.path.join(results_dir, save_json)

                command = (
                    f"python -m scripts.find_best_ckpt --run_test --learning_type mtl "
                    f"--log_dir {log_dir} "
                    f"--metric {checkpoint_type} "
                    f"--mtl_checkpoint_type {args.mtl_checkpoint_type} "
                    f"--save_json {save_json}"
                )
                job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_mtl_jtt_disjoint_tasks_test(args):
    TASK = TASKS["MTL_DISJOINT"][0]  # SINGLE PAIR
    assert args.opt in ["mtl_jtt"], "This method only supports --opt=mtl_jtt"

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)
    for seed in SEED_GRID:
        for checkpoint_type in ["avg", "group"]:
            job_name = f"eval_mtl_train:jtt,task:{len(TASK)}_tasks_{args.mtl_weighting}_task_weighting,seed:{seed}"
            log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

            save_json = f"test_stats_{checkpoint_type}_checkpoint_{args.mtl_checkpoint_type}_mtl_type.json"

            log_dir = os.path.join(LOG_DIR, "mtl_jtt", job_name[5:], "stage_2")
            results_dir = os.path.join(log_dir, "results")
            save_json = os.path.join(results_dir, save_json)

            command = (
                f"python -m scripts.find_best_ckpt --run_test --learning_type mtl "
                f"--log_dir {log_dir} "
                f"--metric {checkpoint_type} "
                f"--save_json {save_json} "
                f"--mtl_checkpoint_type {args.mtl_checkpoint_type}"
            )
            job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_mtl_erm_ablate_disjoint_tasks_test(args):
    TASK_GRID = TASKS["MTL_ABLATE_DISJOINT"]

    assert args.opt in ["mtl_erm_ablate_disjoint"]
    mtl_method = args.opt.replace("_ablate_disjoint", "")
    method = mtl_method.replace("mtl_", "")

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)

    for seed in SEED_GRID:
        for idx, task in enumerate(TASK_GRID):
            for checkpoint_type in ["group", "avg"]:
                #job_name = f"eval_mtl_train:{method},task:{len(task)}_tasks,task_ablation_disjoint_idx:{idx + 1},{args.mtl_weighting}_task_weighting,seed:{seed}"
                job_name = f"eval_mtl_train:erm,task:3_tasks,disjoint_idx:0,static_equal_task_weighting,seed:{seed}"
                log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

                save_json = f"test_stats_{checkpoint_type}_checkpoint_{args.mtl_checkpoint_type}_mtl_type.json"
                log_dir = os.path.join(LOG_DIR, job_name[5:])
                results_dir = os.path.join(log_dir, "results")
                save_json = os.path.join(results_dir, save_json)

                command = (
                    f"python -m scripts.find_best_ckpt --run_test --learning_type mtl "
                    f"--log_dir {log_dir} "
                    f"--metric {checkpoint_type} "
                    f"--mtl_checkpoint_type {args.mtl_checkpoint_type} "
                    f"--save_json {save_json}"
                )
                job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_mtl_erm_ablate_nondisjoint_tasks_test(args):
    TASK_GRID = TASKS["MTL_ABLATE_NONDISJOINT"]

    assert args.opt in ["mtl_erm_ablate_nondisjoint"]
    mtl_method = args.opt.replace("_ablate_nondisjoint", "")
    method = mtl_method.replace("mtl_", "")

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)

    for seed in SEED_GRID:
        for idx, task in enumerate(TASK_GRID):
            for checkpoint_type in ["group", "avg"]:
                job_name = f"eval_mtl_train:{method},task:{len(task)}_tasks,task_ablation_nondisjoint_idx:{idx},{args.mtl_weighting}_task_weighting,seed:{seed}"
                log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

                save_json = f"test_stats_{checkpoint_type}_checkpoint_{args.mtl_checkpoint_type}_mtl_type.json"
                log_dir = os.path.join(LOG_DIR, job_name[5:])
                results_dir = os.path.join(log_dir, "results")
                save_json = os.path.join(results_dir, save_json)

                command = (
                    f"python -m scripts.find_best_ckpt --run_test --learning_type mtl "
                    f"--log_dir {log_dir} "
                    f"--metric {checkpoint_type} "
                    f"--mtl_checkpoint_type {args.mtl_checkpoint_type} "
                    f"--save_json {save_json}"
                )
                job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_mtl_erm_disjoint_tasks_test(args):
    TASK_GRID = TASKS["MTL_DISJOINT"]  # all, but we need the pair indices, so we just skip the 0th one

    assert args.opt in ["mtl_erm_disjoint"]
    mtl_method = args.opt.replace("_disjoint", "")
    method = mtl_method.replace("mtl_", "")

    if args.mtl_checkpoint_type is None:
        raise ValueError(f"Please specify an option for --mtl_checkpoint_type={args.mtl_checkpoint_type}")

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)

    for seed in SEED_GRID:
        for idx, task in enumerate(TASK_GRID):
            if idx == 0:
                continue  # skip the one we tuned on

            for checkpoint_type in ["avg", "group"]:
                # Appending of ,ckpt:{ckpt} in the job_name might be new, deprecated naming doesn't have this ...?
                job_name = f"eval_mtl_train:{method},task:{len(task)}_tasks,disjoint_idx_{idx},{args.mtl_weighting}_task_weighting,seed:{seed}"
                log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

                save_json = f"test_stats_{checkpoint_type}_checkpoint_{args.mtl_checkpoint_type}_mtl_type.json"

                log_dir = os.path.join(LOG_DIR, job_name[5:])
                results_dir = os.path.join(log_dir, "results")
                save_json = os.path.join(results_dir, save_json)

                command = (
                    f"python -m scripts.find_best_ckpt --run_test --learning_type mtl "
                    f"--log_dir {log_dir} "
                    f"--metric {checkpoint_type} "
                    f"--save_json {save_json} "
                    f"--mtl_checkpoint_type {args.mtl_checkpoint_type}"
                )
                job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_mtl_erm_nondisjoint_tasks_test(args):
    TASK_GRID = TASKS["MTL_NONDISJOINT"]

    assert args.opt in ["mtl_erm_nondisjoint"]
    mtl_method = args.opt.replace("_nondisjoint", "")
    method = mtl_method.replace("mtl_", "")

    if args.mtl_checkpoint_type is None:
        raise ValueError(f"Please specify an option for --mtl_checkpoint_type={args.mtl_checkpoint_type}")

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)

    for seed in SEED_GRID:
        for idx, task in enumerate(TASK_GRID):
            for checkpoint_type in ["avg", "group"]:
                job_name = f"eval_mtl_train:{method},task:{len(task)}_tasks,nondisjoint_idx_{idx},{args.mtl_weighting}_task_weighting,seed:{seed}"
                log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

                save_json = f"test_stats_{checkpoint_type}_checkpoint_{args.mtl_checkpoint_type}_mtl_type.json"

                log_dir = os.path.join(LOG_DIR, job_name[5:])
                results_dir = os.path.join(log_dir, "results")
                save_json = os.path.join(results_dir, save_json)

                command = (
                    f"python -m scripts.find_best_ckpt --run_test --learning_type mtl "
                    f"--log_dir {log_dir} "
                    f"--metric {checkpoint_type} "
                    f"--save_json {save_json} "
                    f"--mtl_checkpoint_type {args.mtl_checkpoint_type}"
                )
                job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_mtl_erm_similar_tasks_test(args):
    TASK_GRID = TASKS["MTL_SIMILAR"]  # all, but we need the pair indices, so we just skip the 0th one

    assert args.opt in ["mtl_erm_similar"]
    mtl_method = args.opt.replace("_similar", "")
    method = mtl_method.replace("mtl_", "")

    if args.mtl_checkpoint_type is None:
        raise ValueError(f"Please specify an option for --mtl_checkpoint_type={args.mtl_checkpoint_type}")

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)

    for seed in SEED_GRID:
        for idx, task in enumerate(TASK_GRID):
            for checkpoint_type in ["avg", "group"]:
                job_name = f"eval_mtl_train:{method},task:{len(task)}_tasks,semantic_similar:{idx + 1},{args.mtl_weighting}_task_weighting,seed:{seed}"
                log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

                save_json = f"test_stats_{checkpoint_type}_checkpoint_{args.mtl_checkpoint_type}_mtl_type.json"

                log_dir = os.path.join(LOG_DIR, job_name[5:])
                results_dir = os.path.join(log_dir, "results")
                save_json = os.path.join(results_dir, save_json)

                command = (
                    f"python -m scripts.find_best_ckpt --run_test --learning_type mtl "
                    f"--log_dir {log_dir} "
                    f"--metric {checkpoint_type} "
                    f"--save_json {save_json} "
                    f"--mtl_checkpoint_type {args.mtl_checkpoint_type}"
                )
                job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_mtl_erm_strong_tasks_test(args):
    TASK_GRID = TASKS["MTL_STRONG"]  # all, but we need the pair indices, so we just skip the 0th one

    assert args.opt in ["mtl_erm_strong"]
    mtl_method = args.opt.replace("_strong", "")
    method = mtl_method.replace("mtl_", "")

    if args.mtl_checkpoint_type is None:
        raise ValueError(f"Please specify an option for --mtl_checkpoint_type={args.mtl_checkpoint_type}")

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)

    for seed in SEED_GRID:
        for idx, task in enumerate(TASK_GRID):
            for checkpoint_type in ["avg", "group"]:
                job_name = f"eval_mtl_train:{method},task:{len(task)}_tasks,strong_spurious_correlation_idx:{idx},{args.mtl_weighting}_task_weighting,seed:{seed}"
                log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

                save_json = f"test_stats_{checkpoint_type}_checkpoint_{args.mtl_checkpoint_type}_mtl_type.json"

                log_dir = os.path.join(LOG_DIR, job_name[5:])
                results_dir = os.path.join(log_dir, "results")
                save_json = os.path.join(results_dir, save_json)

                command = (
                    f"python -m scripts.find_best_ckpt --run_test --learning_type mtl "
                    f"--log_dir {log_dir} "
                    f"--metric {checkpoint_type} "
                    f"--save_json {save_json} "
                    f"--mtl_checkpoint_type {args.mtl_checkpoint_type}"
                )
                job_manager.submit(command, job_name=job_name, log_file=log_file)


def submit_mtl_erm_weak_tasks_test(args):
    TASK_GRID = TASKS["MTL_WEAK"]  # all, but we need the pair indices, so we just skip the 0th one

    assert args.opt in ["mtl_erm_weak"]
    mtl_method = args.opt.replace("_weak", "")
    method = mtl_method.replace("mtl_", "")

    if args.mtl_checkpoint_type is None:
        raise ValueError(f"Please specify an option for --mtl_checkpoint_type={args.mtl_checkpoint_type}")

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)

    for seed in SEED_GRID:
        for idx, task in enumerate(TASK_GRID):
            for checkpoint_type in ["avg", "group"]:
                job_name = f"eval_mtl_train:{method},task:{len(task)}_tasks,weak_spurious_correlation_idx:{idx},{args.mtl_weighting}_task_weighting,seed:{seed}"
                log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

                save_json = f"test_stats_{checkpoint_type}_checkpoint_{args.mtl_checkpoint_type}_mtl_type.json"

                log_dir = os.path.join(LOG_DIR, job_name[5:])
                results_dir = os.path.join(log_dir, "results")
                save_json = os.path.join(results_dir, save_json)

                command = (
                    f"python -m scripts.find_best_ckpt --run_test --learning_type mtl "
                    f"--log_dir {log_dir} "
                    f"--metric {checkpoint_type} "
                    f"--save_json {save_json} "
                    f"--mtl_checkpoint_type {args.mtl_checkpoint_type}"
                )
                job_manager.submit(command, job_name=job_name, log_file=log_file)


def main():
    args = parse_args()

    #######################
    # [0] STL SPURIOUS ID #
    #######################

    # Evals STL tuning methods on the list of spurious ID pairs
    if args.opt in ["erm_tune", "suby_tune"]:
        submit_stl_tune_val(args)

    ####################
    # [1] TUNE MTL ERM #
    ####################

    # Evals MTL tuning methods on the first disjoint pair
    elif args.opt in ["mtl_erm_tune", "mtl_suby_tune"]:
        submit_mtl_tune_val(args)

    ##################
    # [2] MTL VS STL #
    ##################

    # Evals STL methods on all tasks in flattened disjoint pairs
    elif args.opt in ["erm", "suby", "rwy", "jtt"]:
        submit_stl_test(args)

    # Evals MTL methods with CVX optimization on the first disjoint pair
    elif args.opt in ["mtl_rwy", "mtl_suby"]:
        submit_mtl_cvx_disjoint_tasks_test(args)

    elif args.opt in ["mtl_jtt"]:
        submit_mtl_jtt_disjoint_tasks_test(args)

    #########################
    # [3] MTL TASK ABLATION #
    #########################

    # Evals MTL methods on task sizes from 3-6 for disjoint
    elif args.opt in ["mtl_erm_ablate_disjoint"]:
        submit_mtl_erm_ablate_disjoint_tasks_test(args)

    # Evals MTL methods on the lasts 4 disjoint task pairs
    elif args.opt in ["mtl_erm_disjoint"]:
        submit_mtl_erm_disjoint_tasks_test(args)

    # Evals MTL methods on task sizes from 3-6 for nondisjoint
    elif args.opt in ["mtl_erm_ablate_nondisjoint"]:
        submit_mtl_erm_ablate_nondisjoint_tasks_test(args)

    ###############################
    # [4] DISJOINT VS NONDISJOINT #
    ###############################

    elif args.opt in ["mtl_erm_disjoint"]:
        submit_mtl_erm_disjoint_tasks_test(args)

    elif args.opt in ["mtl_erm_nondisjoint"]:
        submit_mtl_erm_nondisjoint_tasks_test(args)

    #############################
    # [5] SIMILAR VS NONSIMILAR #
    #############################

    elif args.opt in ["mtl_erm_similar"]:
        submit_mtl_erm_similar_tasks_test(args)

    ###############################
    # [6] STRONG VS WEAK SPURIOUS #
    ###############################

    elif args.opt in ["mtl_erm_strong"]:
        submit_mtl_erm_strong_tasks_test(args)

    elif args.opt in ["mtl_erm_weak"]:
        submit_mtl_erm_weak_tasks_test(args)

    else:
        raise ValueError(f"Didn't recognize opt={args.opt}. Did you forget to add a check for this function?")


if __name__ == "__main__":
    main()
