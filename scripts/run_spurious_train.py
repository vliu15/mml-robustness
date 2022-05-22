"""
Spawns jobs for spurious correlation identification

Sample usage:
python -m scripts.run_spurious_train \
    --wd 0.1 \
    --lr 0.0001 \
    --epochs 25 \
    --mode debug \
    --opt erm
"""

import argparse
import os

from scripts.job_manager import JobManager

attributes = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose",
    "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee",
    "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair",
    "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"
]

#used_for_tuning = ["Blond_Hair", "Young", "Smiling", "Oval_Face", "Pointy_Nose", "Attractive"]
used_for_tuning = []  #if we simply want to re-run for all attributes
USER = os.environ["USER"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wd", type=float, required=True, help="Weight decay value")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate value")
    parser.add_argument("--batch_size", type=int, required=False, default=32, help="Batch size to use")
    parser.add_argument("--epochs", type=int, required=False, default=25, help="Number of epochs to train for")

    parser.add_argument("--mode", type=str, choices=["debug", "shell", "sbatch"], default="debug", help="Spawn job mode")
    parser.add_argument("--opt", type=str, required=True, help="Type of optimization run to spawn")

    # No need to change any of these tbh
    parser.add_argument(
        "--template", type=str, default="scripts/sbatch_template_rice.sh", required=False, help="SBATCH template file"
    )
    parser.add_argument("--slurm_logs", type=str, default="slurm_logs", required=False, help="Directory to output slurm logs")
    parser.add_argument("--log_dir", type=str, required=False, default="./logs/spurious_id", help="Log directory for all runs")
    return parser.parse_args()


def erm_spurious_id(args, attributes_to_train):
    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)
    for attribute in attributes_to_train:
        job_name = f"task:{attribute},wd:{args.wd},lr:{args.lr}"
        log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

        command = (
            "python train_erm.py exp=erm "
            f"exp.dataset.groupings='[{attribute}:Blond_Hair]' "
            f"exp.optimizer.lr={args.lr} "
            f"exp.optimizer.weight_decay={args.wd} "
            f"exp.train.total_epochs={args.epochs} "
            f"exp.train.log_dir=\\'{os.path.join(args.log_dir, job_name)}\\' "
            "exp.dataset.subgroup_labels=False"
        )

        job_manager.submit(command, job_name=job_name, log_file=log_file)


def suby_spurious_id(args, attributes_to_train):
    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)

    for attribute in attributes_to_train:
        job_name = f"suby_spurious_id:{attribute},wd:{args.wd},lr:{args.lr},batch_size:{args.batch_size}"
        log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

        command = (
            "python train_erm.py exp=suby "
            f"exp.dataset.groupings='[{attribute}:Blond_Hair]' "
            f"exp.optimizer.lr={args.lr} "
            f"exp.optimizer.weight_decay={args.wd} "
            f"exp.train.total_epochs={args.epochs} "
            f"exp.dataloader.batch_size={args.batch_size} "
            f"exp.train.load_ckpt=\\'/farmshare/user_data/{USER}/mml-robustness/logs/spurious_id/{job_name}/ckpts/ckpt.54.pt\\' "
            f"exp.train.log_dir=\\'{os.path.join(args.log_dir, job_name)}\\' "
            "exp.dataset.subgroup_labels=False"
        )

        job_manager.submit(command, job_name=job_name, log_file=log_file)


def main():
    args = parse_args()

    attributes_to_train = sorted(set(attributes).difference(used_for_tuning))
    print(f"Number of tasks: {len(attributes_to_train)}")

    if args.opt == "erm":
        erm_spurious_id(args, attributes_to_train)
    elif args.opt == "suby":
        suby_spurious_id(args, attributes_to_train)
    else:
        raise ValueError(f"Didn't recognize opt={args.opt}. Did you forget to add a check for this function?")


if __name__ == "__main__":
    main()
