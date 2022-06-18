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

# used_for_tuning = ["Blond_Hair", "Young", "Smiling", "Oval_Face", "Pointy_Nose", "Attractive"]
used_for_tuning = []  #if we simply want to re-run for all attributes
USER = os.environ["USER"]

PARAMS = {
    # STL baseline methods
    "erm": {
        "WD": 1e-1,
        "LR": 1e-4,
        "BATCH_SIZE": 128,
        "EPOCHS": 50,
    },
    "suby": {
        "WD": 1e-2,
        "LR": 1e-3,
        "BATCH_SIZE": 128,
        "EPOCHS": 50,
    },
    "rwy": {
        "WD": 1e-2,
        "LR": 1e-4,
        "BATCH_SIZE": 2,
        "EPOCHS": 50,
    },
    "clip_erm": {
        "WD": 1e-1,
        "LR": 1e-4,
        "BATCH_SIZE": 128,  # BATCH_SIZE wasn't tuned for CLIP tuning
        "EPOCHS": 50
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["debug", "shell", "sbatch"], default="debug", help="Spawn job mode")
    parser.add_argument("--opt", type=str, required=True, help="Type of optimization run to spawn")
    parser.add_argument(
        "--model", type=str, default="resnet50", choices=["resnet50", "clip_resnet50"], help="Model architecture to use."
    )

    # No need to change any of these tbh
    parser.add_argument(
        "--template", type=str, default="scripts/sbatch_template_rice.sh", required=False, help="SBATCH template file"
    )
    parser.add_argument("--slurm_logs", type=str, default="slurm_logs", required=False, help="Directory to output slurm logs")
    parser.add_argument("--log_dir", type=str, required=False, default="./logs/spurious_id", help="Log directory for all runs")
    return parser.parse_args()


def run_spurious_id(args, attributes_to_train):
    if args.model == "resnet50":
        wd = PARAMS[args.opt]["WD"]
        lr = PARAMS[args.opt]["LR"]
        batch_size = PARAMS[args.opt]["BATCH_SIZE"]
        epochs = PARAMS[args.opt]["EPOCHS"]
    elif args.model == "clip_resnet50":
        wd = PARAMS[f"clip_{args.opt}"]["WD"]
        lr = PARAMS[f"clip_{args.opt}"]["LR"]
        batch_size = PARAMS[f"clip_{args.opt}"]["BATCH_SIZE"]
        epochs = PARAMS[f"clip_{args.opt}"]["EPOCHS"]

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs)
    for attribute in attributes_to_train:
        job_name = f"task:{attribute},wd:{wd},lr:{lr}"
        log_file = os.path.join(args.slurm_logs, f"{job_name}.log")

        command = (
            "python train_erm.py exp.dataset.subgroup_labels=False "
            f"exp={args.opt} "
            f"exp.model.name={args.model} "
            f"exp.dataset.groupings='[{attribute}:Blond_Hair]' "
            f"exp.dataloader.batch_size={batch_size} "
            f"exp.optimizer.lr={lr} "
            f"exp.optimizer.weight_decay={wd} "
            f"exp.train.total_epochs={epochs} "
            f"exp.train.log_dir=\\'{os.path.join(args.log_dir, job_name)}\\'"
        )

        job_manager.submit(command, job_name=job_name, log_file=log_file)


def main():
    args = parse_args()

    attributes_to_train = sorted(set(attributes).difference(used_for_tuning))
    print(f"Number of tasks: {len(attributes_to_train)}")

    assert args.opt in ["erm", "suby", "rwy"]
    run_spurious_id(args, attributes_to_train)


if __name__ == "__main__":
    main()
