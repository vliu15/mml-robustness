import argparse
import os

from fairness.attributions import attributes
from scripts.job_manager import JobManager


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template", type=str, default="scripts/sbatch_template_sail.sh", required=False, help="SBATCH template file"
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

    parser.add_argument("--log_dir", required=True, type=str, help="Path to log directory")
    parser.add_argument("--ckpt_num", required=True, type=int, help="Checkpoint number to load")

    parser.add_argument(
        "--out_dir", required=False, type=str, default="./outputs/attributions", help="Directory to save saliency map plots"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    job_manager = JobManager(mode=args.mode, template=args.template, slurm_logs=args.slurm_logs, cpu=True)

    for attr in attributes:
        command = (
            "python -m fairness.attributions "
            f"--log_dir {args.log_dir} "
            f"--ckpt_num {args.ckpt_num} "
            f"--out_dir {args.out_dir} "
            f"--attr {attr}"
        )
        job_name = f"attributions:{attr},{os.path.basename(args.log_dir)}"
        log_file = os.path.join(args.slurm_logs, f"{job_name}.log")
        job_manager.submit(command, job_name=job_name, log_file=log_file)


if __name__ == "__main__":
    main()
