#!/bin/bash

#SBATCH --job-name=$JOB_NAME        # Job name
#SBATCH --mail-type=NONE            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --nodes=1                   # Run all processes on a single node
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --cpus-per-task=8           # Number of CPU cores per task
#SBATCH --mem=32GB                  # Job memory request
#SBATCH --time=48:00:00             # Time limit hrs:min:sec
#SBATCH --output=$LOG_FILE          # Standard output and error log
#SBATCH --partition=gpu             # Request a specific partition for the resource allocation
#SBATCH --gres=gpu:1                # Specifies a comma-delimited list of generic consumable resources
#SBATCH --qos=gpu                   # Request a quality of service for the job

export OMP_NUM_THREADS=8            # Set parallel threads to --cpus-per-task

source /home/jsparmar/.bashrc         # Load anaconda and other jazz
conda activate mml-robustness               # Activate the training environment

$COMMAND
