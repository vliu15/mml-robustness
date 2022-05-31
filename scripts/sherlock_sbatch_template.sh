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
#SBATCH --gpus 1                    # Specifies GPU

export OMP_NUM_THREADS=8            # Set parallel threads to --cpus-per-task

#load in cuda version
#module load cuda/11.1.1

#load in pytorch for safety
#module load py-pytorch/1.11.0_py39 

#load in conda enviornment
source /home/users/${USER}/.bashrc
conda activate mml-robustness

# conda init bash
cd /home/groups/thashim/mml-robustness

$COMMAND
