#!/bin/bash

#SBATCH --job-name=$JOB_NAME        # Job name
#SBATCH --mail-type=NONE            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --nodes=1                   # Run all processes on a single node
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --cpus-per-task=8           # Number of CPU cores per task
#SBATCH --mem=32GB                  # Job memory request
#SBATCH --time=48:00:00             # Time limit hrs:min:sec
#SBATCH --output=$LOG_FILE          # Standard output and error log
#SBATCH --partition=gpu,normal      # Request a specific partition for the resource allocation
#SBATCH --gres=gpu:1                # Specifies a comma-delimited list of generic consumable resources

export OMP_NUM_THREADS=8            # Set parallel threads to --cpus-per-task

# conda init bash
cd /farmshare/user_data/$USER/mml-robustness

# Activate the training environment: CUDA 10.0 should be compatible with Oat GPUs (K40)
if [[ $USER == "vliu15" ]]; then
    export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
fi

conda activate mml-robustness

$COMMAND
