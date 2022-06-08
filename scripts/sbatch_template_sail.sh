#!/bin/bash

#SBATCH --job-name=$JOB_NAME                        # Job name
#SBATCH --mail-type=NONE                            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --nodes=1                                   # Run all processes on a single node
#SBATCH --ntasks=1                                  # Run on a single CPU
#SBATCH --cpus-per-task=8                           # Number of CPU cores per task
#SBATCH --mem=32GB                                  # Job memory request
#SBATCH --time=96:00:00                             # Time limit hrs:min:sec
#SBATCH --output=$LOG_FILE                          # Standard output and error log
#SBATCH --partition=jag-hi,jag-important,jag-standard  # Request a specific partition for the resource allocation
#SBATCH --nodelist=jagupard16                       # Specifies specific machine(s) to use
#SBATCH --gres=gpu:1                                # Specifies a comma-delimited list of generic consumable resources

export OMP_NUM_THREADS=8                            # Set parallel threads to --cpus-per-task

cd /u/nlp/data/$USER/mml-robustness

# Activate the training environment: CUDA 11.1 should be compatible with Jaguar GPUs (P8)
if [[ $USER == "vliu" ]]; then
    export PATH=/usr/local/cuda-11.1/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
fi

conda activate mml-robustness-2

# Use this alias to point `python` to `python3` so we don't have to change all command submissions ...
# However, this is a bash alias, so this might not be picked by nested subprocess calls (i.e. train_jtt)
alias python=/sailhome/$USER/miniconda3/envs/mml-robustness/bin/python3

# For python-installed commands like pylint,yapf,isort, since conda is probably installed in a non-default folder
export PATH=/sailhome/$USER/miniconda3/envs/mml-robustness/bin${PATH:+:${PATH}}

nvidia-smi
$COMMAND
