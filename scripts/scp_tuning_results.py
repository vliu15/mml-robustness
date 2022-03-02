"""
Handy script to scp results from tuning runs (generally done on rice) to other locations where
development/evaluation is faster. 

If the destination is a gcloud instance please ensure that you have ran gcloud compute ssh on
the source location to generate the needed private key:

Instructions: 

curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-375.0.0-linux-x86_64.tar.gz
tar -xvzf google-cloud-sdk-375.0.0-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh
./google-cloud-sdk/bin/gcloud init
gcloud compute ssh username@project_name
gcloud compute config-ssh

Sample usage:
python ./scripts/scp_tuning_results.py \
    --remote Jupinder_Parmar@mml-robustness.us-west1-b.hai-gcp-robust:
    --dest_loc ~/mml-robustness/logs 
    --source_loc /farmshare/user_data/jsparmar/mml-robustness/logs/spurious_id

   
NOTE: source loc should be a folder that contains results directories within it 
"""
import argparse
import os
import re
import subprocess
import logging
import find_best_ckpt

logging.config.fileConfig("logger.conf")
logger = logging.getLogger(__name__)

class Color(object):
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dest_loc", type=str, default="~/", required=True, help="The folder on the remote machine that the user wishes to port the results to"
    )
    parser.add_argument(
        "--source_loc", type=str, default="./", required=True, help="The folder which contains all results directories from the source that the user seeks to port to the destination"
    )
    parser.add_argument(
        "--remote", type=str, default="user@ip", required=True, help="The remote location to copy files to"
    )
    args = parser.parse_args()

    return args

def port_tuning_results(args):
    for result_dir in os.listdir(args.source_loc):

        results_path = os.path.join(args.source_loc, result_dir)

        if os.path.isdir(results_path):
            destination_path = os.path.join(args.dest_loc, result_dir) 
            dest_folder_command = f'ssh {args.remote} "mkdir -p {destination_path}"'
            subprocess.run(dest_folder_command, shell=True, check=True)
            message = f"SCP FOLDER: {results_path}"
            logger.info(f"{Color.BOLD}{Color.BLUE}{message}{Color.END}{Color.END}")

            best_epoch = find_best_ckpt.main(results_path, run_test=False, test_groupings="", metric="avg")
            best_checkpoint_path_dest = os.path.join(destination_path, 'ckpts') 
            dest_ckpt_folder_command = f'ssh {args.remote} "mkdir -p {best_checkpoint_path_dest}"'
            subprocess.run(dest_ckpt_folder_command, shell=True, check=True)

            best_checkpoint_path_src = os.path.join(results_path, 'ckpts', f'ckpt.{best_epoch}.pt')
            scp_ckpt_command = f'scp {best_checkpoint_path_src} {args.remote}:{best_checkpoint_path_dest}'
            subprocess.run(scp_ckpt_command, shell=True, check=True)

            val_results_path = os.path.join(results_path, 'results') 
            scp_results_command = f'scp -r {val_results_path} {args.remote}:{destination_path}'
            subprocess.run(scp_results_command, shell=True, check=True)

            results_files = (file for file in os.listdir(results_path) if os.path.isfile(os.path.join(results_path, file)))
            for file in results_files:
                file_path = os.path.join(results_path, file) ##scp
                command = f'scp {file_path} {args.remote}:{destination_path}'
                subprocess.run(command, shell=True, check=True)

def main():
    args = parse_args()

    port_tuning_results(args)


if __name__ == "__main__":
    main()
