"""Helper file to submit shell/sbatch jobs"""

import logging
import logging.config
import os
import subprocess
import uuid
import warnings

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


class JobManager(object):
    """Wrapper class to abstract job submissions and management"""

    def __init__(
        self,
        mode: str,
        template: str = "./scripts/sbatch_template_rice.sh",
        slurm_logs: str = "./slurm_logs",
        cpu: bool = False,
    ):
        with open(template, "r") as f:
            self.template = f.read()
        self.slurm_logs = slurm_logs
        os.makedirs(slurm_logs, exist_ok=True)

        assert mode in ["debug", "shell", "sbatch"]
        self.mode = mode
        self.counter = 0

        if cpu:
            if "rice" in template:
                self.template = self.template.replace("--partition=gpu,normal", "--partition=normal")
                self.template = self.template.replace("--gres=gpu:1", "--gres=gpu:0")
            elif "sail" in template:
                self.template = self.template.replace("--partition=jag-urgent,jag-hi,jag-standard", "--partition=john")
                self.template = self.template.replace("--gres=gpu:1", "--gres=gpu:0")
            else:
                warnings.warn(f"No explicit CPU jobs written for SBATCH template {template}")

    def submit(self, command, job_name=None, log_file=None):
        if self.mode == "debug":
            logger.info(command)

        elif self.mode == "shell":
            message = f"RUNNING COMMAND {self.counter}: {command}"
            logger.info(f"{Color.BOLD}{Color.GREEN}{message}{Color.END}{Color.END}")
            subprocess.run(command, shell=True, check=True)

        elif self.mode == "sbatch":
            # Default job name is `sbatch-gpu`
            if job_name is None:
                job_name = "sbatch-gpu"

            # Default log file is `slurm-${JOBID}.out`
            if log_file is None:
                log_file = os.path.join(self.slurm_logs, "slurm-%A.out")

            sbatch = self.template.replace("$JOB_NAME", job_name).replace("$LOG_FILE", log_file).replace("$COMMAND", command)
            uniq_id = uuid.uuid4()
            with open(f"{uniq_id}.sh", "w") as f:
                f.write(sbatch)
            subprocess.run(f"sbatch {uniq_id}.sh", shell=True, check=True)
            os.remove(f"{uniq_id}.sh")

        self.counter += 1
