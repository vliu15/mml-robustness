"""Script for JTT training"""

import logging
import os
import pickle
import subprocess
from collections import defaultdict

import hydra
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tqdm import tqdm

from utils.init_modules import init_dataloaders, init_model
from utils.train_utils import to_device

logging.config.fileConfig("logger.conf")
logger = logging.getLogger(__name__)


@torch.no_grad()
def construct_error_set(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    device: str,
    task: int = 0,
):
    """Constructs the error set of training examples after initial ERM training"""
    model.eval()

    global_indices = []
    g_totals, g_errors = defaultdict(int), defaultdict(int)

    with tqdm(total=len(train_dataloader), desc="Constructing error set") as pbar:
        for batch in train_dataloader:
            batch = to_device(batch, device)
            output_dict = model.inference_step(batch)

            g_labels = batch[-1]
            g_indices, g_counts = torch.unique(g_labels, return_counts=True)
            for g_index, g_count in zip(g_indices.cpu().tolist(), g_counts.cpu().tolist()):
                g_mask = torch.where(g_labels == g_index)[0]
                yh = (output_dict["yh"][g_mask, task] > 0).float()
                y = output_dict["y"][g_mask, task]

                errors = batch[0][g_mask][yh != y].cpu().tolist()
                global_indices += errors

                g_errors[g_index] += len(errors)
                g_totals[g_index] += g_count

            pbar.update(1)

    # Assemble error rates for all and sub groups
    pickle_meta = {}
    total_examples, total_errors = 0, 0
    for g_index in sorted(g_totals.keys()):
        errors, total = g_errors[g_index], g_totals[g_index]
        total_errors += errors
        total_examples += total
        pickle_meta[g_index] = (errors, total)
    pickle_meta["*"] = (total_errors, total_examples)
    return global_indices, pickle_meta


@hydra.main(config_path="configs/", config_name="default")
def main(config):
    """Entry point into JTT training"""
    config = config.exp

    ###########
    # Stage 1 #
    ###########
    stage_1_log_dir = os.path.join(config.log_dir, "stage_1")

    if not config.load_upweight_pkl:
        # 1. Train f_id on D via ERM for T epochs
        subprocess.run(
            f"python train_erm.py exp={config.stage_1_config} "
            f"exp.train.log_dir={stage_1_log_dir} "
            f"exp.train.load_ckpt={config.load_stage_1_ckpt or 'null'}",
            shell=True,
            check=True,
        )

        # 2. Construct the error set E of training examples misclassified by f_id
        device = "cuda" if torch.cuda.is_available() else "cpu"

        stage_1_config = OmegaConf.load(os.path.join(stage_1_log_dir, "config.yaml"))
        model = init_model(stage_1_config).to(device)
        ckpt = torch.load(os.path.join(stage_1_log_dir, "ckpts", "ckpt.last.pt"))
        model.load_state_dict(ckpt["model"])

        train_dataloader, _ = init_dataloaders(stage_1_config)

        error_indices, pickle_meta = construct_error_set(model, train_dataloader, device, task=config.task)
        config.load_upweight_pkl = os.path.join(config.log_dir, "jtt_error_set.pkl")
        with open(config.load_upweight_pkl, "wb") as f:
            pickle.dump({"error_set": error_indices, "meta": pickle_meta}, f)

    ###########
    # Stage 2 #
    ###########
    stage_2_log_dir = os.path.join(config.log_dir, "stage_2")

    # 3. Construct upsampled dataset D_up containing examples in the error set λ_up times and all other examples once
    # 4. Train final model f_final on D_up via ERM
    subprocess.run(
        f"python train_erm.py "
        f"exp={config.stage_2_config} "
        f"exp.train.lambda_up={config.lambda_up} "
        f"exp.train.load_upweight_pkl={config.load_upweight_pkl} "
        f"exp.train.log_dir={stage_2_log_dir} "
        f"exp.train.load_ckpt={config.load_stage_2_ckpt or 'null'}",
        shell=True,
        check=True,
    )


if __name__ == "__main__":
    main()
