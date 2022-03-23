"""Script for JTT training"""

import json
import logging
import os
import pickle
import subprocess
from collections import Counter, defaultdict
from functools import reduce
from typing import Dict, List

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from datasets.groupings import get_grouping_object
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

    with tqdm(total=len(train_dataloader), desc=f"[{task}] Constructing error set") as pbar:
        for batch in train_dataloader:
            batch = to_device(batch, device)
            output_dict = model.inference_step(batch)

            _, _, _, g_labels, _ = batch
            g_labels = g_labels[:, task]
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


def merge_error_sets(
    config: DictConfig,
    error_sets: Dict[int, List[int]],
    how: str,
    val_stats: List[Dict[str, float]],
):
    """Merges multiple error sets based on the specified method"""
    # Each method will create a weighted_error_indices dict mapping `index` -> `weight`
    # where `weight` will be scaled by `lambda_up` afterward to upsample `index` correspondingly
    if how == "xor":
        # Take the xor of the indices
        weighted_error_indices = reduce(lambda set1, set2: set(set1) ^ set(set2), error_sets.values())
        # Set the weight for each one to be 1
        weighted_error_indices = {index: 1 for index in weighted_error_indices}

    elif how == "inv":
        # Accumulate the frequencies of each index
        weighted_error_indices = Counter([index for error_set in error_sets.values() for index in error_set])
        # Set the weight for each one to be 1/frequency
        weighted_error_indices = {key: 1. / value for key, value in weighted_error_indices.items()}

    elif how == "task":
        # Grab the final validation losses per task
        final_epoch_val_stats = val_stats[-1]
        grouping = get_grouping_object(config.groupings)
        task_losses = np.array([final_epoch_val_stats[f"loss_{task}"] for task in grouping.task_labels], dtype=np.float32)
        # Normalize
        task_losses /= task_losses.sum()
        # Accumulate by frequency weighted by task loss
        weighted_error_indices = defaultdict(float)
        for i, error_set in error_sets.items():
            for index in error_set:
                weighted_error_indices[index] += task_losses[i]

    else:
        raise ValueError(f"mtl_join_type {how} not recognized")

    # Apply final weighting
    final_error_set = []
    for index, weight in weighted_error_indices.items():
        final_error_set += [index
                           ] * int(weight * (config.lambda_up) - 1)  # lambda_up - 1 since this is additional indices concat
    return final_error_set


@hydra.main(config_path="configs/", config_name="default")
def main(config):
    """Entry point into JTT training"""
    config = config.exp

    ###########
    # Stage 1 #
    ###########
    stage_1_log_dir = os.path.join(config.log_dir, "stage_1")

    if not config.load_up_pkl:
        # 1. Train f_id on D via ERM for T epochs
        groupings = json.dumps(list(config.groupings)).replace(" ", "")
        task_weights = json.dumps([str(w) for w in config.task_weights]).replace(" ", "")
        subprocess.run(
            f"python train_erm.py exp={config.stage_1_config} "
            f"exp.dataset.subgroup_labels=true "
            f"exp.dataset.groupings={groupings} "
            f"exp.dataset.task_weights={task_weights} "
            f"exp.dataset.loss_based_task_weighting={config.loss_based_task_weighting} "
            f"exp.dataset.lbtw_alpha={config.lbtw_alpha} "
            f"exp.train.log_dir=\\'{stage_1_log_dir}\\' "
            f"exp.train.total_epochs={config.epochs_stage_1} "
            f"exp.optimizer.lr={config.lr} "
            f"exp.optimizer.weight_decay={config.weight_decay} "
            f"exp.seed={config.seed} "
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

        # Accumulate error set per task
        error_sets = {}
        pickle_metas = {}
        for task in range(len(config.groupings)):
            error_indices, pickle_meta = construct_error_set(model, train_dataloader, device, task=task)
            error_sets[task] = error_indices
            pickle_metas[f"Task {task}"] = pickle_meta

        # Merge error sets: note that error indices are already upsampled, so dataset doesn't handle upsampling
        if len(config.groupings) > 1:
            with open(os.path.join(config.log_dir, "stage_1", "val_stats.json"), "r") as f:
                val_stats = json.load(f)
            error_indices = merge_error_sets(
                config=config,
                error_sets=error_sets,
                how=config.mtl_join_type,
                val_stats=val_stats,
            )
        else:
            error_indices = list(error_indices) * config.lambda_up

        config.load_up_pkl = os.path.join(config.log_dir, f"jtt_error_set_{config.mtl_join_type}.pkl")
        with open(config.load_up_pkl, "wb") as f:
            pickle.dump({"error_set": error_indices, "meta": pickle_metas}, f)

    ###########
    # Stage 2 #
    ###########
    #torch.cuda.empty_cache()
    stage_2_log_dir = os.path.join(config.log_dir, "stage_2")

    # 3. Construct upsampled dataset D_up containing examples in the error set λ_up times and all other examples once
    # 4. Train final model f_final on D_up via ERM
    groupings = json.dumps(list(config.groupings)).replace(" ", "")
    task_weights = json.dumps([str(w) for w in config.task_weights]).replace(" ", "")
    subprocess.run(
        f"python train_erm.py "
        f"exp={config.stage_2_config} "
        f"exp.dataset.subgroup_labels=true "
        f"exp.dataset.groupings={groupings} "
        f"exp.dataset.task_weights={task_weights} "
        f"exp.dataset.loss_based_task_weighting={config.loss_based_task_weighting} "
        f"exp.dataset.lbtw_alpha={config.lbtw_alpha} "
        f"exp.train.up_type={config.up_type} "
        f"exp.train.load_up_pkl=\\'{config.load_up_pkl}\\' "
        f"exp.train.log_dir=\\'{stage_2_log_dir}\\' "
        f"exp.train.total_epochs={config.epochs_stage_2} "
        f"exp.optimizer.lr={config.lr} "
        f"exp.optimizer.weight_decay={config.weight_decay} "
        f"exp.seed={config.seed} "
        f"exp.train.load_ckpt={config.load_stage_2_ckpt or 'null'}",
        shell=True,
        check=True,
    )


if __name__ == "__main__":
    main()
