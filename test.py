"""
Script for testing a model checkpoint on any dataset split

Sample usage:
python test.py \
    --log_dir logs/erm/Blond_Hair:Male \
    --ckpt_num 20 \
    --groupings [Blond_Hair:Male] \
    --split test
"""

import argparse
import json
import logging
import logging.config
import os
from collections import defaultdict

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from utils.init_modules import init_dataloaders, init_model, init_test_dataloader
from utils.train_utils import accumulate_stats, to_device, to_scalar

logging.config.fileConfig("logger.conf")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_dir",
        required=True,
        type=str,
        help="Path to log dir with `ckpts` folder inside",
    )
    parser.add_argument(
        "--ckpt_num",
        required=True,
        type=str,
        help="Checkpoint number to load, corresponds to: `ckpt.{{ckpt_num}}.pt`",
    )
    parser.add_argument(
        "--groupings",
        required=False,
        default="",
        type=str,
        help="JSON-string list of {{task}}:{{subgroup}}",
    )
    parser.add_argument(
        "--split",
        required=False,
        default="val",
        choices=["train", "val", "test"],
        type=str,
        help="Split to run evaluation on"
    )
    parser.add_argument("--save_json", required=False, default="", type=str, help="JSON file to save test results into")
    return parser.parse_args()


def main():
    """Entry point into testing script"""
    args = parse_args()
    config = OmegaConf.load(os.path.join(args.log_dir, "config.yaml"))
    config.dataset.subgroup_labels = True  # cuz some models are trained without subgroup labels (i.e. for spurious ID)

    if args.groupings:
        # Check that the specified groupings contain the exact tasks that were trained on
        new_groupings = json.loads(args.groupings)
        target_tasks = [g.split(":")[0] for g in config.dataset.groupings]
        specified_tasks = [g.split(":")[0] for g in new_groupings]
        assert all(t == s for t, s in zip(target_tasks, specified_tasks)), \
            f"Tasks in training: {target_tasks}. Tasks specified: {specified_tasks}"
        config.dataset.groupings = new_groupings

    # Init and load model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = init_model(config, device=device)
    ckpt = torch.load(os.path.join(args.log_dir, "ckpts", f"ckpt.{args.ckpt_num}.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Run eval on specified split
    if args.split == "train":
        dataloader, _ = init_dataloaders(config)
    elif args.split == "val":
        _, dataloader = init_dataloaders(config)
    else:
        dataloader = init_test_dataloader(config)

    # Defaults to the same save format as val_stats from training
    if args.save_json == "":
        results_dir = os.path.join(args.log_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        args.save_json = os.path.join(results_dir, f"{args.split}_stats_{args.ckpt_num}.json")

    evaluate(
        config=config,
        model=model,
        dataloader=dataloader,
        device=device,
        split=args.split,
        save_json=args.save_json,
    )


@torch.no_grad()
def evaluate(
    config: DictConfig,
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    split: str,
    save_json: str,
):
    losses, metrics = defaultdict(float), defaultdict(float)
    for batch in tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Running eval on {split} split",
            leave=False,
    ):
        # Forward pass
        batch = to_device(batch, device)
        loss_dict, metrics_dict = model.supervised_step(batch, subgroup=config.dataset.subgroup_labels, first_batch_loss=None)
        accumulate_stats(
            loss_dict=loss_dict,
            metrics_dict=metrics_dict,
            accumulated_loss=losses,
            accumulated_metrics=metrics,
            over_n_examples=len(dataloader.dataset),
            batch_size=batch[0].shape[0],
        )

    # Compute final metrics
    logger.info("Metrics on %s set:", split)
    for key in list(metrics.keys()):
        if "counts" in key:
            new_key = key.replace("counts", "acc")
            metrics[new_key] = to_scalar(metrics[key][0] / metrics[key][1]) if metrics[key][1] > 0 else None
            metrics.pop(key)
            key = new_key
        logger.info("%s: %s%", key, 100 * metrics[key])

    with open(save_json, "w") as fp:
        json.dump({**losses, **metrics}, fp)


if __name__ == "__main__":
    main()
