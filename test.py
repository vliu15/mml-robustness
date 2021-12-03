"""Script for running subgroup evaluation on test set"""

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

from utils.init_modules import init_model, init_test_dataloader
from utils.train_utils import to_device

import warnings
warnings.filterwarnings("ignore")

logging.config.fileConfig("logger.conf")


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
        "--extra_groupings",
        required=False,
        default="",
        type=str,
        help="Comma-separated list of additional group ids to run test on",
    )
    return parser.parse_args()


def main():
    """Entry point into testing script"""
    args = parse_args()
    config = OmegaConf.load(os.path.join(args.log_dir, "config.yaml"))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Init and load model
    model = init_model(config).to(device)
    ckpt = torch.load(os.path.join(args.log_dir, "ckpts", f"ckpt.{args.ckpt_num}.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Run test on trained grouping
    test_dataloader = init_test_dataloader(config)
    evaluate(
        config=config,
        model=model,
        test_dataloader=test_dataloader,
        device=device,
        results_json=f"test_results_{config.dataset.grouping}.json",
    )

    # Run test on additional groupings
    extra_groupings = args.extra_groupings.split(",")
    for grouping in extra_groupings:
        config.dataset.grouping = [int(grouping)]

        test_dataloader = init_test_dataloader(config)
        evaluate(
            config=config,
            model=model,
            test_dataloader=test_dataloader,
            device=device,
            results_json=f"test_results_{config.dataset.grouping}.json",
        )

@torch.no_grad()
def evaluate(
    config: DictConfig,
    model: nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    device: str,
    results_json: str = "test_results.json",
):

    test_stats = defaultdict(float)
    for batch in tqdm(
        test_dataloader,
        total=len(test_dataloader),
        desc=f"Running eval on grouping {config.dataset.grouping}",
        leave=False,
    ):
        # Forward pass
        batch = to_device(batch, device)
        out_dict = model.supervised_step(batch, subgroup=args.subgroup_labels, first_batch_loss = None)

        # Accumulate metrics
        for key in out_dict.keys():
            if key.startswith("metric"):
                if "avg" in key:
                    test_stats[key] += out_dict[key].item() * batch[0].shape[0]
                elif "counts" in key:
                    test_stats[key] += out_dict[key]

    # Compute final metrics
    print("Test Set Results:")
    for key in list(test_stats.keys()):
        if key.startswith("metric"):
            if "avg" in key:
                test_stats[key[7:]] = test_stats.pop(key) / len(test_dataloader.dataset)
                print(f"  {key[7:]}: {100 * test_stats[key[7:]]:.4f}%")
            elif "counts" in key:
                correct, total = test_stats.pop(key)
                test_stats[key[7:-7]] = float(correct / total)
                print(f"  {key[7:-7]}: {100 * test_stats[key[7:-7]]:.4f}%")

    # Write results to json

    identifiers = []
    for key in config.dataset.subgroup_attributes.keys():
        identifiers.append(key)
        for attr in config.dataset.subgroup_attributes[key]:
            identifiers.append(attr)
    correlates = "_".join(identifiers)

    results_dir = os.path.join(args.log_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
<<<<<<< cf537b8e5f03e4e3de3937484741bda8fb2a178c
    results_file = os.path.join(results_dir, f"{correlates}_test_results.json")
=======
    results_file = os.path.join(results_dir, results_json)
>>>>>>> Move grouping definitions into python, add batch test script
    with open(results_file, 'w') as fp:
        json.dump(test_stats, fp)


if __name__ == "__main__":
    main()
