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
        "--subgroup_labels",
        required=False,
        default=False,
        action='store_true',
        help="Whether to evaluate subgroup accuracies",
    )

    parser.add_argument(
        "--no_subgroup_labels",
        required=False,
        action='store_false',
        dest='subgroup_labels',
        help="Whether to not evaluate subgroup accuracies",
    )

    ### must be in the following form: '{"task_label_1": ["Spurrious_1", "Spurrious_2"], ...}'
    parser.add_argument(
        "--subgroup_attributes",
        required=False,
        type=str,
        help="Defines the subgroups to deliniate per task",
    )

    return parser.parse_args()


def main():
    """Entry point into testing script"""
    args = parse_args()
    config = OmegaConf.load(os.path.join(args.log_dir, "config.yaml"))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ### config set subggroup attributes , config set subgroup labes , test that args.subgroup_attirbutees iis as expect, that configg
    ### changes as expected, then good
    config.dataset.subgroup_labels = args.subgroup_labels
    config.dataset.subgroup_attributes = json.loads(args.subgroup_attributes)

    # Init modules
    model = init_model(config).to(device)
    test_dataloader = init_test_dataloader(config)

    # Load checkpoint
    ckpt = torch.load(os.path.join(args.log_dir, "ckpts", f"ckpt.{args.ckpt_num}.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    evaluate(config=config, model=model, test_dataloader=test_dataloader, device=device, args=args)


@torch.no_grad()
def evaluate(config: DictConfig, model: nn.Module, test_dataloader: torch.utils.data.DataLoader, device: str, args):

    test_stats = defaultdict(float)
    for batch in tqdm(test_dataloader, total=len(test_dataloader), desc="Running eval on test set"):
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
    results_file = os.path.join(results_dir, f"{correlates}_test_results.json")
    with open(results_file, 'w') as fp:
        json.dump(test_stats, fp)


if __name__ == "__main__":
    main()
