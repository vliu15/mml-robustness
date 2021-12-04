"""Script for running subgroup evaluation on test set"""

import argparse
import json
import logging
import logging.config
import os
import warnings
from collections import defaultdict

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from datasets.groupings import get_grouping_object
from utils.init_modules import init_model, init_test_dataloader
from utils.train_utils import to_device

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
        "--subgroup_attributes",
        required=False,
        default="",
        type=str,
        help="JSON-string list of {{task}}:{{subgroup}}",
    )
    parser.add_argument(
        "--json_name",
        required=False,
        default="test_results.json",
        type=str,
        help="Filename of JSON file in which results are saved into {{args.log_dir}}/results/{{args.json_name}}",
    )
    return parser.parse_args()


def main():
    """Entry point into testing script"""
    args = parse_args()
    config = OmegaConf.load(os.path.join(args.log_dir, "config.yaml"))

    if args.subgroup_attributes:
        config.dataset.groupings = json.loads(args.subgroup_attributes)

    # Init and load model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = init_model(config).to(device)
    ckpt = torch.load(os.path.join(args.log_dir, "ckpts", f"ckpt.{args.ckpt_num}.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    results_dir = os.path.join(args.log_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Run test on trained grouping
    test_dataloader = init_test_dataloader(config)
    evaluate(
        config=config,
        model=model,
        test_dataloader=test_dataloader,
        log_dir=args.log_dir,
        device=device,
        json_file=os.path.join(results_dir, args.json_name),
    )


@torch.no_grad()
def evaluate(
    config: DictConfig,
    model: nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    log_dir: str,
    device: str,
    json_file: str = "test_results.json",
    verbose: bool = False,
):

    test_stats = defaultdict(float)
    for batch in tqdm(
            test_dataloader,
            total=len(test_dataloader),
            desc=f"Running eval on test set",
            leave=False,
    ):
        # Forward pass
        batch = to_device(batch, device)
        out_dict = model.supervised_step(batch, subgroup=True, first_batch_loss=False)

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

    # Save this so we can map g{i} to the corresponding spurious correlation
    test_stats["subgroups"] = get_grouping_object(config.dataset.groupings).subgroup_attributes
    with open(json_file, 'w') as fp:
        json.dump(test_stats, fp)


if __name__ == "__main__":
    main()
