import argparse
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
from trulens.nn.attribution import InputAttribution, IntegratedGradients
from trulens.nn.models import get_model_wrapper
from trulens.visualizations import MaskVisualizer

from fairness.filtered_dataset import FilteredDataset
from utils.init_modules import init_datasets, init_model

attributes = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose",
    "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee",
    "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair",
    "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"
]

mask_visualizer_kwargs = {
    "blur": 5,
    "threshold": 0.9,
    "masked_opacity": 0.5,
    "combine_channels": True,
    "use_attr_as_opacity": False,
    "positive_only": True,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True, type=str, help="Path to log directory")
    parser.add_argument("--ckpt_num", required=True, type=int, help="Checkpoint number to load")
    parser.add_argument("--attr", required=True, type=str, help="Attribute to evaluate against")

    parser.add_argument("--index", required=False, default=0, type=int, help="Index of example to get per group")

    parser.add_argument(
        "--out_dir", required=False, type=str, default="./outputs/attributions", help="Directory to save saliency map plots"
    )
    return parser.parse_args()


def get_attributions(config, attribution, model, dataset, label, attr, save_name, index=0, device="cuda"):
    for l in [0, 1]:
        for a in [0, 1]:
            plt.clf()
            plt.cla()

            filtered_dataset = FilteredDataset(config, dataset, filters={label: l, attr: a})

            x_orig = filtered_dataset.get_original_image(index)
            x_orig = x_orig.unsqueeze(0).numpy()

            _, x, y, _, _ = filtered_dataset.__getitem__(index)
            x = x.unsqueeze(0).numpy()

            with torch.no_grad():
                y_hat = model(torch.from_numpy(x).to(device))
                y_hat = (y_hat > 0).to(torch.int32).reshape(1).item()
                y = int(y.reshape(1).item())
                print(f"Label: {y}. Prediction: {y_hat}")

            attrs_input = attribution.attributions(x)
            masked_image = MaskVisualizer(**mask_visualizer_kwargs)(attrs_input, x_orig)

            ax = plt.gca()
            ax.imshow(masked_image)
            ax.set_title(f"{label}={bool(l)}, {attr}={bool(a)}, Predicted_correctly={y == y_hat}")
            plt.tight_layout()
            plt.savefig(f"{save_name},l:{l},a:{a}.png")


def main():
    args = parse_args()

    # Load config and prepare attributes
    config = OmegaConf.load(os.path.join(args.log_dir, "config.yaml"))
    label = config.dataset.groupings[0].split(":")[0]

    # Set up output directory
    out_dir = os.path.join(args.out_dir, label)
    os.makedirs(out_dir, exist_ok=True)

    # Load pytorch model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = init_model(config, device=device).eval()
    ckpt = torch.load(os.path.join(args.log_dir, "ckpts", f"ckpt.{args.ckpt_num}.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])
    model = model.resnet

    # Load attribution wrappers
    model_wrapper = get_model_wrapper(
        deepcopy(model),
        device="cpu",  # throws error on cuda
        input_shape=[3] + config.dataset.target_resolution,
        logit_layer=-1,
        backend="pytorch",
    )
    saliency_map_attribution = InputAttribution(model_wrapper)
    integrated_gradient_attribution = IntegratedGradients(model_wrapper)

    # Generate attributions
    _, val_dataset = init_datasets(config)
    get_attributions(
        config,
        saliency_map_attribution,
        model,
        val_dataset,
        label,
        args.attr,
        os.path.join(out_dir, f"attr:{args.attr}_saliency_maps"),
        index=args.index,
        device=device
    )
    get_attributions(
        config,
        integrated_gradient_attribution,
        model,
        val_dataset,
        label,
        args.attr,
        os.path.join(out_dir, f"attr:{args.attr}_integrated_grads.png"),
        index=args.index,
        device=device
    )


if __name__ == "__main__":
    main()
