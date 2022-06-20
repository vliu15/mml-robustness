"""
Contains extra utilities for implementing CLIP ResNet. Adapted from
https://github.com/openai/CLIP/blob/main/clip/clip.py
"""

import hashlib
import os
import urllib
import warnings

import torch
from tqdm import tqdm


def download_clip(url: str):
    root = os.path.expanduser("~/.cache/clip")

    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit="iB", unit_scale=True,
                  unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def load_resnet(model_path: str):
    with open(model_path, "rb") as opened_file:
        try:
            # loading JIT archive
            model = torch.jit.load(opened_file, map_location="cpu").eval()
            state_dict = None
        except RuntimeError:
            state_dict = torch.load(opened_file, map_location="cpu")

    state_dict = state_dict or model.state_dict()

    # Since state_dict contains keys for both the vision and language models, we want to revise the keys
    revised_state_dict = {}
    embed_dim = None
    for name, tensor in state_dict.items():
        # Keep all weights corresponding to the vision models
        if name.startswith("visual."):
            revised_state_dict[name[7:]] = tensor
        # Determine embed_dim from the text_projection weight
        elif name == "text_projection":
            embed_dim = tensor.shape[1]

    return revised_state_dict, embed_dim
