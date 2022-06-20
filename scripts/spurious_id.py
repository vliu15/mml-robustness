"""
Runs SVD on the entire matrix of spurious deltas created by
create_spurious_matrix.py for all tasks.

python -m scripts.spurious_id \
    --json_dir outputs/spurious_eval \
    --out_dir outputs/svd \
    --gamma 0.9 \
    --k 8
"""

import argparse
import csv
import json
import logging
import logging.config
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.special import softmax
from sklearn.cluster import KMeans, SpectralCoclustering
from tqdm import tqdm

from scripts.const import ATTRIBUTES

logging.config.fileConfig("logger.conf")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    # I/O dirs
    parser.add_argument(
        "--json_dir",
        type=str,
        required=False,
        default="outputs/spurious_eval",
        help="Path to json directory ingested by this script"
    )
    parser.add_argument(
        "--out_dir", type=str, required=False, default="outputs/svd", help="Path to output directory dumped by this script"
    )

    # Threshold/params
    parser.add_argument(
        "--gamma",
        type=float,
        required=False,
        default=0.9,
        help="Percent variance threshold: $$d = \min_j \max{(\sum_i^j \sigma_i^2) / (\sum_i^n \sigma_i^2) > \gamma}$$"
    )
    parser.add_argument("--k", type=int, required=False, default=16, help="Number of KMeans clusters")
    parser.add_argument("--eps", type=float, required=False, default=33, help="Spurious correlation metric threshold")
    args = parser.parse_args()

    # Some additional checks
    assert 0 <= args.gamma < 1, "$$\gamma$$ should be between [0, 1)"
    return args


def dump_all_spurious_correlations(save_name: str, T: np.ndarray, eps: float):
    # T should be indexed as T[attr, task]
    spurious_correlation_indices = np.argwhere(T > eps)
    spurious_correlations = []
    for attr_idx, task_idx in spurious_correlation_indices:
        attr = ATTRIBUTES[attr_idx]
        task = ATTRIBUTES[task_idx]
        spurious_correlations.append([f"{task}:{attr}", T[attr_idx, task_idx].item()])

    spurious_correlations.sort(reverse=True, key=lambda x: x[1])  # sort by delta
    with open(save_name, "w", newline="") as f:
        spurious_correlations = [["spurious_correlation", "delta"]] + spurious_correlations
        writer = csv.writer(f)
        writer.writerows(spurious_correlations)


def cosine_similarity(A: np.ndarray, B: np.ndarray):
    assert A.shape == B.shape
    dot = (A @ B.T)
    norm_A = np.linalg.norm(A, axis=-1, keepdims=True)
    norm_B = np.linalg.norm(B.T, axis=0, keepdims=True)
    return dot / (norm_A * norm_B)


def column_normalize(X: np.ndarray, how="norm"):
    eps = 1e-7
    norm = np.linalg.norm(X, axis=-1, keepdims=True)
    mask = (norm > eps).astype(np.float32)

    # Normalize to be unit vectors (with the exception of 0-vectors)
    if how == "norm":
        X = (X / np.clip(norm, a_min=eps, a_max=None)) * mask
        X = np.clip(X, a_min=eps, a_max=None)

    # Softmax-normalize
    elif how == "softmax":
        X = softmax(X, axis=-1) * mask + eps * (1. - mask)

    # Scale by 1/100 since these are percentages
    elif how == "scale":
        X = X / 100.
        X = np.clip(X, a_min=eps, a_max=None)

    # Normalize to standard Gaussian
    elif how == "gaussian":
        mean = np.mean(X, axis=-1, keepdims=True)
        std = np.std(X, axis=-1, keepdims=True)

        # Replace 0 std values with large values
        mask = (std > 0).astype(np.float32)
        std = std + (1. / eps) * (1 - mask)

        X = (X - mean) / std + eps * (1 - mask)

    return X


def decompose(X: np.ndarray, gamma: float):
    """X should be an n x n matrix, where we care about extracting correlations in the 0th dimension"""
    # Perform SVD. NOTE that X is not guaranteed to be non-singular
    u, s, _ = np.linalg.svd(X)

    # Compute gammas
    s2 = np.power(s, 2)
    g = s2 / s2.sum()

    # Compute the reduced feature dimension
    d = np.argmax(np.cumsum(g) > gamma)
    logger.info("Reduced feature dimension with gamma=%s: %s", gamma, d)

    d_for_logs = list(map(lambda x: round(x, 3), np.cumsum(g)))
    logger.info("Cumulative percent variance of singular values: %s", d_for_logs)

    # Return the relevant slices
    return u[:, :d]


def cluster(save_name: str, X: np.ndarray, k: int):
    """X should be an n x d matrix and is clustered with KMeans"""
    # Run KMeans
    kmeans = KMeans(n_clusters=k, tol=1e-6)
    kmeans.fit(X)

    # Aggregate clusters
    clusters = defaultdict(list)
    for label, attr in zip(kmeans.labels_, ATTRIBUTES):
        clusters[int(label.item())].append(attr)

    with open(save_name, "w") as f:
        json.dump(clusters, f)


def make_deltas_histogram(save_name: str, X: np.ndarray):
    """X can be an arbitrary matrix whose values are between [0, 100]"""
    plt.clf()
    plt.cla()

    _, ax = plt.subplots(figsize=(15, 6))
    histogram = sns.histplot(X, binwidth=1.0, binrange=(0, 101), ax=ax, multiple="stack")
    ax.set_title("Distribution of raw $\delta$ values, bucketed by percent")

    # Revise legend
    legend = ax.get_legend()
    handles = legend.legendHandles
    legend.remove()
    ax.legend(handles, ATTRIBUTES, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=2)

    # Revise axes scales and ticks
    ax.xaxis.set_ticks(np.arange(0, 101, 10))
    plt.yscale("log")

    plt.tight_layout()
    histogram.figure.savefig(save_name)
    logger.info(f"Saved deltas histogram to {save_name}")


def make_deltas_heatmap(save_name: str, X: np.ndarray, xlabel: str = None, ylabel: str = None):
    """X should be an n x n matrix whose values are between [0, 100]"""
    plt.clf()
    plt.cla()

    fig, ax = plt.subplots(figsize=(13, 13))
    heatmap = sns.heatmap(
        X / 100.,
        annot=False,
        linewidths=0.5,
        ax=ax,
        vmin=0,
        vmax=1,
        square=True,
        xticklabels=ATTRIBUTES,
        yticklabels=ATTRIBUTES,
        cmap=plt.cm.Greens
    )
    fig.suptitle(f"Raw $\delta$ values", fontsize=18)
    if xlabel:
        plt.xlabel(xlabel, labelpad=20, fontweight="bold")
    if ylabel:
        plt.ylabel(ylabel, labelpad=20, fontweight="bold")
    plt.tight_layout()

    heatmap.figure.savefig(save_name)
    logger.info(f"Saved deltas heatmap to {save_name}")


def make_bicluster_heatmap(save_name: str, X: np.ndarray, k: int, xlabel: str = None, ylabel: str = None):
    """X should be an n x n matrix"""
    plt.clf()
    plt.cla()

    # Init biclustering algorithm
    clustering = SpectralCoclustering(
        n_clusters=k,
        svd_method="arpack",  # slower but more accurate
        n_init=100,
    )

    # Fit and format
    clustering.fit(X)
    rows = np.argsort(clustering.row_labels_)
    cols = np.argsort(clustering.column_labels_)

    C = X[rows][:, cols]
    xticklabels = [ATTRIBUTES[c] for c in cols]
    yticklabels = [ATTRIBUTES[r] for r in rows]

    fig, ax = plt.subplots(figsize=(13, 13))
    heatmap = sns.heatmap(
        C,
        annot=False,
        linewidths=0.5,
        ax=ax,
        vmin=0,
        vmax=1,
        square=True,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        cmap=plt.cm.Blues
    )
    fig.suptitle(f"Biclustering", fontsize=18)
    ax.set_title(f"Method: {clustering.__class__.__name__}", fontsize=14, pad=15)
    if xlabel:
        plt.xlabel(xlabel, labelpad=20, fontweight="bold")
    if ylabel:
        plt.ylabel(ylabel, labelpad=20, fontweight="bold")
    plt.tight_layout()

    heatmap.figure.savefig(save_name)
    logger.info(f"Saved biclustering heatmap to {save_name}")


def make_cossim_heatmap(save_name: str, X: np.ndarray, xlabel: str = None, ylabel: str = None):
    """A and T should both be n x d matrices"""
    plt.clf()
    plt.cla()

    X = cosine_similarity(X, X)

    fig, ax = plt.subplots(figsize=(30, 30))
    heatmap = sns.heatmap(
        X,
        annot=False,
        linewidths=1.0,
        ax=ax,
        vmin=-1,
        vmax=1,
        square=True,
        xticklabels=ATTRIBUTES,
        yticklabels=ATTRIBUTES,
        cmap="mako"
    )
    fig.suptitle(f"Pairwise Cosine Similarity", fontsize=18)
    if xlabel:
        plt.xlabel(xlabel, labelpad=20, fontweight="bold")
    if ylabel:
        plt.ylabel(ylabel, labelpad=20, fontweight="bold")
    plt.tight_layout()

    heatmap.figure.savefig(save_name)
    logger.info(f"Saved cosine similarity spurious heatmap to {save_name}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load JSON files for each task
    T = []
    for task in tqdm(ATTRIBUTES, desc="Loading JSONs", total=len(ATTRIBUTES)):
        with open(os.path.join(args.json_dir, task, f"{task}_spurious_eval.json"), "r") as f:
            data = json.load(f)
            column = np.array([data[attr] for attr in ATTRIBUTES], dtype=np.float32)
            T.append(column)

    # Vectorize
    T = np.array(T).T  # shape (40, 40) that corresponds to (attr, task)

    dump_all_spurious_correlations(os.path.join(args.out_dir, "spurious_correlations.csv"), T, args.eps)

    # Histogram of deltas
    make_deltas_histogram(os.path.join(args.out_dir, "deltas_histogram.png"), T)
    make_deltas_heatmap(os.path.join(args.out_dir, "deltas_heatmap.png"), T, xlabel="Task Labels", ylabel="Attributes")

    # Binarize & Bicluster (Cocluster)
    bT = np.clip((T >= args.eps).astype(np.float32), a_min=1e-7, a_max=None)
    make_bicluster_heatmap(
        os.path.join(args.out_dir, "binary_bicluster_T.png"), bT, args.k, xlabel="Task Labels", ylabel="Attributes"
    )
    make_deltas_heatmap(
        os.path.join(args.out_dir, "binary_deltas_heatmap.png"),
        bT + 99 * (bT == 1),
        xlabel="Task Labels",
        ylabel="Attributes"
    )

    # Softmax-Normalize & Bicluster (Cocluster)
    sT = column_normalize(T, how="gaussian")
    sA = column_normalize(T.T, how="gaussian")
    make_bicluster_heatmap(
        os.path.join(args.out_dir, "bicluster_T.png"), sT, args.k, xlabel="Task Labels", ylabel="Attributes"
    )
    make_bicluster_heatmap(
        os.path.join(args.out_dir, "bicluster_TxA.png"),
        cosine_similarity(sT, sA),
        args.k,
        xlabel="Task Labels",
        ylabel="Attributes"
    )
    make_bicluster_heatmap(
        os.path.join(args.out_dir, "bicluster_TxT.png"),
        cosine_similarity(sT, sT),
        args.k,
        xlabel="Task Labels",
        ylabel="Task Labels"
    )

    # L2-Normalize & Decompose (SVD)
    nT = column_normalize(T, how="norm")
    nA = column_normalize(T.T, how="norm")
    uT = decompose(nT, args.gamma)
    uA = decompose(nA, args.gamma)

    # Cluster (KMeans)
    cluster(os.path.join(args.out_dir, "kmeans_T.json"), uT, args.k)

    # Cosine similarities
    make_cossim_heatmap(os.path.join(args.out_dir, "cosine_similarity_T.png"), uT, xlabel="Task Labels", ylabel="Task Labels")
    d = min(uT.shape[-1], uA.shape[-1])
    make_cossim_heatmap(
        os.path.join(args.out_dir, "cosine_similarity_TxA.png"),
        uT[:, :d] @ uA[:, :d].T,
        xlabel="Task Labels",
        ylabel="Attributes"
    )


if __name__ == "__main__":
    main()
