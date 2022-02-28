"""
Runs SVD on the entire matrix of spurious deltas created by
create_spurious_matrix.py for all tasks.

python -m scripts.spurious_svd \
    --json_dir outputs/spurious_eval \
    --out_dir outputs/svd \
    --gamma 0.9 \
    --k 10
"""

import argparse
import json
import logging
import os
from collections import defaultdict
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.special import softmax
from sklearn.cluster import KMeans, SpectralCoclustering
from tqdm import tqdm

from scripts.create_spurious_matrix import attributes

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
    args = parser.parse_args()

    # Some additional checks
    assert 0 <= args.gamma < 1, "$$\gamma$$ should be between [0, 1)"
    return args


def cosine_similarity(A: np.ndarray, B: np.ndarray):
    assert A.shape == B.shape
    dot = (A @ B.T)
    norm_A = np.linalg.norm(A, axis=-1, keepdims=True)
    norm_B = np.linalg.norm(B.T, axis=0, keepdims=True)
    return dot / (norm_A * norm_B)


def column_normalize(X: np.ndarray, how="norm", eps=1e-7):
    norm = np.linalg.norm(X, axis=-1, keepdims=True)
    mask = (norm > eps).astype(np.float32)

    if how == "norm":
        X = (X / np.clip(norm, a_min=eps, a_max=None)) * mask + eps * (1. - mask)

    elif how == "softmax":
        X = softmax(X, axis=-1) * mask + eps * (1. - mask)

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
    d = 14

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
    for label, attr in zip(kmeans.labels_, attributes):
        clusters[int(label.item())].append(attr)

    with open(save_name, "w") as f:
        json.dump(clusters, f)


def bicluster_heatmap(save_name: str, X: np.ndarray, k: int, xlabel: str = None, ylabel: str = None):
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
    xticklabels = [attributes[r] for r in rows]
    yticklabels = [attributes[c] for c in cols]

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


def cosine_similarity_heatmap(save_name: str, X: np.ndarray, xlabel: str = None, ylabel: str = None):
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
        xticklabels=attributes,
        yticklabels=attributes,
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
    for task in tqdm(attributes, desc="Loading JSONs", total=len(attributes)):
        with open(os.path.join(args.json_dir, task, f"{task}_spurious_eval.json"), "r") as f:
            data = json.load(f)
            column = np.array([data[attr] for attr in attributes], dtype=np.float32)
            T.append(column)

    # Vectorize
    T = np.array(T)  # shape (40, 40) that corresponds to (task, attr)
    A = deepcopy(T.T)  # shape (40, 40) that corresponds to (attr, task)

    # Normalize and bicluster (Cocluster)
    sT = column_normalize(T, how="softmax")
    sA = column_normalize(A, how="softmax")
    bicluster_heatmap(
        os.path.join(args.out_dir, "bicluster_T.png"), (sT @ sT.T)**0.5, args.k, xlabel="Task Labels", ylabel="Task Labels"
    )
    bicluster_heatmap(
        os.path.join(args.out_dir, "bicluster_A.png"), (sA @ sA.T)**0.5, args.k, xlabel="Attributes", ylabel="Attributes"
    )
    bicluster_heatmap(
        os.path.join(args.out_dir, "bicluster_TxA.png"), (sT @ sA.T)**0.5, args.k, xlabel="Task Labels", ylabel="Attributes"
    )

    # Decompose (SVD)
    uT = decompose(column_normalize(T, how="norm"), args.gamma)
    uA = decompose(column_normalize(A, how="norm"), args.gamma)

    # Cluster (KMeans)
    cluster(os.path.join(args.out_dir, "kmeans_T.json"), uT, args.k)
    cluster(os.path.join(args.out_dir, "kmeans_A.json"), uA, args.k)

    # Cosine similarities
    cosine_similarity_heatmap(
        os.path.join(args.out_dir, "cosine_similarity_T.png"), uT, xlabel="Task Labels", ylabel="Task Labels"
    )
    cosine_similarity_heatmap(
        os.path.join(args.out_dir, "cosine_similarity_A.png"), uA, xlabel="Attributes", ylabel="Attributes"
    )
    cosine_similarity_heatmap(
        os.path.join(args.out_dir, "cosine_similarity_TxA.png"), uT @ uA.T, xlabel="Task Labels", ylabel="Attributes"
    )


if __name__ == "__main__":
    main()
