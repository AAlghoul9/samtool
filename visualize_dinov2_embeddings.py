#!/usr/bin/env python3
import argparse
import os
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def load_from_dir(results_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    npy_path = os.path.join(results_dir, "dinov2_embeddings.npy")
    if not os.path.isfile(npy_path):
        raise FileNotFoundError(f"dinov2_embeddings.npy not found in: {results_dir}")
    data = np.load(npy_path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object:
        data = data.item()
    indices = np.asarray(data.get("indices"))
    features = np.asarray(data.get("features"))
    return indices, features


def cosine_similarity_matrix(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    x_norm = x / norms
    return x_norm @ x_norm.T


def pca_2d(x: np.ndarray) -> np.ndarray:
    # center
    x = x.astype(np.float64)
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean
    # SVD on centered data (thin)
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    components = vt[:2].T  # (D, 2)
    coords = xc @ components  # (N, 2)
    return coords.astype(np.float32)


def save_similarity_heatmap(sim: np.ndarray, indices: np.ndarray, out_path: str) -> None:
    plt.figure(figsize=(6, 5), dpi=160)
    im = plt.imshow(sim, cmap="viridis", vmin=-1.0, vmax=1.0)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="cosine similarity")
    plt.xticks(ticks=np.arange(len(indices)), labels=[str(i) for i in indices], rotation=90)
    plt.yticks(ticks=np.arange(len(indices)), labels=[str(i) for i in indices])
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_pca_scatter(coords: np.ndarray, indices: np.ndarray, out_path: str) -> None:
    plt.figure(figsize=(6, 5), dpi=160)
    plt.scatter(coords[:, 0], coords[:, 1], c=np.arange(coords.shape[0]), cmap="tab10")
    for i, idx in enumerate(indices):
        plt.text(coords[i, 0], coords[i, 1], str(idx), fontsize=8, ha="center", va="center")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("PCA of DINOv2 object embeddings")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Visualize DINOv2 embeddings: heatmap and PCA scatter.")
    parser.add_argument("--dir", required=True, help="Directory containing dinov2_embeddings.npy (e.g., sam2_results-2)")
    parser.add_argument("--out-dir", default=None, help="Directory to save visualizations; defaults to --dir")
    args = parser.parse_args(argv)

    results_dir = os.path.abspath(args.dir)
    out_dir = os.path.abspath(args.out_dir or results_dir)
    os.makedirs(out_dir, exist_ok=True)

    indices, features = load_from_dir(results_dir)
    print(f"Loaded: indices {indices.shape} dtype={indices.dtype}; features {features.shape} dtype={features.dtype}")

    sim = cosine_similarity_matrix(features)
    heatmap_path = os.path.join(out_dir, "dinov2_similarity_heatmap.png")
    save_similarity_heatmap(sim, indices, heatmap_path)
    print(f"Saved heatmap: {heatmap_path}")

    coords = pca_2d(features)
    scatter_path = os.path.join(out_dir, "dinov2_pca_scatter.png")
    save_pca_scatter(coords, indices, scatter_path)
    print(f"Saved PCA scatter: {scatter_path}")

    # Print top-3 neighbors per item
    np.fill_diagonal(sim, -np.inf)
    for row_idx, idx_val in enumerate(indices):
        nn_order = np.argsort(-sim[row_idx])[:3]
        nn_info = [(int(indices[j]), float(sim[row_idx, j])) for j in nn_order]
        print(f"index {int(idx_val)} → top-3 neighbors: {nn_info}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


