#!/usr/bin/env python3
import argparse
import io
import os
import zipfile
from typing import Optional, Tuple, Dict, List

import numpy as np


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


def load_from_npy(npy_path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.isfile(npy_path):
        raise FileNotFoundError(f"File not found: {npy_path}")
    data = np.load(npy_path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object:
        data = data.item()
    if not isinstance(data, dict):
        raise ValueError("Expected dict with keys 'indices' and 'features'")
    return np.asarray(data.get("indices")), np.asarray(data.get("features"))


def load_from_zip(zip_path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"ZIP not found: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        if "dinov2_embeddings.npy" not in zf.namelist():
            raise FileNotFoundError("dinov2_embeddings.npy not found in ZIP")
        with zf.open("dinov2_embeddings.npy") as f:
            data = np.load(io.BytesIO(f.read()), allow_pickle=True).item()
    return np.asarray(data.get("indices")), np.asarray(data.get("features"))


def load_catalog_from_dir(results_dir: str) -> Optional[Dict[int, Dict[str, str]]]:
    tsv_path = os.path.join(results_dir, "dinov2_catalog.tsv")
    if not os.path.isfile(tsv_path):
        return None
    rows: Dict[int, Dict[str, str]] = {}
    with open(tsv_path, "r", encoding="utf-8", errors="replace") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    if not lines:
        return None
    header = lines[0].split("\t")
    for ln in lines[1:]:
        parts = ln.split("\t")
        row = {k: v for k, v in zip(header, parts)}
        try:
            idx = int(row.get("index", "-1"))
        except ValueError:
            continue
        rows[idx] = row
    return rows


def load_catalog_from_zip(zip_path: str) -> Optional[Dict[int, Dict[str, str]]]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        if "dinov2_catalog.tsv" not in zf.namelist():
            return None
        with zf.open("dinov2_catalog.tsv") as f:
            tsv_text = f.read().decode("utf-8", errors="replace")
    lines = [ln for ln in tsv_text.splitlines() if ln.strip()]
    if not lines:
        return None
    header = lines[0].split("\t")
    rows: Dict[int, Dict[str, str]] = {}
    for ln in lines[1:]:
        parts = ln.split("\t")
        row = {k: v for k, v in zip(header, parts)}
        try:
            idx = int(row.get("index", "-1"))
        except ValueError:
            continue
        rows[idx] = row
    return rows


def cosine_similarity_matrix(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    x_norm = x / norms
    return x_norm @ x_norm.T


def find_closest_pair(indices: np.ndarray, features: np.ndarray) -> Tuple[int, int, float]:
    if features.shape[0] < 2:
        raise ValueError("Need at least two embeddings to compare")
    sim = cosine_similarity_matrix(features)
    np.fill_diagonal(sim, -np.inf)
    # Consider only upper triangle to avoid duplicates
    triu_mask = np.triu(np.ones_like(sim, dtype=bool), k=1)
    flat_idx = np.argmax(sim[triu_mask])
    pair_positions = np.vstack(np.where(triu_mask)).T
    i, j = pair_positions[flat_idx]
    idx_i, idx_j = int(indices[i]), int(indices[j])
    return idx_i, idx_j, float(sim[i, j])


def find_top_k_pairs(indices: np.ndarray, features: np.ndarray, k: int) -> List[Tuple[int, int, float]]:
    if features.shape[0] < 2:
        raise ValueError("Need at least two embeddings to compare")
    sim = cosine_similarity_matrix(features)
    np.fill_diagonal(sim, -np.inf)
    triu_mask = np.triu(np.ones_like(sim, dtype=bool), k=1)
    # Extract upper-triangular similarities into a 1D array
    triu_sims = sim[triu_mask]
    # Get indices of top-k similarities
    k = max(1, min(k, triu_sims.size))
    topk_flat = np.argpartition(-triu_sims, kth=k-1)[:k]
    # Map back to (i, j)
    pair_positions = np.vstack(np.where(triu_mask)).T
    pairs = []
    for idx in topk_flat:
        i, j = pair_positions[idx]
        pairs.append((int(indices[i]), int(indices[j]), float(sim[i, j])))
    # Sort by similarity desc
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Find closest objects by cosine similarity.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", type=str, help="Directory containing dinov2_embeddings.npy")
    group.add_argument("--npy", type=str, help="Path to dinov2_embeddings.npy (dict with indices/features)")
    group.add_argument("--zip", type=str, help="Path to sam2_results.zip containing dinov2_embeddings.npy")
    parser.add_argument("--top-k", type=int, default=1, help="Return top-K pairs (default: 1)")
    args = parser.parse_args(argv)

    names: Dict[int, str] = {}
    variants: Dict[int, str] = {}

    if args.dir:
        indices, features = load_from_dir(args.dir)
        # Build friendly names from files and variants from catalog
        catalog = load_catalog_from_dir(args.dir)
        for idx in indices.tolist():
            obj_png = os.path.join(args.dir, f"object_{idx}.png")
            names[idx] = os.path.basename(obj_png) if os.path.isfile(obj_png) else f"object_{idx}"
            if catalog is not None and idx in catalog and "variant" in catalog[idx]:
                variants[idx] = catalog[idx]["variant"]
    elif args.npy:
        indices, features = load_from_npy(args.npy)
        for idx in indices.tolist():
            names[idx] = f"object_{idx}"
    else:
        indices, features = load_from_zip(args.zip)
        with zipfile.ZipFile(args.zip, "r") as zf:
            namelist = set(zf.namelist())
        catalog = load_catalog_from_zip(args.zip)
        for idx in indices.tolist():
            fname = f"object_{idx}.png"
            names[idx] = fname if fname in namelist else f"object_{idx}"
            if catalog is not None and idx in catalog and "variant" in catalog[idx]:
                variants[idx] = catalog[idx]["variant"]

    if args.top_k <= 1:
        idx_a, idx_b, sim = find_closest_pair(indices, features)
        dist = 1.0 - sim
        name_a = names.get(idx_a, f"object_{idx_a}")
        name_b = names.get(idx_b, f"object_{idx_b}")
        var_a = variants.get(idx_a)
        var_b = variants.get(idx_b)
        if var_a and var_b and var_a == var_b:
            print(f"Closest pair: ({idx_a}:{name_a}, {idx_b}:{name_b})  variant={var_a}  cos_sim={sim:.6f}  cos_dist={dist:.6f}")
        else:
            extra = ""
            if var_a:
                extra += f" variant_a={var_a}"
            if var_b:
                extra += f" variant_b={var_b}"
            print(f"Closest pair: ({idx_a}:{name_a}, {idx_b}:{name_b}){extra}  cos_sim={sim:.6f}  cos_dist={dist:.6f}")
    else:
        pairs = find_top_k_pairs(indices, features, args.top_k)
        for rank, (ia, ib, sim) in enumerate(pairs, start=1):
            dist = 1.0 - sim
            name_a = names.get(ia, f"object_{ia}")
            name_b = names.get(ib, f"object_{ib}")
            var_a = variants.get(ia)
            var_b = variants.get(ib)
            if var_a and var_b and var_a == var_b:
                prefix = f"Top{rank}"
                print(f"{prefix}: ({ia}:{name_a}, {ib}:{name_b})  variant={var_a}  cos_sim={sim:.6f}  cos_dist={dist:.6f}")
            else:
                extra = ""
                if var_a:
                    extra += f" variant_a={var_a}"
                if var_b:
                    extra += f" variant_b={var_b}"
                prefix = f"Top{rank}"
                print(f"{prefix}: ({ia}:{name_a}, {ib}:{name_b}){extra}  cos_sim={sim:.6f}  cos_dist={dist:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


