#!/usr/bin/env python3
import argparse
import io
import os
import sys
import zipfile
from typing import Dict, Tuple, Optional, List

import numpy as np


def load_embeddings_from_zip(zip_path: str) -> Tuple[np.ndarray, np.ndarray, Optional[List[Dict[str, str]]]]:
    """Load indices and features from a sam2_results.zip produced by the app.

    Returns (indices, features, catalog_rows) where:
      - indices: shape (N,), dtype int32
      - features: shape (N, D), dtype float32
      - catalog_rows: list of dicts parsed from dinov2_catalog.tsv or None if missing
    """
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"ZIP not found: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        if "dinov2_embeddings.npy" not in zf.namelist():
            raise FileNotFoundError("dinov2_embeddings.npy not found in ZIP")
        with zf.open("dinov2_embeddings.npy") as f:
            data = np.load(io.BytesIO(f.read()), allow_pickle=True).item()
        indices = np.asarray(data.get("indices"))
        features = np.asarray(data.get("features"))

        catalog_rows: Optional[List[Dict[str, str]]] = None
        if "dinov2_catalog.tsv" in zf.namelist():
            with zf.open("dinov2_catalog.tsv") as f:
                tsv_text = f.read().decode("utf-8", errors="replace")
            lines = [ln for ln in tsv_text.splitlines() if ln.strip()]
            if lines:
                header = lines[0].split("\t")
                catalog_rows = []
                for ln in lines[1:]:
                    parts = ln.split("\t")
                    row = {k: v for k, v in zip(header, parts)}
                    catalog_rows.append(row)

        return indices, features, catalog_rows


def load_embeddings_from_npy(npy_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load indices and features from a standalone .npy dict file saved by the app."""
    if not os.path.isfile(npy_path):
        raise FileNotFoundError(f"File not found: {npy_path}")
    data = np.load(npy_path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object:
        data = data.item()
    if not isinstance(data, dict):
        raise ValueError("Expected a dict with keys 'indices' and 'features' in npy file")
    indices = np.asarray(data.get("indices"))
    features = np.asarray(data.get("features"))
    return indices, features


def read_single_object_vector_from_zip(zip_path: str, object_index: int) -> np.ndarray:
    """Read a single per-object vector (object_{i}_dinov2.npy) from the ZIP if present."""
    filename = f"object_{object_index}_dinov2.npy"
    with zipfile.ZipFile(zip_path, "r") as zf:
        if filename not in zf.namelist():
            raise FileNotFoundError(f"{filename} not found in ZIP")
        with zf.open(filename) as f:
            vec = np.load(io.BytesIO(f.read()))
    return np.asarray(vec)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Read DINOv2 embeddings exported by the app.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--zip", dest="zip_path", type=str, help="Path to sam2_results.zip")
    group.add_argument("--npy", dest="npy_path", type=str, help="Path to dinov2_embeddings.npy (dict)")
    parser.add_argument("--show-catalog", action="store_true", help="Print catalog TSV rows if present (ZIP only)")
    parser.add_argument("--object-idx", type=int, default=None, help="Read and print per-object vector from ZIP")
    parser.add_argument("--export-prefix", type=str, default=None, help="If set, saves <prefix>_indices.npy and <prefix>_features.npy")
    args = parser.parse_args(argv)

    try:
        if args.zip_path:
            indices, features, catalog_rows = load_embeddings_from_zip(args.zip_path)
            print(f"Loaded from ZIP: indices shape={indices.shape}, dtype={indices.dtype}; features shape={features.shape}, dtype={features.dtype}")
            if args.show_catalog and catalog_rows is not None:
                print("Catalog rows (first 10):")
                for row in catalog_rows[:10]:
                    print(row)
            if args.object_idx is not None:
                vec = read_single_object_vector_from_zip(args.zip_path, args.object_idx)
                print(f"object_{args.object_idx}_dinov2.npy shape={vec.shape}, dtype={vec.dtype}")
        else:
            indices, features = load_embeddings_from_npy(args.npy_path)
            print(f"Loaded from NPY: indices shape={indices.shape}, dtype={indices.dtype}; features shape={features.shape}, dtype={features.dtype}")

        if args.export_prefix:
            idx_path = f"{args.export_prefix}_indices.npy"
            feats_path = f"{args.export_prefix}_features.npy"
            np.save(idx_path, indices)
            np.save(feats_path, features)
            print(f"Saved {idx_path} and {feats_path}")

    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


