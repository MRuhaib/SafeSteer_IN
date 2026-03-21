#!/usr/bin/env python
"""
SafeSteer-IN  ·  Standalone Vector Similarity Analysis
======================================================
Computes cosine similarity and linear CKA matrices across languages for
steering vectors (category-wise, per layer) and saves CSV + heatmaps.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import ACTIVE_MODEL
from data.taxonomy import CATEGORY_KEYS
from steering.extract_vectors import load_all_vectors


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    # X, Y: [n_samples, hidden_dim]
    K = X @ X.T
    L = Y @ Y.T
    hsic = np.sum(K * L)
    norm = math.sqrt(np.sum(K * K) * np.sum(L * L))
    if norm == 0:
        return 0.0
    return float(hsic / norm)


def _save_matrix_csv(path: Path, labels: List[str], mat: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["language"] + labels)
        for i, l in enumerate(labels):
            writer.writerow([l] + [f"{mat[i, j]:.6f}" for j in range(len(labels))])


def _save_heatmap(path: Path, labels: List[str], mat: np.ndarray, title: str):
    import matplotlib.pyplot as plt
    import seaborn as sns

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        mat,
        annot=True,
        fmt=".2f",
        xticklabels=labels,
        yticklabels=labels,
        cmap="viridis",
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def run_similarity(
    model_key: str,
    languages: List[str],
    categories: List[str],
    layer: int | None,
    output_dir: Path,
):
    vectors = load_all_vectors(model_key)

    # Choose layer: either provided, or model-wise most common layer.
    if layer is None:
        freq: Dict[int, int] = {}
        for lang in vectors:
            for cat in vectors[lang]:
                for l in vectors[lang][cat]:
                    freq[l] = freq.get(l, 0) + 1
        if not freq:
            raise ValueError("No vectors found to infer layer.")
        layer = max(freq, key=freq.get)

    # Build one representative vector per language by averaging category vectors.
    per_lang_vec: Dict[str, np.ndarray] = {}
    per_lang_cat_matrix: Dict[str, np.ndarray] = {}

    for lang in languages:
        cat_vecs = []
        for cat in categories:
            vec = vectors.get(lang, {}).get(cat, {}).get(layer)
            if vec is None:
                continue
            arr = vec.detach().cpu().float().numpy()
            cat_vecs.append(arr)

        if not cat_vecs:
            continue

        mat = np.stack(cat_vecs, axis=0)  # [n_categories, d]
        per_lang_cat_matrix[lang] = mat
        per_lang_vec[lang] = mat.mean(axis=0)

    labels = sorted(per_lang_vec.keys())
    if len(labels) < 2:
        raise ValueError(
            "Need vectors from at least two languages for similarity analysis."
        )

    n = len(labels)
    cosine_mat = np.zeros((n, n), dtype=np.float32)
    cka_mat = np.zeros((n, n), dtype=np.float32)

    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            cosine_mat[i, j] = _cosine(per_lang_vec[li], per_lang_vec[lj])

            Xi = per_lang_cat_matrix[li]
            Yj = per_lang_cat_matrix[lj]
            m = min(len(Xi), len(Yj))
            cka_mat[i, j] = _linear_cka(Xi[:m], Yj[:m]) if m >= 2 else 0.0

    _save_matrix_csv(output_dir / f"cosine_layer{layer}.csv", labels, cosine_mat)
    _save_matrix_csv(output_dir / f"cka_layer{layer}.csv", labels, cka_mat)
    _save_heatmap(
        output_dir / f"cosine_layer{layer}.png",
        labels,
        cosine_mat,
        f"Cosine similarity (model={model_key}, layer={layer})",
    )
    _save_heatmap(
        output_dir / f"cka_layer{layer}.png",
        labels,
        cka_mat,
        f"Linear CKA (model={model_key}, layer={layer})",
    )

    print(f"[OK] Similarity analysis complete for model={model_key}, layer={layer}")
    print(f"[OK] Files written under: {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Standalone vector similarity analysis"
    )
    parser.add_argument("--model", type=str, default=ACTIVE_MODEL)
    parser.add_argument("--languages", nargs="+", required=True)
    parser.add_argument("--categories", nargs="+", default=CATEGORY_KEYS)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "evaluation" / "results" / "similarity"),
    )
    args = parser.parse_args()

    run_similarity(
        model_key=args.model,
        languages=args.languages,
        categories=args.categories,
        layer=args.layer,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
