#!/usr/bin/env python
"""
SafeSteer-IN  ·  Step 1: Build Dataset
========================================
Constructs the contrastive-pair dataset from seed templates and
(optionally) HuggingFace safety benchmarks.

Usage:
    python src/scripts/01_build_dataset.py
    python src/scripts/01_build_dataset.py --with-hf        # include HF datasets
    python src/scripts/01_build_dataset.py --no-augment      # skip augmentation
"""

import os
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_ROOT.parent
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from data.build_dataset import build_dataset
from config import DATASET_DIR


def _try_configure_mlflow():
    try:
        from mlops.mlflow_tracking import configure_mlflow, log_dataset_build
    except Exception:
        return None

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", "SafeSteer-IN")
    try:
        configure_mlflow(tracking_uri=tracking_uri, experiment_name=experiment)
    except Exception:
        return None
    return log_dataset_build


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Step 1: Build SafeSteer-IN dataset")
    parser.add_argument(
        "--no-augment", action="store_true", help="Disable paraphrase augmentation"
    )
    parser.add_argument(
        "--with-hf", action="store_true", help="Include HuggingFace datasets"
    )
    parser.add_argument(
        "--augment-n",
        type=int,
        default=2,
        help="Number of augmented variants per seed pair",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-pairs-per-slice",
        type=int,
        default=30,
        help="Minimum synthetic contrastive pairs per (language, category) slice",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    out = Path(args.output_dir) if args.output_dir else None

    print("=" * 60)
    print("  SafeSteer-IN  ·  Step 1: Dataset Construction")
    print("=" * 60)

    splits = build_dataset(
        augment=not args.no_augment,
        augment_n=args.augment_n,
        load_hf=args.with_hf,
        seed=args.seed,
        min_pairs_per_slice=args.min_pairs_per_slice,
        output_dir=out,
    )

    log_fn = _try_configure_mlflow()
    if log_fn:
        all_pairs = [p for split in splits.values() for p in split]
        languages = sorted({p.get("language") for p in all_pairs if p.get("language")})
        categories = sorted({p.get("category") for p in all_pairs if p.get("category")})
        try:
            log_fn(
                num_pairs=sum(len(v) for v in splits.values()),
                languages=languages,
                categories=categories,
                source_details={
                    "hf": args.with_hf,
                    "augment": not args.no_augment,
                    "augment_n": args.augment_n,
                    "min_pairs_per_slice": args.min_pairs_per_slice,
                },
            )
        except Exception:
            pass

    total = sum(len(v) for v in splits.values())
    print(
        f"\n[OK] Dataset built successfully -- {total} total pairs across {len(splits)} splits."
    )
    print(f"  Files written to: {out or DATASET_DIR}")


if __name__ == "__main__":
    main()
