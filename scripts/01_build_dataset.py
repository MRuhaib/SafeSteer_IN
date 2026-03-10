#!/usr/bin/env python
"""
SafeSteer-IN  ·  Step 1: Build Dataset
========================================
Constructs the contrastive-pair dataset from seed templates and
(optionally) HuggingFace safety benchmarks.

Usage:
    python scripts/01_build_dataset.py
    python scripts/01_build_dataset.py --no-hf          # skip HF datasets
    python scripts/01_build_dataset.py --no-augment      # skip augmentation
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.build_dataset import build_dataset


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Step 1: Build SafeSteer-IN dataset")
    parser.add_argument(
        "--no-augment", action="store_true", help="Disable paraphrase augmentation"
    )
    parser.add_argument(
        "--no-hf", action="store_true", help="Skip loading HuggingFace datasets"
    )
    parser.add_argument(
        "--augment-n",
        type=int,
        default=2,
        help="Number of augmented variants per seed pair",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    out = Path(args.output_dir) if args.output_dir else None

    print("=" * 60)
    print("  SafeSteer-IN  ·  Step 1: Dataset Construction")
    print("=" * 60)

    splits = build_dataset(
        augment=not args.no_augment,
        augment_n=args.augment_n,
        load_hf=not args.no_hf,
        seed=args.seed,
        output_dir=out,
    )

    total = sum(len(v) for v in splits.values())
    print(
        f"\n[OK] Dataset built successfully -- {total} total pairs across {len(splits)} splits."
    )
    print(f"  Files written to: {out or ROOT / 'data' / 'datasets'}")


if __name__ == "__main__":
    main()
