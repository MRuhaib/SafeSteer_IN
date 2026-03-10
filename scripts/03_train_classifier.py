#!/usr/bin/env python
"""
SafeSteer-IN  ·  Step 3: Train IndicBERT Risk Classifier
==========================================================
Fine-tunes IndicBERT for joint language + harm-category classification.

Usage:
    python scripts/03_train_classifier.py
    python scripts/03_train_classifier.py --epochs 15 --batch-size 32
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from classifier.train_classifier import train_classifier
from data.build_dataset import load_dataset_split


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Step 3: Train risk classifier")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--device", type=str, default=None, help="Force device (cpu / cuda)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  SafeSteer-IN  ·  Step 3: Train IndicBERT Classifier")
    print("=" * 60)
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR:         {args.lr}")
    print()

    train_pairs = load_dataset_split("train")
    val_pairs = load_dataset_split("val")

    print(f"  Train pairs: {len(train_pairs)}")
    print(f"  Val pairs:   {len(val_pairs)}")
    print()

    model = train_classifier(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )

    print(f"\n✓ Classifier training complete.")
    print(f"  Checkpoint saved to: {ROOT / 'models' / 'indic_risk_classifier'}")


if __name__ == "__main__":
    main()
