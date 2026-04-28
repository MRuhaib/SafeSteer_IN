#!/usr/bin/env python
"""
SafeSteer-IN  ·  Step 3: Train IndicBERT Risk Classifier
==========================================================
Fine-tunes IndicBERT for joint language + harm-category classification.

Usage:
    python src/scripts/03_train_classifier.py
    python src/scripts/03_train_classifier.py --epochs 15 --batch-size 32
"""

import os
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_ROOT.parent
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from classifier.train_classifier import train_classifier
from data.build_dataset import load_dataset_split
from config import CLASSIFIER_CHECKPOINT


def _try_configure_mlflow():
    try:
        from mlops.mlflow_tracking import configure_mlflow, log_classifier_training
    except Exception:
        return None

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", "SafeSteer-IN")
    try:
        configure_mlflow(tracking_uri=tracking_uri, experiment_name=experiment)
    except Exception:
        return None
    return log_classifier_training


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

    log_fn = _try_configure_mlflow()

    model, metrics = train_classifier(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        return_metrics=True,
    )

    if log_fn and metrics:
        try:
            train_acc = (
                metrics.get("train_lang_acc", 0.0) + metrics.get("train_cat_acc", 0.0)
            ) / 2
            val_acc = (
                metrics.get("val_lang_acc", 0.0) + metrics.get("val_cat_acc", 0.0)
            ) / 2
            log_fn(
                epochs=args.epochs,
                final_train_acc=train_acc,
                final_val_acc=val_acc,
                final_train_loss=metrics.get("train_loss", 0.0),
                final_val_loss=None,
                num_labels=8,
            )
        except Exception:
            pass

    print(f"\n✓ Classifier training complete.")
    print(f"  Checkpoint saved to: {CLASSIFIER_CHECKPOINT}")


if __name__ == "__main__":
    main()
