#!/usr/bin/env python
"""
SafeSteer-IN  ·  Step 2: Extract Steering Vectors
===================================================
Loads the target LLM and extracts contrastive steering vectors for each
(language × harm-category) slice of the training set.

Usage:
    python src/scripts/02_extract_vectors.py
    python src/scripts/02_extract_vectors.py --model openhathi-base --languages hi hi-en
    python src/scripts/02_extract_vectors.py --no-probe --languages hi ta
"""

import os
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_ROOT.parent
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from steering.extract_vectors import run_full_extraction
from config import VECTORS_DIR


def _try_configure_mlflow() -> bool:
    try:
        from mlops.mlflow_tracking import configure_mlflow
    except Exception:
        return False

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", "SafeSteer-IN")
    try:
        configure_mlflow(tracking_uri=tracking_uri, experiment_name=experiment)
    except Exception:
        return False
    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Step 2: Extract steering vectors")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model key: openhathi-base | airavata | sarvam-1 | sarvam-m | krutrim-2-instruct | phi-3-mini-4k-instruct | phi-4-mini-instruct",
    )
    parser.add_argument(
        "--languages", nargs="+", default=None, help="Language codes (default: all)"
    )
    parser.add_argument(
        "--categories", nargs="+", default=None, help="Category keys (default: all)"
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split to use for extraction"
    )
    parser.add_argument(
        "--no-probe",
        action="store_true",
        help="Skip the layer-probe study; use all candidate layers",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Extract only top-k probed layers instead of all configured target layers",
    )
    parser.add_argument(
        "--probe-k", type=int, default=3, help="Top-k layers to keep from probe"
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--quantize-4bit",
        action="store_true",
        help="Load model in 4-bit quantized mode (bitsandbytes)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  SafeSteer-IN  ·  Step 2: Steering Vector Extraction")
    print("=" * 60)
    print(f"  Model:      {args.model or 'default (from config)'}")
    print(f"  Languages:  {args.languages or 'all'}")
    print(f"  Categories: {args.categories or 'all'}")
    print(f"  Probe:      {'disabled' if args.no_probe else f'top-{args.probe_k}'}")
    print(f"  Probe only: {args.probe_only}")
    print(f"  4-bit:      {args.quantize_4bit}")
    print()

    log_mlflow = _try_configure_mlflow()

    target_layers = run_full_extraction(
        model_key=args.model,
        languages=args.languages,
        categories=args.categories,
        dataset_split=args.split,
        probe_first=not args.no_probe,
        probe_only=args.probe_only,
        probe_k=args.probe_k,
        max_length=args.max_length,
        quantize=True if args.quantize_4bit else None,
        log_mlflow=log_mlflow,
    )

    print(f"\n✓ Extraction complete.  Target layers used: {target_layers}")
    print(f"  Vectors stored in: {VECTORS_DIR}")


if __name__ == "__main__":
    main()
