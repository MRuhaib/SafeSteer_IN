#!/usr/bin/env python
"""
SafeSteer-IN  ·  Step 4: Calibrate Alpha
==========================================
Sweeps the steering strength (alpha) on the validation set to find
the optimal value per (language × category) slice.

Usage:
    python src/scripts/04_calibrate_alpha.py
    python src/scripts/04_calibrate_alpha.py --alphas 5 10 15 20 25 --max-pairs 20
"""

import os
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_ROOT.parent
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from steering.calibrate_alpha import run_full_calibration
from config import ALPHA_SEARCH_RANGE


def _try_configure_mlflow():
    try:
        from mlops.mlflow_tracking import configure_mlflow, log_alpha_values
    except Exception:
        return None

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", "SafeSteer-IN")
    try:
        configure_mlflow(tracking_uri=tracking_uri, experiment_name=experiment)
    except Exception:
        return None
    return log_alpha_values


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Step 4: Calibrate alpha values")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=ALPHA_SEARCH_RANGE,
        help="Alpha values to sweep",
    )
    parser.add_argument(
        "--max-pairs", type=int, default=30, help="Max validation pairs per slice"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  SafeSteer-IN  ·  Step 4: Alpha Calibration")
    print("=" * 60)
    print(f"  Alpha range: {args.alphas}")
    print(f"  Max pairs:   {args.max_pairs}")
    print()

    results = run_full_calibration(
        model_key=args.model,
        languages=args.languages,
        categories=args.categories,
        alphas=args.alphas,
        max_pairs=args.max_pairs,
    )

    log_fn = _try_configure_mlflow()
    if log_fn:
        for key, info in results.items():
            try:
                lang, cat = key.split("/", 1)
            except ValueError:
                continue
            best_alpha = info.get("best")
            if best_alpha is None:
                continue
            metrics = info.get("alphas", {})
            best_metrics = metrics.get(str(best_alpha), {})
            try:
                log_fn(
                    language=lang,
                    category=cat,
                    best_alpha=best_alpha,
                    metrics_dict=best_metrics,
                )
            except Exception:
                pass

    print(f"\n✓ Calibration complete — {len(results)} slices calibrated.")
    for key, info in results.items():
        best = info.get("best", "N/A")
        print(f"  {key}: best alpha = {best}")


if __name__ == "__main__":
    main()
