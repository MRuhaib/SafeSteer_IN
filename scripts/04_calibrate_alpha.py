#!/usr/bin/env python
"""
SafeSteer-IN  ·  Step 4: Calibrate Alpha
==========================================
Sweeps the steering strength (alpha) on the validation set to find
the optimal value per (language × category) slice.

Usage:
    python scripts/04_calibrate_alpha.py
    python scripts/04_calibrate_alpha.py --alphas 5 10 15 20 25 --max-pairs 20
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from steering.calibrate_alpha import run_full_calibration
from config import ALPHA_SEARCH_RANGE


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

    print(f"\n✓ Calibration complete — {len(results)} slices calibrated.")
    for key, info in results.items():
        best = info.get("best", "N/A")
        print(f"  {key}: best alpha = {best}")


if __name__ == "__main__":
    main()
