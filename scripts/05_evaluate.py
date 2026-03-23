#!/usr/bin/env python
"""
SafeSteer-IN  ·  Step 5: Evaluation
=====================================
Run the full evaluation pipeline on the test split.
Generates baseline and steered outputs, computes metrics, and saves
results to CSV/JSON.

Usage:
    python scripts/05_evaluate.py
    python scripts/05_evaluate.py --azure --max-prompts 30
    python scripts/05_evaluate.py --test-dir data/datasets/test --steered-only
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.evaluate import run_evaluation


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Step 5: Run evaluation")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument(
        "--azure", action="store_true", help="Use Azure Content Safety for scoring"
    )
    parser.add_argument(
        "--max-prompts", type=int, default=50, help="Max prompts per evaluation slice"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=128, help="Max new tokens per generation"
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=None,
        help="Directory with external CSV/XLSX prompt files",
    )
    parser.add_argument(
        "--steered-only",
        action="store_true",
        help="Only export steered responses + latency (no baseline metrics)",
    )
    parser.add_argument(
        "--alpha-sweep",
        nargs="+",
        type=float,
        default=None,
        help="Sweep these alpha values during steered export",
    )
    parser.add_argument(
        "--multi-layer",
        action="store_true",
        help="Apply steering at multiple layers for each slice",
    )
    parser.add_argument(
        "--multi-layer-top-k",
        type=int,
        default=3,
        help="Top-k layers to use when --multi-layer is enabled",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  SafeSteer-IN  ·  Step 5: Evaluation")
    print("=" * 60)
    print(f"  Azure scoring: {'enabled' if args.azure else 'disabled'}")
    print(f"  Max prompts:   {args.max_prompts}")
    if args.test_dir:
        print(f"  External test dir: {args.test_dir}")
    print(f"  Steered-only mode: {'enabled' if args.steered_only else 'disabled'}")
    if args.alpha_sweep:
        print(f"  Alpha sweep: {args.alpha_sweep}")
    print(f"  Multi-layer steering: {'enabled' if args.multi_layer else 'disabled'}")
    print(f"  Multi-layer top-k: {args.multi_layer_top_k}")
    print()

    results = run_evaluation(
        model_key=args.model,
        languages=args.languages,
        categories=args.categories,
        use_azure=args.azure,
        max_prompts=args.max_prompts,
        max_new_tokens=args.max_tokens,
        test_dir=Path(args.test_dir) if args.test_dir else None,
        steered_only=args.steered_only,
        alpha_sweep=args.alpha_sweep,
        use_multilayer=args.multi_layer,
        multilayer_top_k=args.multi_layer_top_k,
    )

    print(f"\n✓ Evaluation complete — {len(results)} slices evaluated.")
    print(f"  Results at: {ROOT / 'evaluation' / 'results'}")


if __name__ == "__main__":
    main()
