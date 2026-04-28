#!/usr/bin/env python
"""
SafeSteer-IN  ·  Standalone Layer Sensitivity Analysis
======================================================
Evaluates steering quality across layers and alpha values for selected slices.
Outputs CSV with per-(language, category, layer, alpha) scores.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import List

SRC_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_ROOT.parent
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from config import ACTIVE_MODEL
from data.build_dataset import load_pairs_for_extraction
from evaluation.metrics import SafetyChecker
from steering.extract_vectors import load_model_and_tokenizer
from engine.steering_engine import SteeringEngine


def run_layer_sensitivity(
    model_key: str,
    languages: List[str],
    categories: List[str],
    alphas: List[float],
    split: str,
    max_prompts: int,
    max_tokens: int,
    use_azure: bool,
    output_csv: Path,
):
    model, tokenizer = load_model_and_tokenizer(model_key)
    engine = SteeringEngine(model, tokenizer, model_key)
    checker = SafetyChecker(use_azure=use_azure)

    rows = []

    for lang in languages:
        for cat in categories:
            layer_map = engine.vectors.get(lang, {}).get(cat, {})
            layers = sorted(layer_map.keys())
            if not layers:
                print(f"[WARN] no vectors for {lang}/{cat}")
                continue

            pairs = load_pairs_for_extraction(split, language=lang, category=cat)
            prompts = [p["prompt"] for p in pairs[:max_prompts]]
            if not prompts:
                print(f"[WARN] no prompts for {lang}/{cat}")
                continue

            for layer in layers:
                for alpha in alphas:
                    raw_safe = 0
                    steered_safe = 0
                    for prompt in prompts:
                        raw = engine.generate(prompt, max_new_tokens=max_tokens)
                        steered = engine.generate_steered(
                            prompt,
                            language=lang,
                            category=cat,
                            layer=layer,
                            alpha=alpha,
                            max_new_tokens=max_tokens,
                        )
                        if checker.is_safe(raw):
                            raw_safe += 1
                        if checker.is_safe(steered):
                            steered_safe += 1

                    total = len(prompts)
                    sir = (steered_safe - raw_safe) / total if total else 0.0
                    rows.append(
                        {
                            "model": model_key,
                            "language": lang,
                            "category": cat,
                            "layer": layer,
                            "alpha": alpha,
                            "num_prompts": total,
                            "raw_safe_count": raw_safe,
                            "steered_safe_count": steered_safe,
                            "sir": round(sir, 6),
                        }
                    )
                    print(f"{lang}/{cat} layer={layer} alpha={alpha}: SIR={sir:.3f}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    print(f"[OK] wrote {len(rows)} rows to {output_csv}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Standalone layer sensitivity run")
    parser.add_argument("--model", type=str, default=ACTIVE_MODEL)
    parser.add_argument("--languages", nargs="+", required=True)
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument(
        "--alphas", nargs="+", type=float, default=[5, 10, 15, 20, 25, 30, 35, 40]
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-prompts", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--azure", action="store_true")
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(
            PROJECT_ROOT / "src" / "evaluation" / "results" / "layer_sensitivity.csv"
        ),
    )
    args = parser.parse_args()

    run_layer_sensitivity(
        model_key=args.model,
        languages=args.languages,
        categories=args.categories,
        alphas=args.alphas,
        split=args.split,
        max_prompts=args.max_prompts,
        max_tokens=args.max_tokens,
        use_azure=args.azure,
        output_csv=Path(args.output_csv),
    )


if __name__ == "__main__":
    main()
