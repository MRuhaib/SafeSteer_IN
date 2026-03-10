"""
SafeSteer-IN  ·  Evaluation Pipeline
======================================
Run the full evaluation suite:

    1. Load the test split of the contrastive-pair dataset.
    2. For each (lang, category) slice:
       a. Generate baseline (raw) outputs.
       b. Generate steered outputs.
       c. Compute SIR, FPR, fluency, latency.
       d. (Optional) Score via Azure Content Safety.
    3. Aggregate results into a CSV and JSON report.
    4. Log to Azure ML / MLflow if configured.

Outputs:
    evaluation/results/eval_results.csv
    evaluation/results/eval_results.json
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    ACTIVE_MODEL,
    EVALUATION_DIR,
    MAX_NEW_TOKENS,
    MODEL_CONFIGS,
    VECTORS_DIR,
)
from data.build_dataset import load_pairs_for_extraction
from data.taxonomy import CATEGORY_KEYS
from evaluation.metrics import SafetyMetrics, compute_metrics

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
)


# ─────────────────────────────────────────────────────────────────────────────
# Per-slice evaluation
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_slice(
    engine,  # SteeringEngine
    pairs: List[dict],
    language: str,
    category: str,
    model=None,
    tokenizer=None,
    use_azure: bool = False,
    max_prompts: int = 50,
    max_new_tokens: int = 128,
) -> Dict:
    """
    Evaluate one (language, category) slice.

    Returns a dict with raw_outputs, steered_outputs, metrics, and
    per-prompt comparison rows.
    """
    prompts = [p["prompt"] for p in pairs[:max_prompts]]

    raw_outputs: List[str] = []
    steered_outputs: List[str] = []
    latencies: List[float] = []

    for prompt in tqdm(prompts, desc=f"{language}/{category}", leave=False):
        # Baseline
        raw = engine.generate(prompt, max_new_tokens=max_new_tokens)
        raw_outputs.append(raw)

        # Steered + timing
        t0 = time.time()
        steered = engine.generate_steered(
            prompt,
            language,
            category,
            max_new_tokens=max_new_tokens,
        )
        latencies.append((time.time() - t0) * 1000)
        steered_outputs.append(steered)

    # Metrics
    metrics = compute_metrics(
        raw_outputs,
        steered_outputs,
        latencies_ms=latencies,
        model=model,
        tokenizer=tokenizer,
        use_azure=use_azure,
    )

    # Per-prompt comparison
    rows = []
    for i, prompt in enumerate(prompts):
        rows.append(
            {
                "prompt": prompt,
                "language": language,
                "category": category,
                "raw_output": raw_outputs[i],
                "steered_output": steered_outputs[i],
                "latency_ms": round(latencies[i], 2),
            }
        )

    return {
        "language": language,
        "category": category,
        "num_prompts": len(prompts),
        "metrics": metrics.to_dict(),
        "rows": rows,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation
# ─────────────────────────────────────────────────────────────────────────────


def run_evaluation(
    model_key: Optional[str] = None,
    languages: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    use_azure: bool = False,
    max_prompts: int = 50,
    max_new_tokens: int = 128,
    output_dir: Optional[Path] = None,
):
    """
    Run evaluation over all available (language × category) slices.
    """
    from config import LANGUAGES
    from steering.extract_vectors import load_model_and_tokenizer
    from engine.steering_engine import SteeringEngine

    _model_key = model_key or ACTIVE_MODEL
    _languages = languages or list(LANGUAGES.keys())
    _categories = categories or CATEGORY_KEYS
    out_dir = output_dir or EVALUATION_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model + engine
    model, tokenizer = load_model_and_tokenizer(_model_key)
    engine = SteeringEngine(model, tokenizer, _model_key)

    all_results: List[Dict] = []
    summary_rows: List[Dict] = []

    for lang in _languages:
        for cat in _categories:
            if not engine.has_vector(lang, cat):
                logger.info("No vector for %s/%s — skipping eval", lang, cat)
                continue

            pairs = load_pairs_for_extraction("test", language=lang, category=cat)
            if len(pairs) < 2:
                logger.info("Insufficient test pairs for %s/%s", lang, cat)
                continue

            logger.info(
                "━━━ Evaluating: %s / %s  (%d pairs) ━━━", lang, cat, len(pairs)
            )
            result = evaluate_slice(
                engine,
                pairs,
                lang,
                cat,
                model=model,
                tokenizer=tokenizer,
                use_azure=use_azure,
                max_prompts=max_prompts,
                max_new_tokens=max_new_tokens,
            )
            all_results.append(result)

            summary_rows.append(
                {
                    "language": lang,
                    "category": cat,
                    "model": _model_key,
                    **result["metrics"],
                }
            )

    # ── Save results ─────────────────────────────────────────────────────

    # JSON (full)
    json_path = out_dir / "eval_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info("Full results → %s", json_path)

    # CSV (summary)
    csv_path = out_dir / "eval_results.csv"
    if summary_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        logger.info("Summary CSV  → %s", csv_path)

    # ── Print summary table ──────────────────────────────────────────────
    _print_summary(summary_rows)

    # ── Log to Azure ML if available ─────────────────────────────────────
    try:
        from azure_ml.tracking import log_evaluation_results

        log_evaluation_results(summary_rows, _model_key)
    except Exception as e:
        logger.info("Azure ML logging skipped: %s", e)

    return all_results


def _print_summary(rows: List[Dict]):
    if not rows:
        print("No evaluation results to display.")
        return

    header = f"{'Language':<10} {'Category':<28} {'Model':<12} {'SIR':>6} {'FPR':>6} {'Fluency':>8} {'Latency':>10}"
    print("\n" + "=" * len(header))
    print("  SafeSteer-IN  ·  Evaluation Summary")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['language']:<10} {r['category']:<28} {r['model']:<12} "
            f"{r.get('SIR', 0):>6.3f} {r.get('FPR', 0):>6.3f} "
            f"{r.get('fluency_score', -1):>8.3f} {r.get('avg_latency_ms', 0):>8.1f} ms"
        )
    print("=" * len(header))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SafeSteer-IN evaluation")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--azure", action="store_true", help="Use Azure Content Safety")
    parser.add_argument("--max-prompts", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    run_evaluation(
        model_key=args.model,
        languages=args.languages,
        categories=args.categories,
        use_azure=args.azure,
        max_prompts=args.max_prompts,
        max_new_tokens=args.max_tokens,
    )
