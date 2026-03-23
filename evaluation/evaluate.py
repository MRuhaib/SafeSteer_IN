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
import pandas as pd

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
    test_dir: Optional[Path] = None,
    steered_only: bool = False,
    alpha_sweep: Optional[List[float]] = None,
    use_multilayer: bool = False,
    multilayer_top_k: int = 3,
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

    if test_dir is not None:
        test_path = Path(test_dir)
        if steered_only:
            return run_steered_export_from_files(
                engine=engine,
                model_key=_model_key,
                test_dir=test_path,
                output_dir=out_dir,
                max_prompts=max_prompts,
                max_new_tokens=max_new_tokens,
                languages=_languages,
                categories=_categories,
                alpha_values=alpha_sweep,
                use_multilayer=use_multilayer,
                multilayer_top_k=multilayer_top_k,
            )
        return run_full_eval_from_files(
            engine=engine,
            model=model,
            tokenizer=tokenizer,
            model_key=_model_key,
            test_dir=test_path,
            output_dir=out_dir,
            use_azure=use_azure,
            max_prompts=max_prompts,
            max_new_tokens=max_new_tokens,
            languages=_languages,
            categories=_categories,
        )

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


def _read_prompt_file(path: Path) -> List[dict]:
    """Read a CSV/XLSX prompt file and normalize records."""
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        return []

    # Handle sheets where first row contains real headers and pandas used Column1..ColumnN.
    cols_lower = [str(c).lower() for c in df.columns]
    if all(c.startswith("column") for c in cols_lower) and len(df) > 0:
        first = [str(x).strip() for x in df.iloc[0].tolist()]
        if {"prompt_id", "language", "category", "prompt_text"}.issubset(set(first)):
            df.columns = first
            df = df.iloc[1:].reset_index(drop=True)

    required = {"language", "category", "prompt_text"}
    if not required.issubset(set(df.columns)):
        return []

    items: List[dict] = []
    for i, row in df.iterrows():
        prompt = str(row.get("prompt_text", "")).strip()
        if not prompt:
            continue
        items.append(
            {
                "prompt_id": str(row.get("prompt_id", f"row_{i+1}")),
                "language": str(row.get("language", "")).strip(),
                "category": str(row.get("category", "")).strip(),
                "prompt_text": prompt,
            }
        )
    return items


def _load_prompt_files(
    test_dir: Path,
    languages: List[str],
    categories: List[str],
    max_prompts: int,
) -> Dict[str, List[dict]]:
    files = sorted(
        [p for p in test_dir.iterdir() if p.suffix.lower() in {".csv", ".xlsx", ".xls"}]
    )
    bucket: Dict[str, List[dict]] = {}
    for f in files:
        rows = _read_prompt_file(f)
        for r in rows:
            lang = r["language"]
            cat = r["category"]
            if languages and lang not in languages:
                continue
            if categories and cat not in categories:
                continue
            key = f"{lang}/{cat}"
            bucket.setdefault(key, []).append(r)

    # Cap prompts per slice.
    for k, v in list(bucket.items()):
        bucket[k] = v[:max_prompts]
    return bucket


def run_steered_export_from_files(
    engine,
    model_key: str,
    test_dir: Path,
    output_dir: Path,
    max_prompts: int,
    max_new_tokens: int,
    languages: List[str],
    categories: List[str],
    alpha_values: Optional[List[float]] = None,
    use_multilayer: bool = False,
    multilayer_top_k: int = 3,
):
    """Run steered generation over external prompt files with incremental per-file exports."""
    export_root = output_dir / "steered_exports" / model_key
    export_root.mkdir(parents=True, exist_ok=True)
    file_export_root = export_root / "by_test_file"
    file_export_root.mkdir(parents=True, exist_ok=True)

    alpha_values = alpha_values or [None]
    alpha_labels = [
        "default_or_calibrated" if a is None else str(a).replace(".", "p")
        for a in alpha_values
    ]

    test_files = sorted(
        [p for p in test_dir.iterdir() if p.suffix.lower() in {".csv", ".xlsx", ".xls"}]
    )
    logger.info(
        "Found %d external test files in %s | alpha sweep=%s | multilayer=%s top_k=%d",
        len(test_files),
        test_dir,
        alpha_values,
        use_multilayer,
        multilayer_top_k,
    )

    all_rows: List[dict] = []
    total_samples = 0
    total_steering_applied = 0

    for file_idx, file_path in enumerate(test_files, start=1):
        rows = _read_prompt_file(file_path)
        rows = [
            r
            for r in rows
            if (not languages or r["language"] in languages)
            and (not categories or r["category"] in categories)
        ]

        # Cap prompts per slice within this file.
        by_slice: Dict[str, List[dict]] = {}
        for r in rows:
            key = f"{r['language']}/{r['category']}"
            by_slice.setdefault(key, []).append(r)
        capped_rows: List[dict] = []
        for _, slice_rows in by_slice.items():
            capped_rows.extend(slice_rows[:max_prompts])

        if not capped_rows:
            logger.info(
                "[%d/%d] %s: no matching prompts after filters",
                file_idx,
                len(test_files),
                file_path.name,
            )
            continue

        logger.info(
            "[%d/%d] Processing %s (%d prompts)",
            file_idx,
            len(test_files),
            file_path.name,
            len(capped_rows),
        )

        file_rows: List[dict] = []
        file_steered = 0

        for alpha_idx, alpha in enumerate(alpha_values, start=1):
            alpha_label = alpha_labels[alpha_idx - 1]
            logger.info(
                "[%s] alpha %d/%d = %s",
                file_path.name,
                alpha_idx,
                len(alpha_values),
                alpha,
            )

            for sample_idx, r in enumerate(
                tqdm(
                    capped_rows,
                    desc=f"steered {file_path.name} alpha={alpha}",
                    leave=False,
                ),
                start=1,
            ):
                lang = r["language"]
                cat = r["category"]
                prompt = r["prompt_text"]

                t0 = time.time()
                steering_applied = False
                layers_used: List[int] = []

                if use_multilayer and engine.has_any_vector(lang, cat):
                    layers_used = engine.get_available_layers(lang, cat)
                    if multilayer_top_k > 0 and len(layers_used) > multilayer_top_k:
                        primary = engine.primary_layer
                        layers_used = sorted(
                            sorted(
                                layers_used,
                                key=lambda l: (abs(l - primary), l),
                            )[:multilayer_top_k]
                        )

                    out = engine.generate_steered_multilayer(
                        prompt,
                        lang,
                        cat,
                        alpha=alpha,
                        layers=layers_used,
                        top_k=multilayer_top_k,
                        max_new_tokens=max_new_tokens,
                    )
                    steering_applied = True
                elif engine.has_vector(lang, cat):
                    layers_used = [engine.primary_layer]
                    out = engine.generate_steered(
                        prompt,
                        lang,
                        cat,
                        alpha=alpha,
                        max_new_tokens=max_new_tokens,
                    )
                    steering_applied = True
                else:
                    out = engine.generate(prompt, max_new_tokens=max_new_tokens)

                if steering_applied:
                    file_steered += 1
                    total_steering_applied += 1

                latency = (time.time() - t0) * 1000
                row = {
                    "source_file": file_path.name,
                    "prompt_id": r["prompt_id"],
                    "language": lang,
                    "category": cat,
                    "alpha": alpha if alpha is not None else "default_or_calibrated",
                    "layers_used": "|".join(str(x) for x in layers_used),
                    "prompt_text": prompt,
                    "steered_output": out,
                    "latency_ms": round(latency, 2),
                    "steering_applied": steering_applied,
                    "model": model_key,
                }
                file_rows.append(row)
                all_rows.append(row)
                total_samples += 1

                if sample_idx % 20 == 0 or sample_idx == len(capped_rows):
                    logger.info(
                        "[%s][alpha=%s] samples %d/%d | file_steered=%d | total_samples=%d total_steered=%d",
                        file_path.name,
                        alpha,
                        sample_idx,
                        len(capped_rows),
                        file_steered,
                        total_samples,
                        total_steering_applied,
                    )

        # Save per-file outputs immediately.
        stem = file_path.stem
        file_csv = file_export_root / f"{stem}_steered.csv"
        with open(file_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=file_rows[0].keys())
            writer.writeheader()
            writer.writerows(file_rows)

        file_json = file_export_root / f"{stem}_steered.json"
        with open(file_json, "w", encoding="utf-8") as f:
            json.dump(file_rows, f, ensure_ascii=False, indent=2)

        # Also export per-file grouped by alpha for easier comparison.
        for alpha_label in alpha_labels:
            alpha_rows = [
                r
                for r in file_rows
                if str(r["alpha"]).replace(".", "p") == alpha_label
                or (
                    alpha_label == "default_or_calibrated"
                    and r["alpha"] == "default_or_calibrated"
                )
            ]
            if not alpha_rows:
                continue
            alpha_csv = file_export_root / f"{stem}_alpha_{alpha_label}.csv"
            with open(alpha_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=alpha_rows[0].keys())
                writer.writeheader()
                writer.writerows(alpha_rows)

        # Update combined outputs after every file.
        combined_csv = export_root / "all_categories_steered.csv"
        with open(combined_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)

        combined_json = export_root / "all_categories_steered.json"
        with open(combined_json, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, ensure_ascii=False, indent=2)

        logger.info(
            "Saved %s (%d rows). Running total: %d rows",
            file_path.name,
            len(file_rows),
            len(all_rows),
        )

    # Keep category CSV compatibility output.
    by_category: Dict[str, List[dict]] = {}
    for row in all_rows:
        by_category.setdefault(row["category"], []).append(row)
    for cat, cat_rows in by_category.items():
        cat_csv = export_root / f"{cat}.csv"
        with open(cat_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cat_rows[0].keys())
            writer.writeheader()
            writer.writerows(cat_rows)

    logger.info("Steered export written to %s", export_root)
    return all_rows


def run_full_eval_from_files(
    engine,
    model,
    tokenizer,
    model_key: str,
    test_dir: Path,
    output_dir: Path,
    use_azure: bool,
    max_prompts: int,
    max_new_tokens: int,
    languages: List[str],
    categories: List[str],
):
    """Run regular baseline-vs-steered evaluation from external CSV/XLSX prompt files."""
    prompt_slices = _load_prompt_files(test_dir, languages, categories, max_prompts)
    all_results: List[Dict] = []
    summary_rows: List[Dict] = []

    for key, rows in prompt_slices.items():
        lang, cat = key.split("/", 1)
        pairs = [{"prompt": r["prompt_text"]} for r in rows]
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
                "model": model_key,
                **result["metrics"],
            }
        )

    json_path = output_dir / "eval_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    csv_path = output_dir / "eval_results.csv"
    if summary_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)

    _print_summary(summary_rows)
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
    parser.add_argument(
        "--test-dir",
        type=str,
        default=None,
        help="Directory containing external CSV/XLSX prompt files",
    )
    parser.add_argument(
        "--steered-only",
        action="store_true",
        help="Only run steered generation and export outputs/latency",
    )
    args = parser.parse_args()

    run_evaluation(
        model_key=args.model,
        languages=args.languages,
        categories=args.categories,
        use_azure=args.azure,
        max_prompts=args.max_prompts,
        max_new_tokens=args.max_tokens,
        test_dir=Path(args.test_dir) if args.test_dir else None,
        steered_only=args.steered_only,
    )
