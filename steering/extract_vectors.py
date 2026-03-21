"""
SafeSteer-IN  ·  Steering Vector Extraction
=============================================
Extracts contrastive activation vectors from an Indic LLM for each
(language × harm-category) combination.

Pipeline:
    1. Load model with 4-bit quantisation (bitsandbytes)
    2. For each (lang, category) slice of the training set:
       a. Pass safe completions  → collect mean residual-stream activation
       b. Pass unsafe completions → collect mean residual-stream activation
       c. steering_vec = mean_unsafe − mean_safe   (per target layer)
       d. L2-normalise and save to disk
    3. (Optional) log vectors as Azure ML Data Assets

Outputs:
    steering_vectors/{model}/{lang}/{category}/layer{L}.pt
    steering_vectors/{model}/{lang}/{category}/metadata.json
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from tqdm import tqdm

# ── project imports ──────────────────────────────────────────────────────────
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    ACTIVE_MODEL,
    BNB_4BIT_CONFIG,
    HF_TOKEN,
    MODEL_CONFIGS,
    USE_4BIT,
    VECTORS_DIR,
    get_model_config,
)
from data.build_dataset import load_pairs_for_extraction
from steering.hooks import ActivationCollector

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────


def load_model_and_tokenizer(
    model_key: Optional[str] = None,
    quantize: Optional[bool] = None,
):
    """
    Load a HuggingFace CausalLM + tokenizer.
    Supports 4-bit quantisation via bitsandbytes when ``quantize=True``.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    cfg = MODEL_CONFIGS[model_key or ACTIVE_MODEL]
    model_id = cfg["model_id"]
    do_quantize = quantize if quantize is not None else USE_4BIT

    logger.info("Loading model: %s  (4-bit=%s)", model_id, do_quantize)

    tok_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
    model_kwargs["device_map"] = "auto"
    model_kwargs["torch_dtype"] = torch.float16

    if do_quantize:
        bnb_cfg = BitsAndBytesConfig(**BNB_4BIT_CONFIG)
        model_kwargs["quantization_config"] = bnb_cfg

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()

    logger.info("Model loaded on: %s", next(model.parameters()).device)
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Single-pair activation extraction
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def collect_activations(
    model,
    tokenizer,
    texts: List[str],
    target_layers: List[int],
    layer_accessor: str = "model.layers",
    max_length: int = 512,
    batch_size: int = 1,
    reduction: str = "mean",
) -> Dict[int, torch.Tensor]:
    """
    Run ``texts`` through the model and collect mean activations per layer.

    Returns:
        {layer_idx: Tensor of shape (hidden_dim,)}
    """
    collector = ActivationCollector(
        model,
        target_layers,
        layer_accessor,
        reduction=reduction,
    )

    # Accumulators: per-layer sum and count
    sums: Dict[int, torch.Tensor] = {}
    count = 0

    for i in tqdm(
        range(0, len(texts), batch_size), desc="Collecting activations", leave=False
    ):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)

        collector.clear()
        model(**inputs)
        acts = collector.get()  # {layer: (batch, hidden)}

        for layer_idx, act in acts.items():
            act_mean = act.mean(dim=0)  # average over batch → (hidden,)
            if layer_idx not in sums:
                sums[layer_idx] = torch.zeros_like(act_mean)
            sums[layer_idx] += act_mean
        count += 1

    collector.remove_hooks()

    # Final mean
    return {l: s / count for l, s in sums.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Steering vector computation for one (lang, category) pair
# ─────────────────────────────────────────────────────────────────────────────


def extract_vectors_for_slice(
    model,
    tokenizer,
    pairs: List[dict],
    target_layers: List[int],
    layer_accessor: str = "model.layers",
    max_length: int = 512,
) -> Dict[int, torch.Tensor]:
    """
    Given a list of contrastive pairs, compute the steering vector per layer:
        sv = mean_unsafe_activation − mean_safe_activation
    Then L2-normalise.
    """
    safe_texts = [p["prompt"] + " " + p["safe_response"] for p in pairs]
    unsafe_texts = [p["prompt"] + " " + p["unsafe_response"] for p in pairs]

    logger.info("  Extracting safe activations  (%d texts)", len(safe_texts))
    safe_acts = collect_activations(
        model,
        tokenizer,
        safe_texts,
        target_layers,
        layer_accessor,
        max_length,
    )

    logger.info("  Extracting unsafe activations (%d texts)", len(unsafe_texts))
    unsafe_acts = collect_activations(
        model,
        tokenizer,
        unsafe_texts,
        target_layers,
        layer_accessor,
        max_length,
    )

    vectors: Dict[int, torch.Tensor] = {}
    for layer_idx in target_layers:
        sv = unsafe_acts[layer_idx] - safe_acts[layer_idx]
        norm = sv.norm()
        if norm > 0:
            sv = sv / norm
        vectors[layer_idx] = sv
        logger.info("  Layer %d  |  norm(raw diff) = %.4f", layer_idx, norm.item())

    return vectors


# ─────────────────────────────────────────────────────────────────────────────
# Save / load vectors
# ─────────────────────────────────────────────────────────────────────────────


def save_vectors(
    vectors: Dict[int, torch.Tensor],
    model_key: str,
    language: str,
    category: str,
    base_dir: Optional[Path] = None,
    extra_meta: Optional[dict] = None,
):
    """Save steering vectors to disk and write a metadata.json."""
    out_dir = (base_dir or VECTORS_DIR) / model_key / language / category
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "model": model_key,
        "language": language,
        "category": category,
        "layers": {},
    }

    for layer_idx, vec in vectors.items():
        fpath = out_dir / f"layer{layer_idx}.pt"
        torch.save(vec, fpath)
        meta["layers"][str(layer_idx)] = {
            "file": fpath.name,
            "norm": vec.norm().item(),
            "shape": list(vec.shape),
        }
        logger.info("  Saved: %s", fpath)

    if extra_meta:
        meta.update(extra_meta)

    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("  Metadata: %s", meta_path)


def load_vector(
    model_key: str,
    language: str,
    category: str,
    layer_idx: int,
    base_dir: Optional[Path] = None,
) -> torch.Tensor:
    """Load a single steering vector from disk."""
    fpath = (
        (base_dir or VECTORS_DIR)
        / model_key
        / language
        / category
        / f"layer{layer_idx}.pt"
    )
    if not fpath.exists():
        raise FileNotFoundError(f"Steering vector not found: {fpath}")
    vec = torch.load(fpath, map_location="cpu")
    return vec


def load_all_vectors(
    model_key: str,
    base_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, Dict[int, torch.Tensor]]]:
    """
    Load all vectors for a model.
    Returns:  {language: {category: {layer_idx: Tensor}}}
    """
    root = (base_dir or VECTORS_DIR) / model_key
    if not root.exists():
        raise FileNotFoundError(f"No vectors found for model: {model_key}")

    result: Dict[str, Dict[str, Dict[int, torch.Tensor]]] = {}
    for lang_dir in root.iterdir():
        if not lang_dir.is_dir():
            continue
        lang = lang_dir.name
        result[lang] = {}
        for cat_dir in lang_dir.iterdir():
            if not cat_dir.is_dir():
                continue
            cat = cat_dir.name
            result[lang][cat] = {}
            for pt_file in cat_dir.glob("layer*.pt"):
                layer_num = int(pt_file.stem.replace("layer", ""))
                result[lang][cat][layer_num] = torch.load(pt_file, map_location="cpu")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Layer probe study
# ─────────────────────────────────────────────────────────────────────────────


def probe_best_layers(
    model,
    tokenizer,
    pairs: List[dict],
    candidate_layers: List[int],
    layer_accessor: str = "model.layers",
    max_length: int = 512,
    top_k: int = 3,
) -> List[int]:
    """
    Run a small probe study on ``pairs`` to rank layers by the magnitude
    of the contrastive activation difference.  Returns the top-k layers.
    """
    logger.info("Probing %d layers on %d pairs …", len(candidate_layers), len(pairs))
    vectors = extract_vectors_for_slice(
        model,
        tokenizer,
        pairs,
        candidate_layers,
        layer_accessor,
        max_length,
    )
    ranked = sorted(vectors.items(), key=lambda kv: kv[1].norm().item(), reverse=True)
    best = [l for l, _ in ranked[:top_k]]
    logger.info("Best layers by separation magnitude: %s", best)
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Full extraction pipeline
# ─────────────────────────────────────────────────────────────────────────────


def run_full_extraction(
    model_key: Optional[str] = None,
    languages: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    dataset_split: str = "train",
    probe_first: bool = True,
    probe_only: bool = False,
    probe_k: int = 3,
    probe_n_pairs: int = 50,
    max_length: int = 512,
    quantize: Optional[bool] = None,
):
    """
    Run extraction across all requested (language × category) slices.

    Steps:
        1. Load model
        2. (optional) Probe best layers on a small subset
        3. For each slice, extract and save vectors
    """
    from data.taxonomy import CATEGORY_KEYS
    from config import LANGUAGES

    _model_key = model_key or ACTIVE_MODEL
    cfg = MODEL_CONFIGS[_model_key]
    _languages = languages or list(LANGUAGES.keys())
    _categories = categories or CATEGORY_KEYS

    model, tokenizer = load_model_and_tokenizer(_model_key, quantize=quantize)
    accessor = cfg["layer_accessor"]
    candidate_layers = cfg["target_layers"]

    # Optional: probe to find best layers
    target_layers = candidate_layers
    if probe_first:
        # Use a small Hindi subset for the probe
        probe_pairs = load_pairs_for_extraction(
            dataset_split,
            language="hi",
            category=_categories[0],
        )[:probe_n_pairs]
        if len(probe_pairs) >= 5:
            target_layers = probe_best_layers(
                model,
                tokenizer,
                probe_pairs,
                candidate_layers,
                accessor,
                max_length,
                probe_k,
            )
        else:
            logger.warning("Not enough pairs for probing; using all candidate layers")
            target_layers = candidate_layers

    if not probe_only:
        # Save vectors from every configured target layer for downstream analysis.
        target_layers = list(candidate_layers)

    primary_layer = cfg.get("primary_layer")
    if primary_layer is not None and primary_layer not in target_layers:
        target_layers = list(target_layers) + [primary_layer]

    # Extract per slice
    t0 = time.time()
    for lang in _languages:
        for cat in _categories:
            pairs = load_pairs_for_extraction(
                dataset_split, language=lang, category=cat
            )
            if len(pairs) < 3:
                logger.warning("Skipping %s/%s — only %d pairs", lang, cat, len(pairs))
                continue

            logger.info(
                "━━━ Extracting: %s / %s  (%d pairs) ━━━", lang, cat, len(pairs)
            )
            vectors = extract_vectors_for_slice(
                model,
                tokenizer,
                pairs,
                target_layers,
                accessor,
                max_length,
            )
            save_vectors(
                vectors,
                _model_key,
                lang,
                cat,
                extra_meta={"num_pairs": len(pairs), "target_layers": target_layers},
            )

    elapsed = time.time() - t0
    logger.info("Full extraction finished in %.1f s", elapsed)
    return target_layers


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract steering vectors")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model key (openhathi-base | airavata | sarvam-1 | sarvam-m)",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Language codes to extract (default: all)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Category keys to extract (default: all)",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--no-probe", action="store_true", help="Skip the layer-probe study"
    )
    parser.add_argument("--probe-k", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    run_full_extraction(
        model_key=args.model,
        languages=args.languages,
        categories=args.categories,
        dataset_split=args.split,
        probe_first=not args.no_probe,
        probe_k=args.probe_k,
        max_length=args.max_length,
    )
