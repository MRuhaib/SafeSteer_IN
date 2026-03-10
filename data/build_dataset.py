"""
SafeSteer-IN  ·  Dataset Construction Pipeline
================================================
Builds the India Safety contrastive-pair dataset from three sources:

1. **Templates** — hand-crafted seed pairs from `data/templates.py`
2. **HuggingFace** — existing open-source safety / toxicity datasets
   (textdetox multilingual, ai4bharat/indic-align toxic, india hate speech,
    MultiJail, Aya, etc.)
3. **Augmentation** — paraphrase-based expansion of seed pairs

Outputs JSONL files per split (train / val / test) at:
    data/datasets/{split}.jsonl

Each line follows the schema:
    {"prompt": "...", "safe_response": "...", "unsafe_response": "...",
     "language": "hi", "category": "communal_religious_hate"}
"""

from __future__ import annotations

import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

# ── project imports ──────────────────────────────────────────────────────────
import sys, os

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DATASET_DIR,
    DATASET_SPLIT_RATIOS,
    HARM_CATEGORIES,
    LANGUAGES,
    PAIRS_PER_CATEGORY,
)
from data.taxonomy import TAXONOMY, CATEGORY_KEYS
from data.templates import ALL_TEMPLATES

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load seed templates
# ─────────────────────────────────────────────────────────────────────────────


def load_seed_templates() -> List[dict]:
    """Flatten ALL_TEMPLATES into a list of dicts ready for JSONL."""
    pairs: List[dict] = []
    for lang, cat_dict in ALL_TEMPLATES.items():
        for category, items in cat_dict.items():
            for item in items:
                pairs.append(
                    {
                        "prompt": item["prompt"],
                        "safe_response": item["safe"],
                        "unsafe_response": item["unsafe"],
                        "language": lang,
                        "category": category,
                    }
                )
    logger.info("Loaded %d seed template pairs", len(pairs))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# 2.  HuggingFace dataset loaders
# ─────────────────────────────────────────────────────────────────────────────


def _try_load_hf_dataset(dataset_id: str, **kwargs):
    """Attempt to load a HuggingFace dataset; return None on failure."""
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_id, **kwargs)
        logger.info("Loaded HuggingFace dataset: %s", dataset_id)
        return ds
    except Exception as e:
        logger.warning("Could not load HF dataset '%s': %s", dataset_id, e)
        return None


def load_textdetox_pairs() -> List[dict]:
    """
    Load multilingual toxicity data from textdetox/multilingual_toxicity_dataset.

    This is the PAN 2024 multilingual toxicity dataset — free, no gating,
    no trust_remote_code required.  Each record has a ``text`` field plus a
    binary toxicity label (column name varies by split but defaults to
    ``toxic`` / ``label``).

    We use Hindi (``hi``) and Hinglish (``hinglish``) splits.
    Toxic texts become ``unsafe_response``; non-toxic texts become
    ``safe_response``.  We randomly pair them to build contrastive pairs.
    """
    pairs: List[dict] = []
    # Map HF split name → our language code
    # Note: the dataset uses "hin" for Hinglish (code-mixed Hindi-English)
    split_lang_map = {"hi": "hi", "hin": "hi-en"}

    for hf_split, lang_code in split_lang_map.items():
        ds = _try_load_hf_dataset(
            "textdetox/multilingual_toxicity_dataset", split=hf_split
        )
        if ds is None:
            continue

        # Determine the label column robustly
        label_col = None
        for candidate in ("toxic", "label", "toxicity", "is_toxic"):
            if candidate in ds.column_names:
                label_col = candidate
                break
        if label_col is None:
            logger.warning(
                "textdetox/%s: could not find a label column (columns=%s)",
                hf_split,
                ds.column_names,
            )
            continue

        # Separate toxic and non-toxic
        toxic_texts: List[str] = []
        safe_texts: List[str] = []
        for row in ds:
            text = row.get("text", "")
            if not text:
                continue
            label = row.get(label_col, 0)
            # Label is 1 or "toxic" → unsafe; 0 or "non-toxic" → safe
            is_toxic = (label == 1) or (str(label).lower() in ("toxic", "1", "true"))
            if is_toxic:
                toxic_texts.append(text)
            else:
                safe_texts.append(text)

        if not toxic_texts or not safe_texts:
            logger.warning(
                "textdetox/%s: no usable texts (toxic=%d, safe=%d)",
                hf_split,
                len(toxic_texts),
                len(safe_texts),
            )
            continue

        random.shuffle(safe_texts)
        n = min(len(toxic_texts), len(safe_texts), 500)  # cap per language
        for i in range(n):
            pairs.append(
                {
                    "prompt": f"निम्नलिखित विषय पर अपना विचार दें: {toxic_texts[i][:80]}",
                    "unsafe_response": toxic_texts[i],
                    "safe_response": safe_texts[i],
                    "language": lang_code,
                    "category": "code_mixed_toxicity",
                    "source": "textdetox",
                }
            )

    logger.info(
        "Loaded %d pairs from textdetox/multilingual_toxicity_dataset", len(pairs)
    )
    return pairs


def load_indic_align_toxic_pairs() -> List[dict]:
    """
    Load toxic-alignment pairs from ai4bharat/indic-align (IndicAlign-Toxic).

    IndicAlign-Toxic contains:
      - HHRLHF-T : Anthropic HH-RLHF toxic prompts with LLaMA-2-70B refusals
      - Toxic-Matrix : Synthetic toxic prompts + non-toxic responses

    Both sub-sets already have a *prompt* and a *safe refusal*, making them
    ideal for CAA contrastive pairs.  We treat the prompt as the harmful
    query and the provided response as the safe_response; the prompt itself
    (containing the harmful intent) serves as the unsafe_response.
    """
    pairs: List[dict] = []

    lang_map = {
        "Hindi": "hi",
        "Tamil": "ta",
        "Bengali": "bn",
        "Gujarati": "gu",
        "Marathi": "mr",
        "hi": "hi",
        "ta": "ta",
        "bn": "bn",
        "gu": "gu",
        "mr": "mr",
    }

    # IndicAlign has two toxic sub-configs: HHRLHF_T and Toxic_Matrix
    for config_name in ["HHRLHF_T", "Toxic_Matrix"]:
        ds = _try_load_hf_dataset("ai4bharat/indic-align", name=config_name)
        if ds is None:
            continue

        # DatasetDict.column_names is a dict; Dataset.column_names is a list
        if isinstance(ds.column_names, list):
            split = ds
        else:
            split = ds.get("train", list(ds.values())[0] if ds else None)

        if split is None:
            continue

        for row in split:
            prompt_text = row.get(
                "prompt", row.get("question", row.get("input", row.get("text", "")))
            )
            response_text = row.get(
                "response", row.get("answer", row.get("output", row.get("targets", "")))
            )
            if not prompt_text or not response_text:
                continue

            lang_raw = row.get("language", row.get("lang", "hi"))
            lang_code = lang_map.get(lang_raw, "hi")
            if lang_code not in LANGUAGES:
                continue

            pairs.append(
                {
                    "prompt": prompt_text,
                    "safe_response": response_text,
                    "unsafe_response": prompt_text,  # toxic prompt as the unsafe signal
                    "language": lang_code,
                    "category": "communal_religious_hate",
                    "source": "indic_align_toxic",
                }
            )

    logger.info("Loaded %d pairs from ai4bharat/indic-align (toxic)", len(pairs))
    return pairs


def load_india_hate_speech_pairs() -> List[dict]:
    """
    Load India hate speech superset (manueltonneau/india-hate-speech-superset).

    This is a gated dataset — requires a HuggingFace token AND the user to
    have accepted the dataset terms on the HF Hub.  We attempt the load and
    silently skip on failure.

    Columns: text, labels (0=non-hate, 1=hate), source, dataset, nb_annotators
    """
    import os

    pairs: List[dict] = []
    hf_token = os.getenv("HF_TOKEN", None)
    ds = _try_load_hf_dataset(
        "manueltonneau/india-hate-speech-superset",
        token=hf_token,
    )
    if ds is None:
        return pairs

    split = ds.get("train", list(ds.values())[0] if ds else None)
    if split is None:
        return pairs

    hate_texts: List[str] = []
    safe_texts: List[str] = []
    for row in split:
        text = row.get("text", "")
        label = row.get("labels", row.get("label", 0))
        if not text:
            continue
        if label == 1 or str(label) == "1":
            hate_texts.append(text)
        else:
            safe_texts.append(text)

    random.shuffle(safe_texts)
    n = min(len(hate_texts), len(safe_texts), 500)
    for i in range(n):
        pairs.append(
            {
                "prompt": f"इस विषय पर टिप्पणी करें: {hate_texts[i][:80]}",
                "unsafe_response": hate_texts[i],
                "safe_response": safe_texts[i],
                "language": "hi",
                "category": "communal_religious_hate",
                "source": "india_hate_speech",
            }
        )

    logger.info(
        "Loaded %d pairs from manueltonneau/india-hate-speech-superset", len(pairs)
    )
    return pairs


def load_multijail_pairs() -> List[dict]:
    """
    Load MultiJail jailbreak prompts in Hindi / Bengali / Tamil.
    HF ID: DAMO-NLP-SG/MultiJail  — plain parquet, no trust_remote_code.

    MultiJail is wide-format: each row is one jailbreak prompt with a
    separate column for each language (e.g. ``hi``, ``bn``, ``ta``).
    There is no ``language`` column per row.
    """
    pairs: List[dict] = []
    ds = _try_load_hf_dataset("DAMO-NLP-SG/MultiJail")
    if ds is None:
        return pairs

    # MultiJail only has Bengali among our supported Indic languages.
    # Actual columns: en, zh, it, vi, ar, ko, th, bn, sw, jv  (no hi/ta/mr)
    lang_col_map = {"bn": "bn"}

    for split_name in ["train", "test"]:
        split = ds.get(split_name)
        if split is None:
            continue
        for row in split:
            for col, lang_code in lang_col_map.items():
                text = row.get(col, "")
                if not text:
                    continue
                pairs.append(
                    {
                        "prompt": text,
                        "safe_response": generate_safe_refusal(lang_code),
                        "unsafe_response": "",
                        "language": lang_code,
                        "category": "code_mixed_toxicity",
                        "source": "multijail",
                    }
                )
    logger.info("Loaded %d pairs from MultiJail", len(pairs))
    return pairs


def load_aya_safe_completions() -> List[dict]:
    """
    Load safe Indic-language completions from Cohere Aya dataset.
    These serve as the safe half of contrastive pairs.
    HF ID: CohereForAI/aya_dataset  — no trust_remote_code needed.
    """
    safe_items: List[dict] = []
    ds = _try_load_hf_dataset("CohereForAI/aya_dataset")
    if ds is None:
        return safe_items

    lang_map = {
        "Hindi": "hi",
        "Tamil": "ta",
        "Bengali": "bn",
        "Gujarati": "gu",
        "Marathi": "mr",
    }
    split = ds.get("train", None)
    if split is None:
        return safe_items

    for row in split:
        lang_full = row.get("language", "")
        lang_code = lang_map.get(lang_full, None)
        if lang_code is None:
            continue
        inputs = row.get("inputs", row.get("prompt", ""))
        targets = row.get("targets", row.get("completion", ""))
        if inputs and targets:
            safe_items.append(
                {
                    "prompt": inputs,
                    "safe_response": targets,
                    "language": lang_code,
                }
            )
    logger.info("Loaded %d safe completions from Aya", len(safe_items))
    return safe_items


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Paraphrase / augmentation helpers
# ─────────────────────────────────────────────────────────────────────────────

# Refusal templates per language — used to generate safe responses
REFUSAL_TEMPLATES = {
    "hi": [
        "मुझे इस विषय पर हानिकारक सामग्री देने में असमर्थता है। कृपया किसी सकारात्मक विषय पर बात करें।",
        "यह अनुरोध अनुचित है। मैं ऐसी जानकारी देने से मना करता हूँ। क्या मैं किसी अन्य तरीके से मदद कर सकता हूँ?",
        "मैं ऐसी सामग्री प्रदान नहीं कर सकता जो हानिकारक हो सकती है। आइए किसी रचनात्मक विषय पर चर्चा करें।",
    ],
    "ta": [
        "இந்த தலைப்பில் தீங்கு விளைவிக்கும் உள்ளடக்கத்தை வழங்க இயலாது. தயவுசெய்து வேறு தலைப்பில் பேசுங்கள்.",
        "இந்த கோரிக்கை பொருத்தமற்றது. இதுபோன்ற தகவல்களை நான் வழங்க மறுக்கிறேன்.",
        "தீங்கு விளைவிக்கும் உள்ளடக்கத்தை வழங்க முடியாது. ஆக்கப்பூர்வமான தலைப்பில் பேசலாம்.",
    ],
    "bn": [
        "এই বিষয়ে ক্ষতিকর বিষয়বস্তু দেওয়া আমার পক্ষে সম্ভব নয়। দয়া করে অন্য বিষয়ে কথা বলুন।",
        "এই অনুরোধ অনুচিত। আমি এই ধরনের তথ্য দিতে অস্বীকার করছি।",
    ],
    "gu": [
        "આ વિષય પર હાનિકારક સામગ્રી આપવી મારા માટે શક્ય નથી. કૃપયા અન્ય વિષય પર વાત કરો.",
        "આ વિનંતી અયોગ્ય છે. હું આ પ્રકારની માહિતી આપવાનો ઇનકાર કરું છું.",
    ],
    "mr": [
        "या विषयावर हानिकारक सामग्री देणे मला शक्य नाही. कृपया इतर विषयावर बोला.",
        "ही विनंती अयोग्य आहे. मी अशी माहिती देण्यास नकार देतो.",
    ],
    "hi-en": [
        "Main is topic pe harmful content provide nahi kar sakta. Please kisi positive topic pe baat karte hain.",
        "Ye request inappropriate hai. Main aisi information dene se mana karta hun. Kya main kisi aur tarike se help kar sakta hun?",
        "Mujhe aisi content dene mein asamarthta hai jo harmful ho sakti hai. Aao kisi constructive topic pe discuss karte hain.",
    ],
}


def generate_safe_refusal(lang: str) -> str:
    """Pick a random refusal template for the given language."""
    templates = REFUSAL_TEMPLATES.get(lang, REFUSAL_TEMPLATES["hi"])
    return random.choice(templates)


def augment_pair(pair: dict, n: int = 3) -> List[dict]:
    """
    Simple augmentation: create n slight variants of the pair by
    prepending / appending contextual wrappers.
    For production use, replace with IndicTrans2 or GPT-4o paraphrasing.
    """
    wrappers_hi = [
        ("इस बारे में विस्तार से बताओ: ", ""),
        ("मुझे बताओ, ", ""),
        ("", " — विस्तार से बताइए।"),
    ]
    wrappers_en = [
        ("Tell me about this: ", ""),
        ("Explain in detail: ", ""),
        ("", " — give a detailed answer."),
    ]

    lang = pair.get("language", "hi")
    wrappers = wrappers_en if lang == "hi-en" else wrappers_hi

    augmented = []
    for i in range(min(n, len(wrappers))):
        prefix, suffix = wrappers[i]
        new_pair = pair.copy()
        new_pair["prompt"] = prefix + pair["prompt"] + suffix
        augmented.append(new_pair)
    return augmented


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Master build function
# ─────────────────────────────────────────────────────────────────────────────


def build_dataset(
    augment: bool = True,
    augment_n: int = 2,
    load_hf: bool = True,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict[str, List[dict]]:
    """
    Build the full contrastive-pair dataset.

    Returns dict  {"train": [...], "val": [...], "test": [...]}
    and writes JSONL files to output_dir.
    """
    random.seed(seed)
    out = output_dir or DATASET_DIR
    out.mkdir(parents=True, exist_ok=True)

    # ── Collect from all sources ─────────────────────────────────────────
    all_pairs: List[dict] = []

    # Source 1: seed templates
    all_pairs.extend(load_seed_templates())

    # Source 2: HuggingFace (optional, graceful fallback)
    if load_hf:
        all_pairs.extend(load_textdetox_pairs())
        all_pairs.extend(load_indic_align_toxic_pairs())
        all_pairs.extend(load_india_hate_speech_pairs())
        all_pairs.extend(load_multijail_pairs())
        # Aya safe completions — merge into existing pairs as safe half
        aya = load_aya_safe_completions()
        # Store separately; they don't have unsafe halves so we use them
        # to replace generic refusals in matching-language pairs later
        aya_by_lang: Dict[str, List[dict]] = {}
        for item in aya:
            aya_by_lang.setdefault(item["language"], []).append(item)
        logger.info(
            "Aya safe completions indexed by language: %s",
            {k: len(v) for k, v in aya_by_lang.items()},
        )

    # ── Fill missing safe/unsafe halves ──────────────────────────────────
    for pair in all_pairs:
        if not pair.get("safe_response"):
            pair["safe_response"] = generate_safe_refusal(pair["language"])
        if not pair.get("unsafe_response"):
            # Leave empty; extraction code will skip pairs without both halves
            pair["unsafe_response"] = ""

    # ── Filter: keep only pairs that have both halves ────────────────────
    complete = [p for p in all_pairs if p["safe_response"] and p["unsafe_response"]]
    incomplete = len(all_pairs) - len(complete)
    if incomplete:
        logger.warning(
            "%d pairs lack an unsafe_response and will be saved separately "
            "for manual completion",
            incomplete,
        )
        # Save incomplete for manual filling
        _write_jsonl(
            out / "incomplete_pairs.jsonl",
            [p for p in all_pairs if not p["unsafe_response"]],
        )

    # ── Augmentation ─────────────────────────────────────────────────────
    if augment and complete:
        aug_pairs: List[dict] = []
        for pair in complete:
            aug_pairs.extend(augment_pair(pair, n=augment_n))
        logger.info(
            "Augmented %d → %d pairs", len(complete), len(complete) + len(aug_pairs)
        )
        complete.extend(aug_pairs)

    # ── Deduplicate on prompt text ───────────────────────────────────────
    seen = set()
    deduped: List[dict] = []
    for p in complete:
        key = (p["prompt"], p["language"], p["category"])
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    logger.info("After dedup: %d pairs", len(deduped))

    # ── Split ────────────────────────────────────────────────────────────
    random.shuffle(deduped)
    n = len(deduped)
    train_end = int(n * DATASET_SPLIT_RATIOS["train"])
    val_end = train_end + int(n * DATASET_SPLIT_RATIOS["val"])

    splits = {
        "train": deduped[:train_end],
        "val": deduped[train_end:val_end],
        "test": deduped[val_end:],
    }

    for name, items in splits.items():
        path = out / f"{name}.jsonl"
        _write_jsonl(path, items)
        logger.info("Wrote %s  →  %d pairs", path, len(items))

    # ── Stats ────────────────────────────────────────────────────────────
    _print_stats(splits)
    return splits


def _write_jsonl(path: Path, items: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _print_stats(splits: Dict[str, List[dict]]):
    """Print a quick summary table."""
    from collections import Counter

    for split_name, items in splits.items():
        lang_counts = Counter(p["language"] for p in items)
        cat_counts = Counter(p["category"] for p in items)
        print(f"\n{'='*60}")
        print(f"  Split: {split_name}  ({len(items)} pairs)")
        print(f"{'='*60}")
        print("  Languages:", dict(lang_counts))
        print("  Categories:", dict(cat_counts))


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Utilities for loading the built dataset
# ─────────────────────────────────────────────────────────────────────────────


def load_dataset_split(
    split: str = "train", dataset_dir: Optional[Path] = None
) -> List[dict]:
    """Load a JSONL split from disk."""
    path = (dataset_dir or DATASET_DIR) / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset split not found at {path}. Run build_dataset() first."
        )
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    logger.info("Loaded %d pairs from %s", len(items), path)
    return items


def load_pairs_for_extraction(
    split: str = "train",
    language: Optional[str] = None,
    category: Optional[str] = None,
    dataset_dir: Optional[Path] = None,
) -> List[dict]:
    """Load pairs filtered by language and/or category — ready for vector extraction."""
    items = load_dataset_split(split, dataset_dir)
    if language:
        items = [p for p in items if p["language"] == language]
    if category:
        items = [p for p in items if p["category"] == category]
    # Ensure both halves present
    items = [p for p in items if p.get("safe_response") and p.get("unsafe_response")]
    logger.info(
        "Filtered → %d complete pairs (lang=%s, cat=%s)", len(items), language, category
    )
    return items


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build SafeSteer-IN dataset")
    parser.add_argument(
        "--no-augment", action="store_true", help="Disable augmentation"
    )
    parser.add_argument(
        "--no-hf", action="store_true", help="Skip HuggingFace datasets"
    )
    parser.add_argument("--augment-n", type=int, default=2, help="Augmentation factor")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else None
    build_dataset(
        augment=not args.no_augment,
        augment_n=args.augment_n,
        load_hf=not args.no_hf,
        seed=args.seed,
        output_dir=out_dir,
    )
    print("\n✓ Dataset build complete.")
