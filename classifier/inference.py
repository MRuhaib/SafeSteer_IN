"""
SafeSteer-IN  ·  IndicBERT Risk Classifier — Inference
========================================================
Load a trained IndicRiskClassifier checkpoint and run inference on a
single prompt or batch of prompts.

Returns: (predicted_language, predicted_category, risk_score)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CLASSIFIER_CHECKPOINT, INDICBERT_MODEL_ID
from classifier.train_classifier import (
    CAT_TO_ID,
    LANG_TO_ID,
    IndicRiskClassifier,
)

logger = logging.getLogger(__name__)

# Reverse maps
ID_TO_LANG = {v: k for k, v in LANG_TO_ID.items()}
ID_TO_CAT = {v: k for k, v in CAT_TO_ID.items()}


class RiskPredictor:
    """
    Wraps IndicRiskClassifier for inference.

    Falls back to a simple rule-based detector when no trained checkpoint
    is found (useful for early demo before the classifier is trained).
    """

    def __init__(
        self,
        checkpoint_dir: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = Path(checkpoint_dir or CLASSIFIER_CHECKPOINT)
        self._model = None
        self._tokenizer = None

        if (ckpt / "classifier.pt").exists():
            self._load_trained(ckpt)
        else:
            logger.warning(
                "No trained classifier at %s — using rule-based fallback", ckpt
            )

    def _load_trained(self, ckpt: Path):
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
        self._model = IndicRiskClassifier()
        state = torch.load(ckpt / "classifier.pt", map_location=self._device)
        self._model.load_state_dict(state)
        self._model.to(self._device)
        self._model.eval()
        logger.info("Loaded risk classifier from %s", ckpt)

    # ── Main predict ─────────────────────────────────────────────────────

    def predict(self, prompt: str) -> Dict:
        """
        Returns dict with keys:
            language      : str   (e.g. "hi")
            category      : str   (e.g. "communal_religious_hate")
            risk_score    : float (0–1, higher = more harmful)
            lang_probs    : dict  {lang: prob}
            cat_probs     : dict  {cat: prob}
        """
        if self._model is not None:
            return self._predict_trained(prompt)
        return self._predict_fallback(prompt)

    @torch.no_grad()
    def _predict_trained(self, prompt: str) -> Dict:
        enc = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length",
        )
        input_ids = enc["input_ids"].to(self._device)
        attention_mask = enc["attention_mask"].to(self._device)

        lang_logits, cat_logits = self._model(input_ids, attention_mask)

        lang_probs = F.softmax(lang_logits, dim=-1).squeeze(0).cpu().numpy()
        cat_probs = F.softmax(cat_logits, dim=-1).squeeze(0).cpu().numpy()

        lang_id = int(lang_probs.argmax())
        cat_id = int(cat_probs.argmax())

        # Risk score = max category probability (how confident the model is
        # that the prompt belongs to a harmful category)
        risk_score = float(cat_probs.max())

        return {
            "language": ID_TO_LANG.get(lang_id, "hi"),
            "category": ID_TO_CAT.get(cat_id, "communal_religious_hate"),
            "risk_score": risk_score,
            "lang_probs": {ID_TO_LANG[i]: float(p) for i, p in enumerate(lang_probs)},
            "cat_probs": {ID_TO_CAT[i]: float(p) for i, p in enumerate(cat_probs)},
        }

    # ── Rule-based fallback ──────────────────────────────────────────────

    def _predict_fallback(self, prompt: str) -> Dict:
        """
        Simple heuristic detector for use before the classifier is trained.
        """
        lang = self._detect_language_heuristic(prompt)
        cat, score = self._detect_category_heuristic(prompt)
        return {
            "language": lang,
            "category": cat,
            "risk_score": score,
            "lang_probs": {lang: 1.0},
            "cat_probs": {cat: score},
        }

    @staticmethod
    def _detect_language_heuristic(text: str) -> str:
        """Detect language by Unicode block heuristics."""
        devanagari = sum(1 for c in text if "\u0900" <= c <= "\u097f")
        tamil = sum(1 for c in text if "\u0b80" <= c <= "\u0bff")
        bengali = sum(1 for c in text if "\u0980" <= c <= "\u09ff")
        gujarati = sum(1 for c in text if "\u0a80" <= c <= "\u0aff")
        latin = sum(1 for c in text if "a" <= c.lower() <= "z")

        counts = {
            "hi": devanagari,
            "ta": tamil,
            "bn": bengali,
            "gu": gujarati,
        }

        if not any(counts.values()):
            return "hi-en"  # all Latin → likely Hinglish / English

        dominant = max(counts, key=counts.get)

        # Hinglish detection: mixed Devanagari + Latin
        if dominant == "hi" and latin > devanagari * 0.3:
            return "hi-en"

        # Marathi vs Hindi: rough heuristic (Marathi shares Devanagari)
        marathi_markers = ["आहे", "आहेत", "नाही", "पाहिजे", "करा", "सांगा"]
        if dominant == "hi" and any(m in text for m in marathi_markers):
            return "mr"

        return dominant

    @staticmethod
    def _detect_category_heuristic(text: str) -> Tuple[str, float]:
        """Keyword-based harm-category detection."""
        from data.taxonomy import TAXONOMY

        text_lower = text.lower()
        best_cat = "communal_religious_hate"
        best_score = 0.1  # default low risk

        for key, cat in TAXONOMY.items():
            hits = 0
            total_kw = len(cat.keywords_hi) + len(cat.keywords_en)
            if total_kw == 0:
                continue
            for kw in cat.keywords_hi + cat.keywords_en:
                if kw.lower() in text_lower:
                    hits += 1
            score = hits / total_kw
            if score > best_score:
                best_score = score
                best_cat = key

        # Cap the score at 0.95 for the heuristic
        return best_cat, min(best_score, 0.95)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level convenience
# ─────────────────────────────────────────────────────────────────────────────

_global_predictor: Optional[RiskPredictor] = None


def get_predictor(**kwargs) -> RiskPredictor:
    """Singleton accessor."""
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = RiskPredictor(**kwargs)
    return _global_predictor


def classify_prompt(prompt: str) -> Dict:
    """Quick convenience function."""
    return get_predictor().predict(prompt)
