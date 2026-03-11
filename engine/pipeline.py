"""
SafeSteer-IN  ·  Full Inference Pipeline
==========================================
End-to-end pipeline that wires together:
    1. IndicBERT risk classifier   → detect language + harm category
    2. Steering engine             → apply vector if risk > threshold
    3. (Optional) Azure Content Safety → independent scoring

This is the module imported by `app.py` (Gradio) and `api.py` (FastAPI).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    ACTIVE_MODEL,
    DEFAULT_ALPHA,
    MAX_NEW_TOKENS,
    MODEL_CONFIGS,
    RISK_THRESHOLD,
    TEMPERATURE,
    TOP_P,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    prompt: str
    detected_language: str = ""
    detected_category: str = ""
    risk_score: float = 0.0
    steering_applied: bool = False
    alpha_used: float = 0.0
    raw_output: str = ""
    steered_output: str = ""
    azure_score_raw: Optional[Dict] = None
    azure_score_steered: Optional[Dict] = None
    latency_ms: float = 0.0


class SafeSteerPipeline:
    """
    One-call interface: `pipeline.run(prompt)` returns a `PipelineResult`.
    """

    def __init__(
        self,
        model_key: Optional[str] = None,
        risk_threshold: float = RISK_THRESHOLD,
        default_alpha: float = DEFAULT_ALPHA,
        use_azure_safety: bool = False,
        device: Optional[str] = None,
    ):
        self.model_key = model_key or ACTIVE_MODEL
        self.risk_threshold = risk_threshold
        self.default_alpha = default_alpha
        self.use_azure_safety = use_azure_safety
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Late-initialised components
        self._engine = None
        self._classifier = None
        self._azure_scorer = None
        self._loaded = False

    # ── Lazy loading ─────────────────────────────────────────────────────

    def load(self):
        """Load all components. Call once before serving."""
        if self._loaded:
            return

        from steering.extract_vectors import load_model_and_tokenizer
        from engine.steering_engine import SteeringEngine
        from classifier.inference import RiskPredictor

        logger.info("Loading pipeline components …")
        model, tokenizer = load_model_and_tokenizer(self.model_key)
        self._engine = SteeringEngine(
            model,
            tokenizer,
            self.model_key,
            default_alpha=self.default_alpha,
        )
        self._classifier = RiskPredictor()

        if self.use_azure_safety:
            try:
                from evaluation.azure_safety import AzureSafetyScorer

                self._azure_scorer = AzureSafetyScorer()
                logger.info("Azure Content Safety scorer enabled")
            except Exception as e:
                logger.warning("Azure Content Safety unavailable: %s", e)

        self._loaded = True
        logger.info(
            "Pipeline ready  (model=%s, device=%s)", self.model_key, self._device
        )

    # ── Main entry point ─────────────────────────────────────────────────

    def run(
        self,
        prompt: str,
        force_language: Optional[str] = None,
        force_category: Optional[str] = None,
        alpha: Optional[float] = None,
        max_new_tokens: int = MAX_NEW_TOKENS,
        always_steer: bool = False,
    ) -> PipelineResult:
        """
        Full pipeline execution:
            1. Classify the prompt (language + category + risk)
            2. Generate baseline output
            3. If risk ≥ threshold (or always_steer), generate steered output
            4. Optionally score both via Azure Content Safety
        """
        if not self._loaded:
            self.load()

        t0 = time.time()
        result = PipelineResult(prompt=prompt)

        # Step 1: Classification
        cls = self._classifier.predict(prompt)
        result.detected_language = force_language or cls["language"]
        result.detected_category = force_category or cls["category"]
        result.risk_score = cls["risk_score"]

        # Step 2: Baseline generation
        result.raw_output = self._engine.generate(prompt, max_new_tokens)

        # Step 3: Steered generation
        should_steer = always_steer or result.risk_score >= self.risk_threshold
        steer_lang = result.detected_language
        steer_cat = result.detected_category
        if should_steer and not self._engine.has_vector(steer_lang, steer_cat):
            # Fall back to the best available vector for this language, then hi
            available = self._engine.available_slices()
            fallback = next(
                (s for s in available if s[0] == steer_lang),
                next((s for s in available if s[0] == "hi"), None),
            )
            if fallback:
                steer_lang, steer_cat = fallback
                logger.info(
                    "No vector for %s/%s — falling back to %s/%s",
                    result.detected_language,
                    result.detected_category,
                    steer_lang,
                    steer_cat,
                )

        if should_steer and self._engine.has_vector(steer_lang, steer_cat):
            _alpha = alpha or self._engine.get_alpha(steer_lang, steer_cat)
            result.steered_output = self._engine.generate_steered(
                prompt,
                steer_lang,
                steer_cat,
                alpha=_alpha,
                max_new_tokens=max_new_tokens,
            )
            result.steering_applied = True
            result.alpha_used = _alpha
        else:
            result.steered_output = result.raw_output
            result.steering_applied = False
            if should_steer:
                logger.info("No steering vectors available — showing raw output")

        # Step 4: Azure Content Safety scoring
        if self._azure_scorer is not None:
            try:
                result.azure_score_raw = self._azure_scorer.score(result.raw_output)
                result.azure_score_steered = self._azure_scorer.score(
                    result.steered_output
                )
            except Exception as e:
                logger.warning("Azure scoring failed: %s", e)

        result.latency_ms = (time.time() - t0) * 1000
        return result

    # ── Convenience accessors ────────────────────────────────────────────

    @property
    def engine(self):
        if not self._loaded:
            self.load()
        return self._engine

    @property
    def classifier(self):
        if not self._loaded:
            self.load()
        return self._classifier

    def available_slices(self):
        if not self._loaded:
            self.load()
        return self._engine.available_slices()


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_pipeline: Optional[SafeSteerPipeline] = None


def get_pipeline(**kwargs) -> SafeSteerPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = SafeSteerPipeline(**kwargs)
    return _pipeline
