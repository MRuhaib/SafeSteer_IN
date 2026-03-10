"""
SafeSteer-IN  ·  Core Steering Engine
=======================================
Manages the lifecycle of steering vectors + forward hooks for inference.

Responsibilities:
    1. Pre-load all steering vectors for the active model.
    2. Accept a (language, category, layer) selector and attach the right hook.
    3. Generate text with and without steering.
    4. Provide a clean context-manager / method API for the pipeline & demo.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    ACTIVE_MODEL,
    DEFAULT_ALPHA,
    MAX_NEW_TOKENS,
    MODEL_CONFIGS,
    TEMPERATURE,
    TOP_P,
    VECTORS_DIR,
)
from steering.hooks import SteeringHook, MultiLayerSteeringHook
from steering.extract_vectors import load_all_vectors, load_vector

logger = logging.getLogger(__name__)


class SteeringEngine:
    """
    High-level steering engine.

    Usage::

        engine = SteeringEngine(model, tokenizer, model_key="openhathi-base")
        raw = engine.generate(prompt)
        steered = engine.generate_steered(prompt, language="hi",
                                           category="communal_religious_hate")
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        model_key: str = ACTIVE_MODEL,
        vectors_dir: Optional[Path] = None,
        default_alpha: float = DEFAULT_ALPHA,
        calibration: Optional[Dict] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_key = model_key
        self.cfg = MODEL_CONFIGS[model_key]
        self.default_alpha = default_alpha
        self.primary_layer = self.cfg["primary_layer"]
        self.layer_accessor = self.cfg["layer_accessor"]

        # Load calibrated alphas if available
        self.calibrated_alphas: Dict[str, float] = {}
        self._load_calibration(calibration, vectors_dir)

        # Pre-load all vectors
        self.vectors: Dict[str, Dict[str, Dict[int, torch.Tensor]]] = {}
        self._load_vectors(vectors_dir)

    # ── Initialisation helpers ───────────────────────────────────────────

    def _load_vectors(self, vectors_dir: Optional[Path]):
        try:
            self.vectors = load_all_vectors(self.model_key, vectors_dir)
            total = sum(
                len(layers)
                for cats in self.vectors.values()
                for layers in cats.values()
            )
            logger.info("Loaded %d steering vectors for '%s'", total, self.model_key)
        except FileNotFoundError:
            logger.warning("No steering vectors found for '%s'", self.model_key)

    def _load_calibration(self, provided: Optional[Dict], vectors_dir: Optional[Path]):
        if provided:
            self.calibrated_alphas = provided
            return
        cal_path = (
            (vectors_dir or VECTORS_DIR) / self.model_key / "calibration_results.json"
        )
        if cal_path.exists():
            with open(cal_path) as f:
                data = json.load(f)
            for key, info in data.items():
                if isinstance(info, dict) and "best" in info:
                    self.calibrated_alphas[key] = info["best"]
            logger.info(
                "Loaded calibrated alphas for %d slices", len(self.calibrated_alphas)
            )

    # ── Vector lookup ────────────────────────────────────────────────────

    def get_vector(
        self,
        language: str,
        category: str,
        layer: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """Look up the steering vector for (lang, cat, layer)."""
        l = layer or self.primary_layer
        return self.vectors.get(language, {}).get(category, {}).get(l, None)

    def get_alpha(self, language: str, category: str) -> float:
        """Return calibrated alpha or the default."""
        key = f"{language}/{category}"
        return self.calibrated_alphas.get(key, self.default_alpha)

    def has_vector(self, language: str, category: str) -> bool:
        return self.get_vector(language, category) is not None

    def available_slices(self) -> List[Tuple[str, str]]:
        """Return list of (language, category) pairs that have vectors."""
        slices = []
        for lang, cats in self.vectors.items():
            for cat in cats:
                slices.append((lang, cat))
        return slices

    # ── Generation ───────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
    ) -> str:
        """Generate without steering (baseline)."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

    @torch.no_grad()
    def generate_steered(
        self,
        prompt: str,
        language: str,
        category: str,
        alpha: Optional[float] = None,
        layer: Optional[int] = None,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
    ) -> str:
        """Generate with the appropriate steering vector applied."""
        vec = self.get_vector(language, category, layer)
        if vec is None:
            logger.warning(
                "No vector for %s/%s — returning unsteered output", language, category
            )
            return self.generate(prompt, max_new_tokens, temperature, top_p)

        _alpha = alpha or self.get_alpha(language, category)
        _layer = layer or self.primary_layer

        hook = SteeringHook(
            self.model,
            _layer,
            vec,
            _alpha,
            self.layer_accessor,
        )
        try:
            result = self.generate(prompt, max_new_tokens, temperature, top_p)
        finally:
            hook.remove()
        return result

    # ── Convenience: side-by-side ────────────────────────────────────────

    @torch.no_grad()
    def compare(
        self,
        prompt: str,
        language: str,
        category: str,
        alpha: Optional[float] = None,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> Dict[str, str]:
        """Return both raw and steered outputs for side-by-side comparison."""
        raw = self.generate(prompt, max_new_tokens)
        steered = self.generate_steered(
            prompt, language, category, alpha, max_new_tokens=max_new_tokens
        )
        return {"raw": raw, "steered": steered}
