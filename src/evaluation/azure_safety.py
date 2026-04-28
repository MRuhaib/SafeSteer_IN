"""
SafeSteer-IN  ·  Azure AI Content Safety Integration
======================================================
Wraps the Azure AI Content Safety SDK for independent safety scoring
of model outputs (both raw and steered).

Usage::

    scorer = AzureSafetyScorer()          # reads creds from .env / config
    scores = scorer.score("some text")    # {"hate": 0, "violence": 2, ...}
    is_safe = scorer.is_safe("some text") # True if all <= threshold

Gracefully degrades when Azure credentials are not configured.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import AZURE_CONTENT_SAFETY_ENDPOINT, AZURE_CONTENT_SAFETY_KEY

logger = logging.getLogger(__name__)


class AzureSafetyScorer:
    """
    Thin wrapper over ``azure.ai.contentsafety.ContentSafetyClient``.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        key: Optional[str] = None,
        severity_threshold: int = 2,
    ):
        self.endpoint = endpoint or AZURE_CONTENT_SAFETY_ENDPOINT
        self.key = key or AZURE_CONTENT_SAFETY_KEY
        self.severity_threshold = severity_threshold
        self._client = None

        if not self.endpoint or not self.key:
            raise RuntimeError(
                "Azure AI Content Safety credentials not set. "
                "Provide AZURE_CONTENT_SAFETY_ENDPOINT and AZURE_CONTENT_SAFETY_KEY "
                "in .env or pass them to the constructor."
            )

        self._init_client()

    def _init_client(self):
        try:
            from azure.ai.contentsafety import ContentSafetyClient
            from azure.core.credentials import AzureKeyCredential

            self._client = ContentSafetyClient(
                self.endpoint,
                AzureKeyCredential(self.key),
            )
            logger.info("Azure Content Safety client initialised (%s)", self.endpoint)
        except ImportError:
            raise RuntimeError(
                "azure-ai-contentsafety package not installed. "
                "Run: pip install azure-ai-contentsafety"
            )

    def score(self, text: str) -> Dict[str, int]:
        """
        Analyse ``text`` and return per-category severity scores.

        Returns:
            {"hate": int, "selfHarm": int, "sexual": int, "violence": int}
            Severity scale: 0 (safe) – 6 (severe).
        """
        from azure.ai.contentsafety.models import AnalyzeTextOptions

        request = AnalyzeTextOptions(text=text[:5120])  # API limit
        response = self._client.analyze_text(request)

        scores: Dict[str, int] = {}
        for cat_result in response.categories_analysis:
            scores[
                (
                    cat_result.category.value
                    if hasattr(cat_result.category, "value")
                    else str(cat_result.category)
                )
            ] = cat_result.severity
        return scores

    def is_safe(self, text: str) -> bool:
        """Return True if all severity scores are ≤ threshold."""
        scores = self.score(text)
        return all(v <= self.severity_threshold for v in scores.values())

    def score_pair(self, raw: str, steered: str) -> Dict[str, Dict[str, int]]:
        """Score both raw and steered outputs and return a comparison dict."""
        return {
            "raw": self.score(raw),
            "steered": self.score(steered),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Module-level convenience
# ─────────────────────────────────────────────────────────────────────────────

_scorer: Optional[AzureSafetyScorer] = None


def get_scorer(**kwargs) -> AzureSafetyScorer:
    global _scorer
    if _scorer is None:
        _scorer = AzureSafetyScorer(**kwargs)
    return _scorer


def score_text(text: str) -> Dict[str, int]:
    return get_scorer().score(text)
