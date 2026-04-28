"""
SafeSteer-IN  ·  FastAPI REST API
====================================
Production-ready REST endpoint that wraps the SafeSteer-IN pipeline.

Endpoints:
    POST /steer       — run the full pipeline (classify → steer → score)
    POST /generate     — baseline generation only
    GET  /health       — liveness check
    GET  /slices       — list available (language, category) vectors

Launch:
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from config import API_HOST, API_PORT, DEFAULT_ALPHA, MAX_NEW_TOKENS

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
)

app = FastAPI(
    title="SafeSteer-IN API",
    description="Inference-Time Safety Steering for Indic LLMs",
    version="0.1.0",
)

REQUESTS_TOTAL = Counter("safesteer_requests_total", "Total requests", ["endpoint"])
ERRORS_TOTAL = Counter("safesteer_errors_total", "Total request errors", ["endpoint"])
FALLBACK_TOTAL = Counter(
    "safesteer_fallback_total", "Total routing fallbacks", ["endpoint"]
)
LATENCY_SECONDS = Histogram(
    "safesteer_latency_seconds", "Request latency in seconds", ["endpoint"]
)


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────────────────────


class SteerRequest(BaseModel):
    prompt: str = Field(..., description="User prompt in any supported Indic language")
    language: Optional[str] = Field(
        None, description="Force language code (e.g. 'hi', 'ta')"
    )
    category: Optional[str] = Field(None, description="Force harm category key")
    alpha: Optional[float] = Field(None, ge=0, le=50, description="Steering strength")
    max_new_tokens: int = Field(MAX_NEW_TOKENS, ge=1, le=1024)
    always_steer: bool = Field(
        True, description="Apply steering regardless of risk score"
    )


class SteerResponse(BaseModel):
    prompt: str
    detected_language: str
    detected_category: str
    risk_score: float
    steering_applied: bool
    alpha_used: float
    raw_output: str
    steered_output: str
    azure_score_raw: Optional[Dict] = None
    azure_score_steered: Optional[Dict] = None
    latency_ms: float


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(MAX_NEW_TOKENS, ge=1, le=1024)


class GenerateResponse(BaseModel):
    prompt: str
    output: str


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline singleton (lazy)
# ─────────────────────────────────────────────────────────────────────────────

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from engine.pipeline import SafeSteerPipeline

        _pipeline = SafeSteerPipeline(use_azure_safety=True)
        _pipeline.load()
    return _pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    REQUESTS_TOTAL.labels(endpoint="/health").inc()
    with LATENCY_SECONDS.labels(endpoint="/health").time():
        return {"status": "ok"}


@app.get("/ready")
async def ready():
    REQUESTS_TOTAL.labels(endpoint="/ready").inc()
    with LATENCY_SECONDS.labels(endpoint="/ready").time():
        try:
            pipeline = _get_pipeline()
            slices = pipeline.available_slices()
            return {"status": "ready", "loaded_slices": len(slices)}
        except Exception as e:
            ERRORS_TOTAL.labels(endpoint="/ready").inc()
            raise HTTPException(status_code=503, detail=str(e))


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/slices")
async def available_slices():
    """Return list of (language, category) pairs with loaded vectors."""
    REQUESTS_TOTAL.labels(endpoint="/slices").inc()
    with LATENCY_SECONDS.labels(endpoint="/slices").time():
        try:
            pipeline = _get_pipeline()
            slices = pipeline.available_slices()
            return {"slices": [{"language": l, "category": c} for l, c in slices]}
        except Exception as e:
            ERRORS_TOTAL.labels(endpoint="/slices").inc()
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/steer", response_model=SteerResponse)
async def steer(req: SteerRequest):
    """
    Main endpoint: classify the prompt, generate raw + steered outputs,
    and optionally score both with Azure Content Safety.
    """
    REQUESTS_TOTAL.labels(endpoint="/steer").inc()
    with LATENCY_SECONDS.labels(endpoint="/steer").time():
        try:
            pipeline = _get_pipeline()
            result = pipeline.run(
                prompt=req.prompt,
                force_language=req.language,
                force_category=req.category,
                alpha=req.alpha,
                max_new_tokens=req.max_new_tokens,
                always_steer=req.always_steer,
            )
            fallback_used = (
                result.steering_language_used != result.detected_language
                or result.steering_category_used != result.detected_category
            )
            if fallback_used:
                FALLBACK_TOTAL.labels(endpoint="/steer").inc()
            return SteerResponse(
                prompt=result.prompt,
                detected_language=result.detected_language,
                detected_category=result.detected_category,
                risk_score=result.risk_score,
                steering_applied=result.steering_applied,
                alpha_used=result.alpha_used,
                raw_output=result.raw_output,
                steered_output=result.steered_output,
                azure_score_raw=result.azure_score_raw,
                azure_score_steered=result.azure_score_steered,
                latency_ms=result.latency_ms,
            )
        except Exception as e:
            ERRORS_TOTAL.labels(endpoint="/steer").inc()
            logger.exception("Error in /steer")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Baseline generation without steering."""
    REQUESTS_TOTAL.labels(endpoint="/generate").inc()
    with LATENCY_SECONDS.labels(endpoint="/generate").time():
        try:
            pipeline = _get_pipeline()
            output = pipeline.engine.generate(
                req.prompt, max_new_tokens=req.max_new_tokens
            )
            return GenerateResponse(prompt=req.prompt, output=output)
        except Exception as e:
            ERRORS_TOTAL.labels(endpoint="/generate").inc()
            logger.exception("Error in /generate")
            raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Launch
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=False)
