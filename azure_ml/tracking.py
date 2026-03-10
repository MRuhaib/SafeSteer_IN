"""
SafeSteer-IN  ·  Azure ML Experiment Tracking
================================================
Log metrics, artifacts (steering vectors), and evaluation results to
Azure ML using the MLflow integration.

All functions degrade gracefully when Azure ML is not configured —
they just print a warning and continue.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    AZURE_ML_EXPERIMENT,
    AZURE_ML_RESOURCE_GROUP,
    AZURE_ML_SUBSCRIPTION_ID,
    AZURE_ML_WORKSPACE,
    VECTORS_DIR,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# MLflow setup
# ─────────────────────────────────────────────────────────────────────────────

_mlflow_configured = False


def _configure_mlflow():
    """Point MLflow at the Azure ML tracking server (if available)."""
    global _mlflow_configured
    if _mlflow_configured:
        return True

    try:
        import mlflow
        from azureml.core import Workspace

        if not AZURE_ML_SUBSCRIPTION_ID:
            logger.info("Azure ML subscription not set — using local MLflow tracking")
            mlflow.set_experiment(AZURE_ML_EXPERIMENT)
            _mlflow_configured = True
            return True

        ws = Workspace.get(
            name=AZURE_ML_WORKSPACE,
            subscription_id=AZURE_ML_SUBSCRIPTION_ID,
            resource_group=AZURE_ML_RESOURCE_GROUP,
        )
        mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
        mlflow.set_experiment(AZURE_ML_EXPERIMENT)
        _mlflow_configured = True
        logger.info("MLflow tracking URI set to Azure ML workspace")
        return True

    except ImportError:
        logger.warning("mlflow or azureml-core not installed — tracking disabled")
        return False
    except Exception as e:
        logger.warning(
            "Could not configure Azure ML tracking: %s — using local MLflow", e
        )
        try:
            import mlflow

            mlflow.set_experiment(AZURE_ML_EXPERIMENT)
            _mlflow_configured = True
            return True
        except ImportError:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Logging functions
# ─────────────────────────────────────────────────────────────────────────────


def log_extraction_run(
    model_key: str,
    language: str,
    category: str,
    layer_idx: int,
    vector_norm: float,
    num_pairs: int,
    extra_params: Optional[Dict] = None,
):
    """Log a single steering-vector extraction run to MLflow."""
    if not _configure_mlflow():
        return

    import mlflow

    with mlflow.start_run(
        run_name=f"extract_{model_key}_{language}_{category}_L{layer_idx}"
    ):
        mlflow.log_params(
            {
                "model": model_key,
                "language": language,
                "category": category,
                "layer": layer_idx,
                "num_pairs": num_pairs,
                **(extra_params or {}),
            }
        )
        mlflow.log_metrics(
            {
                "vector_norm": vector_norm,
            }
        )

        # Log the vector file as an artifact
        vec_path = (
            VECTORS_DIR / model_key / language / category / f"layer{layer_idx}.pt"
        )
        if vec_path.exists():
            mlflow.log_artifact(str(vec_path), artifact_path="steering_vectors")

    logger.info(
        "Logged extraction run: %s/%s/%s/L%d", model_key, language, category, layer_idx
    )


def log_calibration_run(
    model_key: str,
    language: str,
    category: str,
    best_alpha: float,
    safety_rate: float,
    ppl_increase: float,
):
    """Log alpha calibration results."""
    if not _configure_mlflow():
        return

    import mlflow

    with mlflow.start_run(run_name=f"calibrate_{model_key}_{language}_{category}"):
        mlflow.log_params(
            {
                "model": model_key,
                "language": language,
                "category": category,
            }
        )
        mlflow.log_metrics(
            {
                "best_alpha": best_alpha,
                "safety_rate": safety_rate,
                "ppl_increase": ppl_increase,
            }
        )
    logger.info("Logged calibration: %s/%s α=%.1f", language, category, best_alpha)


def log_evaluation_results(
    summary_rows: List[Dict],
    model_key: str,
):
    """Log the full evaluation summary."""
    if not _configure_mlflow():
        return

    import mlflow

    with mlflow.start_run(run_name=f"eval_{model_key}"):
        mlflow.log_param("model", model_key)
        mlflow.log_param("num_slices", len(summary_rows))

        # Log aggregate metrics
        if summary_rows:
            avg_sir = sum(r.get("SIR", 0) for r in summary_rows) / len(summary_rows)
            avg_fpr = sum(r.get("FPR", 0) for r in summary_rows) / len(summary_rows)
            avg_lat = sum(r.get("avg_latency_ms", 0) for r in summary_rows) / len(
                summary_rows
            )
            mlflow.log_metrics(
                {
                    "avg_SIR": avg_sir,
                    "avg_FPR": avg_fpr,
                    "avg_latency_ms": avg_lat,
                }
            )

        # Log per-slice metrics
        for r in summary_rows:
            tag = f"{r['language']}_{r['category']}"
            mlflow.log_metrics(
                {
                    f"{tag}_SIR": r.get("SIR", 0),
                    f"{tag}_FPR": r.get("FPR", 0),
                }
            )

        # Log results files as artifacts
        from config import EVALUATION_DIR

        for f in EVALUATION_DIR.glob("eval_results.*"):
            mlflow.log_artifact(str(f), artifact_path="evaluation")

    logger.info(
        "Logged evaluation results for %s (%d slices)", model_key, len(summary_rows)
    )


def log_classifier_training(
    epochs: int,
    final_train_acc: float,
    final_val_acc: float,
    best_val_acc: float,
):
    """Log classifier training metrics."""
    if not _configure_mlflow():
        return

    import mlflow

    with mlflow.start_run(run_name="classifier_training"):
        mlflow.log_params({"epochs": epochs})
        mlflow.log_metrics(
            {
                "final_train_acc": final_train_acc,
                "final_val_acc": final_val_acc,
                "best_val_acc": best_val_acc,
            }
        )

        from config import CLASSIFIER_CHECKPOINT

        ckpt = Path(CLASSIFIER_CHECKPOINT)
        if ckpt.exists():
            mlflow.log_artifacts(str(ckpt), artifact_path="classifier")

    logger.info("Logged classifier training metrics")
