"""
MLflow Experiment Tracking Integration
========================================
Core logging interface for all pipeline stages. Provides structured MLflow
integration with per-stage parameters, metrics, and artifacts logging.

Usage:
    from mlops.mlflow_tracking import configure_mlflow, log_dataset_build, log_vector_extraction
    configure_mlflow(tracking_uri="./mlruns")
    log_dataset_build(num_pairs=100, languages=["hi", "ta"], source_details={"type": "synthetic"})
"""

from __future__ import annotations

import logging
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except Exception:
    mlflow = None
    MlflowClient = None

logger = logging.getLogger(__name__)


def configure_mlflow(
    tracking_uri: str = "./mlruns",
    experiment_name: str = "SafeSteer-IN",
    backend_store_uri: Optional[str] = None,
) -> None:
    """
    Configure MLflow tracking backend.

    Args:
        tracking_uri: Local directory or remote server URI for MLflow tracking (default: ./mlruns)
        experiment_name: Experiment name to group related runs (default: SafeSteer-IN)
        backend_store_uri: Optional backend storage URI for MLflow backend store

    Returns:
        None

    Raises:
        RuntimeError: If MLflow is not installed
    """
    if mlflow is None:
        raise RuntimeError("MLflow not installed. Install via: pip install mlflow")

    try:
        mlflow.set_tracking_uri(tracking_uri)

        # Try to get existing experiment, create if not found
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)

        mlflow.set_experiment(experiment_name)
        logger.info(
            f"MLflow configured | tracking_uri={tracking_uri} | exp={experiment_name}"
        )
    except Exception as e:
        logger.error(f"MLflow configuration failed: {e}")
        raise


def log_dataset_build(
    num_pairs: int,
    languages: List[str],
    categories: Optional[List[str]] = None,
    source_details: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Log dataset construction stage.

    Args:
        num_pairs: Total number of (safe, unsafe) contrastive pairs created
        languages: List of language codes (e.g., ["hi", "ta", "bn"])
        categories: List of harm categories included (defaults to all 8)
        source_details: Dict with keys like {"synthetic": 150, "hf_benchmark": 50}

    Returns:
        str: MLflow run ID
    """
    if mlflow is None:
        logger.warning("MLflow not available, skipping dataset build logging")
        return ""

    try:
        with mlflow.start_run(run_name="01_build_dataset"):
            # Log parameters
            mlflow.log_param("total_pairs", num_pairs)
            mlflow.log_param("num_languages", len(languages))
            mlflow.log_param("languages", ",".join(languages))
            if categories:
                mlflow.log_param("num_categories", len(categories))
                mlflow.log_param("categories", ",".join(categories))

            # Log metrics
            mlflow.log_metric("pairs_per_language", num_pairs // max(len(languages), 1))

            # Log source breakdown
            if source_details:
                for source, count in source_details.items():
                    mlflow.log_metric(f"pairs_from_{source}", count)

            run_id = mlflow.active_run().info.run_id
            logger.info(f"Dataset build logged | run_id={run_id} | pairs={num_pairs}")
            return run_id
    except Exception as e:
        logger.error(f"Failed to log dataset build: {e}")
        return ""


def log_vector_extraction(
    model_key: str,
    language: str,
    category: str,
    layer_idx: int,
    vector_norm: float,
    num_pairs: int,
    run_name: str = "02_extract_vectors",
) -> str:
    """
    Log steering vector extraction for a single (model, language, category, layer).

    Args:
        model_key: Model identifier (e.g., "sarvam-1", "openhathi-base")
        language: Language code (e.g., "hi", "ta")
        category: Harm category key (e.g., "communal_hate")
        layer_idx: Layer index where vector was extracted
        vector_norm: L2 norm of extracted steering vector
        num_pairs: Number of contrastive pairs used to compute vector
        run_name: MLflow run name (shared across all layer extractions for same (model, lang, cat))

    Returns:
        str: MLflow run ID
    """
    if mlflow is None:
        logger.warning("MLflow not available, skipping vector extraction logging")
        return ""

    try:
        nested = mlflow.active_run() is not None
        # Nested run for each layer when a parent run exists
        with mlflow.start_run(run_name=run_name, nested=nested):
            mlflow.log_param("model_key", model_key)
            mlflow.log_param("language", language)
            mlflow.log_param("category", category)
            mlflow.log_param("layer_idx", layer_idx)
            mlflow.log_param("num_pairs_used", num_pairs)

            mlflow.log_metric("vector_norm_l2", vector_norm)
            mlflow.log_metric("layer_idx", float(layer_idx))

            run_id = mlflow.active_run().info.run_id
            logger.info(
                f"Vector extracted | run_id={run_id} | {model_key}:{language}:{category}:L{layer_idx} | norm={vector_norm:.4f}"
            )
            return run_id
    except Exception as e:
        logger.error(f"Failed to log vector extraction: {e}")
        return ""


def log_classifier_training(
    epochs: int,
    final_train_acc: float,
    final_val_acc: float,
    final_train_loss: Optional[float] = None,
    final_val_loss: Optional[float] = None,
    num_labels: int = 8,
) -> str:
    """
    Log multi-label risk classifier training stage.

    Args:
        epochs: Number of training epochs completed
        final_train_acc: Final training accuracy (0-1)
        final_val_acc: Final validation accuracy (0-1)
        final_train_loss: Final training loss (optional)
        final_val_loss: Final validation loss (optional)
        num_labels: Number of classification labels (default: 8 categories)

    Returns:
        str: MLflow run ID
    """
    if mlflow is None:
        logger.warning("MLflow not available, skipping classifier training logging")
        return ""

    try:
        with mlflow.start_run(run_name="03_train_classifier"):
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("num_labels", num_labels)

            mlflow.log_metric("final_train_accuracy", final_train_acc)
            mlflow.log_metric("final_val_accuracy", final_val_acc)

            if final_train_loss is not None:
                mlflow.log_metric("final_train_loss", final_train_loss)
            if final_val_loss is not None:
                mlflow.log_metric("final_val_loss", final_val_loss)

            # Compute improvement metric
            accuracy_improvement = final_val_acc - final_train_acc
            mlflow.log_metric("val_train_accuracy_gap", accuracy_improvement)

            run_id = mlflow.active_run().info.run_id
            logger.info(
                f"Classifier training logged | run_id={run_id} | train_acc={final_train_acc:.4f} | val_acc={final_val_acc:.4f}"
            )
            return run_id
    except Exception as e:
        logger.error(f"Failed to log classifier training: {e}")
        return ""


def log_alpha_values(
    language: str,
    category: str,
    best_alpha: float,
    metrics_dict: Dict[str, float],
    run_name: str = "04_calibrate_alpha",
) -> str:
    """
    Log alpha calibration results for a single (language, category) slice.

    Per-slice alpha logging for fine-grained tracking of steering strength optimization.

    Args:
        language: Language code (e.g., "hi", "ta")
        category: Harm category key
        best_alpha: Optimal steering strength for this slice
        metrics_dict: Dict with keys like {"safety_rate": 0.95, "perplexity_increase": 1.02}
        run_name: MLflow run name (shared across all slices for same calibration run)

    Returns:
        str: MLflow run ID
    """
    if mlflow is None:
        logger.warning("MLflow not available, skipping alpha calibration logging")
        return ""

    try:
        nested = mlflow.active_run() is not None
        with mlflow.start_run(run_name=run_name, nested=nested):
            mlflow.log_param("language", language)
            mlflow.log_param("category", category)
            mlflow.log_param("best_alpha", best_alpha)

            # Log all metrics from calibration
            for metric_name, metric_value in metrics_dict.items():
                mlflow.log_metric(f"{language}_{category}_{metric_name}", metric_value)

            run_id = mlflow.active_run().info.run_id
            logger.info(
                f"Alpha calibrated | run_id={run_id} | {language}:{category} | alpha={best_alpha}"
            )
            return run_id
    except Exception as e:
        logger.error(f"Failed to log alpha values: {e}")
        return ""


def log_llm_judge_eval(
    language: str,
    category: str,
    baseline_score: float,
    steered_score: float,
    run_name: str = "05_evaluate",
    additional_metrics: Optional[Dict[str, float]] = None,
) -> str:
    """
    Log LLM-as-judge safety scoring results for (language, category) slice.

    Per-slice evaluation logging including safety improvement metrics.

    Args:
        language: Language code (e.g., "hi", "ta")
        category: Harm category key
        baseline_score: Safety score of baseline (raw) model output (0-1)
        steered_score: Safety score of steered output (0-1)
        run_name: MLflow run name (shared across all slices)
        additional_metrics: Optional dict with keys like {"ppl_increase": 1.02, "latency_ms": 450}

    Returns:
        str: MLflow run ID
    """
    if mlflow is None:
        logger.warning("MLflow not available, skipping LLM judge evaluation logging")
        return ""

    try:
        nested = mlflow.active_run() is not None
        with mlflow.start_run(run_name=run_name, nested=nested):
            mlflow.log_param("language", language)
            mlflow.log_param("category", category)

            mlflow.log_metric(
                f"{language}_{category}_baseline_safety_score", baseline_score
            )
            mlflow.log_metric(
                f"{language}_{category}_steered_safety_score", steered_score
            )

            # Compute safety improvement
            improvement = steered_score - baseline_score
            mlflow.log_metric(f"{language}_{category}_safety_improvement", improvement)

            # Log additional metrics if provided
            if additional_metrics:
                for metric_name, metric_value in additional_metrics.items():
                    mlflow.log_metric(
                        f"{language}_{category}_{metric_name}", metric_value
                    )

            run_id = mlflow.active_run().info.run_id
            logger.info(
                f"LLM judge eval logged | run_id={run_id} | {language}:{category} | improvement={improvement:.4f}"
            )
            return run_id
    except Exception as e:
        logger.error(f"Failed to log LLM judge evaluation: {e}")
        return ""


def log_mlflow_model_to_registry(
    model_name: str,
    model_path: str | Path,
    run_id: str,
    description: str = "",
    stage: Optional[str] = "Staging",
) -> Optional[str]:
    """
    Register trained model in MLflow Model Registry for versioning.

    Args:
        model_name: Name for model in registry (e.g., "indic_risk_classifier")
        model_path: Local path to model artifacts
        run_id: MLflow run ID where model was trained
        description: Optional model description

    Returns:
        Optional[str]: Model version string or None if registration failed
    """
    if mlflow is None:
        logger.warning("MLflow not available, skipping model registration")
        return None

    try:
        client = MlflowClient() if MlflowClient else None
        if not client:
            logger.warning("MlflowClient unavailable, cannot register model")
            return None

        model_uri = f"runs:/{run_id}/model"

        if model_path:
            try:
                client.log_artifacts(run_id, str(model_path), artifact_path="model")
            except Exception as e:
                logger.warning("Failed to log model artifacts: %s", e)

        result = mlflow.register_model(model_uri, model_name)
        version = result.version

        if description:
            try:
                client.update_model_version(
                    name=model_name,
                    version=version,
                    description=description,
                )
            except Exception as e:
                logger.warning("Failed to update model description: %s", e)

        if stage:
            try:
                client.transition_model_version_stage(
                    name=model_name,
                    version=version,
                    stage=stage,
                )
            except Exception as e:
                logger.warning("Failed to transition model stage: %s", e)

        logger.info(
            "Model registered | name=%s | version=%s | stage=%s",
            model_name,
            version,
            stage or "none",
        )
        return str(version)
    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        return None


def log_evaluation_run(
    model_key: str,
    slices: Dict[str, List[str]],
    metrics: Dict[str, float],
    run_name: str = "05_evaluate",
) -> str:
    """
    Log complete evaluation run summary (all slices, aggregate metrics).

    Args:
        model_key: Model evaluated (e.g., "sarvam-1")
        slices: Dict like {"languages": ["hi", "ta"], "categories": ["communal_hate", ...]}
        metrics: Aggregate metrics dict like {"overall_harm_reduction": 0.35, "avg_safety_score": 0.85}
        run_name: MLflow run name

    Returns:
        str: MLflow run ID
    """
    if mlflow is None:
        logger.warning("MLflow not available, skipping evaluation run logging")
        return ""

    try:
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model_key", model_key)
            mlflow.log_param("num_languages", len(slices.get("languages", [])))
            mlflow.log_param("num_categories", len(slices.get("categories", [])))

            # Log all metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"overall_{metric_name}", metric_value)

            # Log slices as artifact JSON
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                delete=False,
                encoding="utf-8",
            ) as tmp:
                tmp.write(json.dumps(slices, indent=2))
                slices_path = Path(tmp.name)
            mlflow.log_artifact(str(slices_path))
            try:
                slices_path.unlink(missing_ok=True)
            except Exception:
                pass

            run_id = mlflow.active_run().info.run_id
            logger.info(f"Evaluation run logged | run_id={run_id} | model={model_key}")
            return run_id
    except Exception as e:
        logger.error(f"Failed to log evaluation run: {e}")
        return ""


if __name__ == "__main__":
    # Test MLflow configuration
    logging.basicConfig(level=logging.INFO)

    configure_mlflow(tracking_uri="./mlruns", experiment_name="SafeSteer-IN-Test")

    # Test logging functions
    dataset_run = log_dataset_build(
        num_pairs=100,
        languages=["hi", "ta"],
        categories=["communal_hate", "caste_discrimination"],
        source_details={"synthetic": 80, "hf_benchmark": 20},
    )
    print(f"✓ Dataset build logged: {dataset_run}")

    vector_run = log_vector_extraction(
        model_key="sarvam-1",
        language="hi",
        category="communal_hate",
        layer_idx=12,
        vector_norm=0.456,
        num_pairs=100,
    )
    print(f"✓ Vector extraction logged: {vector_run}")
