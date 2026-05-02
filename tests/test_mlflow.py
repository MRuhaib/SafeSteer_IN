from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _load_mlflow_tracking():
    # Local Windows envs can hard-crash during mlflow import because of optional
    # binary deps; CI runs on Ubuntu, so we skip here for stability.
    if sys.platform.startswith("win"):
        pytest.skip("Skipping MLflow import test on Windows")
    return importlib.import_module("src.mlops.mlflow_tracking")


def test_configure_mlflow_raises_when_missing(monkeypatch):
    mlflow_tracking = _load_mlflow_tracking()
    monkeypatch.setattr(mlflow_tracking, "mlflow", None)
    with pytest.raises(RuntimeError):
        mlflow_tracking.configure_mlflow()


def test_log_dataset_build_skips_when_missing(monkeypatch):
    mlflow_tracking = _load_mlflow_tracking()
    monkeypatch.setattr(mlflow_tracking, "mlflow", None)
    run_id = mlflow_tracking.log_dataset_build(
        num_pairs=10,
        languages=["hi"],
        categories=["communal_religious_hate"],
        source_details={"synthetic": 10},
    )
    assert run_id == ""


def test_log_vector_extraction_skips_when_missing(monkeypatch):
    mlflow_tracking = _load_mlflow_tracking()
    monkeypatch.setattr(mlflow_tracking, "mlflow", None)
    run_id = mlflow_tracking.log_vector_extraction(
        model_key="sarvam-1",
        language="hi",
        category="communal_religious_hate",
        layer_idx=12,
        vector_norm=1.0,
        num_pairs=10,
    )
    assert run_id == ""


def test_log_llm_judge_eval_skips_when_missing(monkeypatch):
    mlflow_tracking = _load_mlflow_tracking()
    monkeypatch.setattr(mlflow_tracking, "mlflow", None)
    run_id = mlflow_tracking.log_llm_judge_eval(
        language="hi",
        category="communal_religious_hate",
        baseline_score=0.5,
        steered_score=0.7,
    )
    assert run_id == ""


def test_log_model_registry_skips_when_missing(monkeypatch, tmp_path):
    mlflow_tracking = _load_mlflow_tracking()
    monkeypatch.setattr(mlflow_tracking, "mlflow", None)
    version = mlflow_tracking.log_mlflow_model_to_registry(
        model_name="indic_risk_classifier",
        model_path=tmp_path,
        run_id="dummy",
    )
    assert version is None
