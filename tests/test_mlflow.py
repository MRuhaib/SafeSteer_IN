from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mlops import mlflow_tracking


def test_configure_mlflow_raises_when_missing(monkeypatch):
    monkeypatch.setattr(mlflow_tracking, "mlflow", None)
    with pytest.raises(RuntimeError):
        mlflow_tracking.configure_mlflow()


def test_log_dataset_build_skips_when_missing(monkeypatch):
    monkeypatch.setattr(mlflow_tracking, "mlflow", None)
    run_id = mlflow_tracking.log_dataset_build(
        num_pairs=10,
        languages=["hi"],
        categories=["communal_religious_hate"],
        source_details={"synthetic": 10},
    )
    assert run_id == ""


def test_log_vector_extraction_skips_when_missing(monkeypatch):
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
    monkeypatch.setattr(mlflow_tracking, "mlflow", None)
    run_id = mlflow_tracking.log_llm_judge_eval(
        language="hi",
        category="communal_religious_hate",
        baseline_score=0.5,
        steered_score=0.7,
    )
    assert run_id == ""


def test_log_model_registry_skips_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(mlflow_tracking, "mlflow", None)
    version = mlflow_tracking.log_mlflow_model_to_registry(
        model_name="indic_risk_classifier",
        model_path=tmp_path,
        run_id="dummy",
    )
    assert version is None
