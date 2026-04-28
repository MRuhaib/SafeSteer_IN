"""
Airflow DAG for SafeSteer-IN Offline Pipeline
=============================================
Sequentially runs dataset build, vector extraction, classifier training,
alpha calibration, and evaluation.

To use:
  - Place this file in your Airflow DAGs folder or set AIRFLOW__CORE__DAGS_FOLDER
  - Ensure PYTHON_BIN env var points to the project venv python if needed
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_BIN = os.getenv("PYTHON_BIN", "python")


def _script_cmd(script_name: str) -> str:
    script_path = PROJECT_ROOT / "src" / "scripts" / script_name
    return f'cd "{PROJECT_ROOT}" && {PYTHON_BIN} "{script_path}"'


def _default_args() -> dict:
    return {
        "owner": "safesteer",
        "depends_on_past": False,
        "retries": 0,
    }


with DAG(
    dag_id="safesteer_offline_pipeline",
    description="SafeSteer-IN offline artifact generation",
    default_args=_default_args(),
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["safesteer", "mlops"],
) as dag:
    build_dataset = BashOperator(
        task_id="build_dataset",
        bash_command=_script_cmd("01_build_dataset.py"),
    )

    extract_vectors = BashOperator(
        task_id="extract_vectors",
        bash_command=_script_cmd("02_extract_vectors.py"),
    )

    train_classifier = BashOperator(
        task_id="train_classifier",
        bash_command=_script_cmd("03_train_classifier.py"),
    )

    calibrate_alpha = BashOperator(
        task_id="calibrate_alpha",
        bash_command=_script_cmd("04_calibrate_alpha.py"),
    )

    evaluate = BashOperator(
        task_id="evaluate",
        bash_command=_script_cmd("05_evaluate.py"),
    )

    build_dataset >> extract_vectors >> train_classifier >> calibrate_alpha >> evaluate
