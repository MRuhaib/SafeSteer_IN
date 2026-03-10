"""
SafeSteer-IN  ·  Azure ML Workspace Setup
===========================================
Initialises the Azure ML Workspace and creates required resources:
    - Workspace (if not exists)
    - Compute clusters  (GPU + CPU)
    - Experiment

Uses the **azure-ai-ml v2 SDK** style where possible, falling back to the
v1 azureml-core SDK for compatibility with the student pack.

Run once at the start of the project:
    python -m azure_ml.setup
"""

from __future__ import annotations

import logging

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    AZURE_ML_EXPERIMENT,
    AZURE_ML_RESOURCE_GROUP,
    AZURE_ML_SUBSCRIPTION_ID,
    AZURE_ML_WORKSPACE,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
)


def get_workspace():
    """
    Connect to an existing Azure ML Workspace or print instructions to create one.
    """
    try:
        from azureml.core import Workspace
        from azureml.core.authentication import InteractiveLoginAuthentication

        if not AZURE_ML_SUBSCRIPTION_ID:
            logger.error(
                "AZURE_ML_SUBSCRIPTION_ID not set. "
                "Add it to your .env file or set the environment variable."
            )
            return None

        logger.info(
            "Connecting to Azure ML Workspace: %s / %s",
            AZURE_ML_RESOURCE_GROUP,
            AZURE_ML_WORKSPACE,
        )

        try:
            ws = Workspace.get(
                name=AZURE_ML_WORKSPACE,
                subscription_id=AZURE_ML_SUBSCRIPTION_ID,
                resource_group=AZURE_ML_RESOURCE_GROUP,
            )
            logger.info("Connected to workspace: %s", ws.name)
            return ws
        except Exception:
            logger.warning(
                "Workspace not found. You can create it in the Azure Portal:\n"
                "  1. Go to https://portal.azure.com\n"
                "  2. Search for 'Machine Learning'\n"
                "  3. Create a new workspace named '%s' in resource group '%s'\n"
                "  4. Re-run this script.\n",
                AZURE_ML_WORKSPACE,
                AZURE_ML_RESOURCE_GROUP,
            )
            return None

    except ImportError:
        logger.error("azureml-core not installed. Run: pip install azureml-core")
        return None


def create_compute_clusters(ws):
    """Create GPU and CPU compute clusters if they don't exist."""
    from azureml.core.compute import AmlCompute, ComputeTarget
    from azureml.core.compute_target import ComputeTargetException

    # GPU cluster for extraction
    gpu_name = "gpu-extraction"
    try:
        gpu_target = ComputeTarget(workspace=ws, name=gpu_name)
        logger.info("GPU cluster '%s' already exists", gpu_name)
    except ComputeTargetException:
        logger.info("Creating GPU cluster '%s' …", gpu_name)
        compute_config = AmlCompute.provisioning_configuration(
            vm_size="Standard_NC6s_v3",  # 1× V100 16GB
            min_nodes=0,
            max_nodes=1,
            idle_seconds_before_scaledown=600,
        )
        gpu_target = ComputeTarget.create(ws, gpu_name, compute_config)
        gpu_target.wait_for_completion(show_output=True)
        logger.info("GPU cluster created: %s", gpu_name)

    # CPU cluster for inference API
    cpu_name = "cpu-inference"
    try:
        cpu_target = ComputeTarget(workspace=ws, name=cpu_name)
        logger.info("CPU cluster '%s' already exists", cpu_name)
    except ComputeTargetException:
        logger.info("Creating CPU cluster '%s' …", cpu_name)
        compute_config = AmlCompute.provisioning_configuration(
            vm_size="Standard_D4s_v3",
            min_nodes=0,
            max_nodes=1,
            idle_seconds_before_scaledown=600,
        )
        cpu_target = ComputeTarget.create(ws, cpu_name, compute_config)
        cpu_target.wait_for_completion(show_output=True)
        logger.info("CPU cluster created: %s", cpu_name)


def create_experiment(ws):
    """Create or get the MLflow experiment."""
    from azureml.core import Experiment

    exp = Experiment(workspace=ws, name=AZURE_ML_EXPERIMENT)
    logger.info("Experiment: %s", exp.name)
    return exp


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def setup_all():
    """Run the full Azure setup pipeline."""
    ws = get_workspace()
    if ws is None:
        logger.info("Azure ML setup incomplete — workspace not available.")
        return None

    create_compute_clusters(ws)
    exp = create_experiment(ws)
    logger.info("Azure ML setup complete.")
    return ws


if __name__ == "__main__":
    setup_all()
