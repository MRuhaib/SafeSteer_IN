# SafeSteer-IN Architecture

SafeSteer-IN is a two-pipeline safety steering system for Indic language models. The offline pipeline builds reusable safety artifacts once, while the runtime pipeline consumes those artifacts for inference-time steering without updating model weights.

## System View

```text
Offline Pipeline
Seed templates + synthetic generation
        ↓
Contrastive dataset (train/val/test)
        ↓
Steering vector extraction per (language, category, layer)
        ↓
Classifier training for language + harm-category routing
        ↓
Alpha calibration per slice
        ↓
Evaluation + MLflow artifacts + exports

Runtime Pipeline
User prompt
        ↓
FastAPI / Gradio entry point
        ↓
Risk classifier (language + category + risk)
        ↓
Vector lookup with slice fallback
        ↓
Forward-hook steering on the target LLM
        ↓
Raw output + steered output + safety scores
```

## Main Blocks

### 1. Data and taxonomy

The `src/data/` package defines the safety taxonomy, seed templates, and dataset construction flow. The project covers 9 languages and 8 harm categories, giving 72 intended slices. The synthetic and contrastive dataset is the foundation for both vector extraction and classifier training.

### 2. Steering artifact generation

The `src/steering/` package computes steering directions from safe/unsafe response pairs. For each slice and layer, the vector is derived as the difference between unsafe and safe activation means. The resulting `.pt` files are stored in `src/steering_vectors/` and are consumed by the runtime steering engine.

### 3. Classifier routing

The `src/classifier/` package provides prompt routing. It predicts the most likely language and harm category and returns a risk score. The runtime uses this prediction to decide whether steering should be applied and which slice to select.

### 4. Inference engine

The `src/engine/` package contains the orchestration logic. `src/engine/pipeline.py` loads the model, classifier, and steering engine, then coordinates baseline generation, steered generation, and optional safety scoring.

### 5. Serving layer

The repository exposes two user-facing entry points:

- `app.py` for the Gradio interface.
- `api.py` for the FastAPI service.

The frontend is intentionally kept loose-coupled: the UI should call the backend over REST rather than importing model code directly.

### 6. MLOps and observability

The new foundation modules introduced in this update provide the rubric-facing MLOps layer:

- `src/mlops/mlflow_tracking.py` for MLflow logging and registry integration.
- `src/utils/consolidate_metrics.py` for shared metric computation.
- `src/evaluation/llm_judge.py` for heuristic judge scoring.
- `notifications/smtp_handler.py` for completion and failure notifications.

The next implementation block adds monitoring endpoints and Prometheus/Grafana configuration to complete the observability path.

## Request Flow

1. The user submits a prompt in Gradio or through the FastAPI endpoint.
2. The pipeline classifies the prompt to infer language, category, and risk.
3. The engine generates a baseline response.
4. If steering is required, the engine selects the best slice vector and applies activation steering with alpha.
5. The pipeline returns raw output, steered output, risk metadata, and optional safety scores.

## Slice Availability Rule

The runtime should only treat a slice as available when at least one `layer*.pt` file exists for that `(language, category)` pair. Metadata files alone are not sufficient. This keeps demo routing stable and avoids empty slice selection.

## Design Principles

- Offline artifacts are generated once and reused at runtime.
- Steering is applied at inference time through forward hooks, not weight updates.
- Slice fallback is required so that missing vectors degrade gracefully.
- Experiment tracking must preserve the link between code version, MLflow run, and generated artifacts.

## Related Files

- [README.md](../README.md)
- [DOCUMENTATION.md](../DOCUMENTATION.md)
- [src/engine/pipeline.py](../src/engine/pipeline.py)
- [src/engine/steering_engine.py](../src/engine/steering_engine.py)
- [api.py](../api.py)
- [app.py](../app.py)
