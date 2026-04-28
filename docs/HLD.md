# SafeSteer-IN High-Level Design

## Problem Statement

SafeSteer-IN addresses the need for controllable safety intervention in Indic language models without retraining the base model. The design must support multilingual routing, slice-specific steering, offline calibration, runtime serving, and rubric-friendly MLOps instrumentation.

## High-Level Goals

- Provide inference-time safety steering for 9 Indic language settings.
- Keep the runtime stateless and light enough for local deployment.
- Generate reusable artifacts offline, including vectors, classifier checkpoints, and calibration results.
- Expose results through both a visual demo and a REST API.
- Support experiment tracking, monitoring, and notifications for grading visibility.

## Design Choices

| Area | Choice | Rationale |
|---|---|---|
| Runtime steering | Forward hooks in PyTorch | Avoids weight modification and keeps inference reversible. |
| Routing | Multi-label risk classifier | Selects language/category slice before steering. |
| Artifact storage | Filesystem-based vectors and checkpoints | Simple local deployment and easy evaluation. |
| Experiment tracking | MLflow | Captures parameters, metrics, and artifacts per stage. |
| Serving | FastAPI + Gradio | Clean REST API plus non-technical UI. |
| Notifications | SMTP email handler | Provides stage completion and failure visibility. |
| Metrics reuse | Shared utility module | Avoids duplicated perplexity and scoring logic. |

## Pipeline Summary

### Offline pipeline

1. Build the contrastive dataset.
2. Extract steering vectors from safe/unsafe activations.
3. Train the classifier used for runtime routing.
4. Calibrate alpha per slice.
5. Evaluate and log outputs, metrics, and artifacts.

### Runtime pipeline

1. Accept a prompt from UI or API.
2. Predict language, category, and risk.
3. Generate the baseline output.
4. Resolve the best slice vector with fallback rules.
5. Apply steering with alpha through a forward hook.
6. Return both outputs with safety metadata.

## Fallback Strategy

The design must not fail just because one slice is incomplete. The fallback order used by the steering engine is:

1. Same language, same category.
2. Same category in Hindi.
3. Same category in any language.
4. Any available slice.

This is consistent with the repository note that some slice folders may contain metadata without layer vectors.

## Monitoring and MLOps Scope

The grading rubric requires more than model code. The architecture therefore includes:

- MLflow for experiment logging and registry-ready model tracking.
- SMTP notifications for pipeline completion and failure alerts.
- Prometheus/Grafana for runtime monitoring.
- Docker Compose for service separation and environment parity.
- A custom orchestration runner plus Airflow-compatible DAG code.

## Non-Functional Requirements

- Loose coupling between frontend and backend.
- Reproducible local execution.
- Clear logging and exception handling.
- Fast feedback during demo runs.
- Graceful degradation when a vector or checkpoint is missing.

## Current Implementation Notes

The repository already exposes the main runtime entry points in `api.py` and `app.py`. The new documentation aligns those entry points with the offline artifacts stored under `data/`, `steering_vectors/`, `models/`, and `evaluation/results/`.

## Related Files

- [README.md](../README.md)
- [DOCUMENTATION.md](../DOCUMENTATION.md)
- [src/engine/pipeline.py](../src/engine/pipeline.py)
- [src/classifier/inference.py](../src/classifier/inference.py)
- [src/evaluation/evaluate.py](../src/evaluation/evaluate.py)
