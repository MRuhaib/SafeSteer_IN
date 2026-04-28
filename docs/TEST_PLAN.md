# SafeSteer-IN Test Plan

## Purpose

This test plan covers the offline pipeline, runtime serving, observability hooks, and the documentation-visible acceptance criteria required by the rubric.

## Test Strategy

The fastest useful validation path is a set of small, reproducible checks:

1. Verify package imports and entry points.
2. Run the offline steps with small sample sizes.
3. Validate the API endpoints locally.
4. Confirm the metrics and notification modules can be imported.
5. Review generated MLflow artifacts and evaluation outputs.

## Test Categories

### 1. Unit tests

- `tests/test_api.py`
- `tests/test_mlflow.py`
- `tests/TEST_REPORT.txt` (execution summary template to be updated after local run)

### 2. Integration tests

- Dataset build to vector extraction handoff.
- Classifier inference to pipeline routing handoff.
- Steering engine slice lookup with fallback.
- FastAPI request to pipeline response round-trip.

### 3. System tests

- Gradio UI loads and can call the backend.
- `GET /health` returns healthy status.
- `GET /slices` returns only slices with at least one vector file.
- Pipeline produces raw and steered outputs.

## Quick Test Cases

| ID | Test | Expected Result |
|---|---|---|
| T1 | Import `api.py` | Module loads without syntax errors. |
| T2 | Import `src/mlops/mlflow_tracking.py` | MLflow helper functions are available. |
| T3 | Import `notifications/smtp_handler.py` | SMTP notifier class is available. |
| T4 | Call `GET /health` | Returns `{"status": "ok"}`. |
| T5 | Call `GET /ready` | Returns readiness status and loaded slice count. |
| T6 | Call `GET /slices` | Returns a JSON list of available slices. |
| T7 | Run one small steering request | Returns both raw and steered outputs. |
| T8 | Run a tiny evaluation slice | Produces CSV/JSON output and metrics. |
| T9 | Log a mock MLflow run | Run appears in local MLflow UI. |

## Acceptance Criteria

The implementation is acceptable when all of the following are true:

- The docs describe the real project flow and the actual entry points.
- The UI and API remain loosely coupled.
- The offline pipeline can be executed with small sample sizes.
- The runtime can generate a baseline and a steered response.
- MLflow records at least one run with parameters and metrics.
- The documentation identifies the slice fallback rule.
- The documentation gives evaluators a clear route to verify the project.

## Evidence to Capture

- MLflow experiment screenshot.
- API `/health` response.
- API `/slices` response.
- Gradio screen showing raw vs. steered output.
- Final test summary with pass/fail counts.

## Known Risks

- Some slice folders may contain metadata but no vectors.
- SMTP tests require credentials or a dry-run mode.
- GPU-heavy steps may need a small-sample fallback run.
- Monitoring stack requires Prometheus/Grafana services to be started via Docker Compose.

## Related Files

- [README.md](../README.md)
- [DOCUMENTATION.md](../DOCUMENTATION.md)
- [api.py](../api.py)
- [app.py](../app.py)
- [src/engine/pipeline.py](../src/engine/pipeline.py)
