# SafeSteer-IN Low-Level Design

## Scope

This document defines the runtime interfaces and the supporting internal flow for SafeSteer-IN. It focuses on the API layer, pipeline inputs and outputs, and the main data contracts used by the implementation.

## API Endpoints

### `GET /health`

- Purpose: liveness check.
- Request: none.
- Response:

```json
{ "status": "ok" }
```

### `GET /ready`

- Purpose: readiness check (pipeline load + slice availability).
- Request: none.
- Response:

```json
{ "status": "ready", "loaded_slices": 72 }
```

### `GET /slices`

- Purpose: list available steering slices.
- Request: none.
- Response:

```json
{
  "slices": [
    { "language": "hi", "category": "communal_religious_hate" }
  ]
}
```

### `POST /generate`

- Purpose: baseline generation without steering.
- Request body:

```json
{
  "prompt": "...",
  "max_new_tokens": 128
}
```

- Response body:

```json
{
  "prompt": "...",
  "output": "..."
}
```

### `POST /steer`

- Purpose: full classify → steer → score flow.
- Request body:

```json
{
  "prompt": "...",
  "language": "hi",
  "category": "communal_religious_hate",
  "alpha": 12.0,
  "max_new_tokens": 128,
  "always_steer": true
}
```

- Response body:

```json
{
  "prompt": "...",
  "detected_language": "hi",
  "detected_category": "communal_religious_hate",
  "risk_score": 0.82,
  "steering_applied": true,
  "alpha_used": 12.0,
  "raw_output": "...",
  "steered_output": "...",
  "azure_score_raw": null,
  "azure_score_steered": null,
  "latency_ms": 1234.5
}
```

## Internal Data Contracts

### `SteerRequest`

- `prompt`: required string.
- `language`: optional override for language code.
- `category`: optional override for harm category key.
- `alpha`: optional steering strength.
- `max_new_tokens`: integer, defaults to `MAX_NEW_TOKENS`.
- `always_steer`: whether steering should be forced.

### `SteerResponse`

- `prompt`: original input.
- `detected_language`: language returned by the classifier or override.
- `detected_category`: category returned by the classifier or override.
- `risk_score`: classifier confidence proxy.
- `steering_applied`: whether a steering vector was used.
- `alpha_used`: the alpha value actually applied.
- `raw_output`: baseline model response.
- `steered_output`: response after steering.
- `azure_score_raw` and `azure_score_steered`: optional safety scores.
- `latency_ms`: end-to-end elapsed time.

## Runtime Flow

1. `api.py` accepts the request and lazily loads `SafeSteerPipeline`.
2. `src/engine/pipeline.py` loads the model, tokenizer, steering engine, and classifier.
3. `src/classifier/inference.py` predicts language, category, and risk score.
4. `src/engine/steering_engine.py` performs baseline and steered generation.
5. `src/evaluation/metrics.py` and `src/evaluation/llm_judge.py` provide scoring logic.
6. MLflow logging records the stage, metrics, and artifacts.

## Artifact Contracts

### Steering vectors

- Location: `src/steering_vectors/<model>/<language>/<category>/layer<N>.pt`
- Type: PyTorch tensor
- Constraint: at least one layer file must exist for the slice to be considered available.

### Classifier checkpoint

- Location: `src/models/indic_risk_classifier/`
- Files: `classifier.pt`, `label_maps.json`, tokenizer files.

### Evaluation outputs

- Location: `src/evaluation/results/`
- Outputs: CSV, JSON, charts, and slice summaries.

## Validation Rules

- API responses must remain JSON serializable.
- Layer fallback must prefer nearby layers when a forced layer is unavailable.
- Missing vectors must degrade to baseline generation, not a crash.
- The UI must not import model internals directly; it should call the REST API.

## Related Files

- [api.py](../api.py)
- [app.py](../app.py)
- [src/engine/pipeline.py](../src/engine/pipeline.py)
- [src/engine/steering_engine.py](../src/engine/steering_engine.py)
- [src/classifier/inference.py](../src/classifier/inference.py)
- [src/evaluation/evaluate.py](../src/evaluation/evaluate.py)
