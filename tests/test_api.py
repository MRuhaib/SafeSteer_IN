from __future__ import annotations

from dataclasses import dataclass
from fastapi.testclient import TestClient

import api


@dataclass
class DummyResult:
    prompt: str
    detected_language: str = "hi"
    detected_category: str = "communal_religious_hate"
    risk_score: float = 0.5
    steering_applied: bool = True
    alpha_used: float = 12.0
    raw_output: str = "raw-output"
    steered_output: str = "steered-output"
    azure_score_raw: dict | None = None
    azure_score_steered: dict | None = None
    latency_ms: float = 12.3
    steering_language_used: str = "hi"
    steering_category_used: str = "communal_religious_hate"


class DummyEngine:
    def generate(self, prompt: str, max_new_tokens: int = 32) -> str:
        return f"generated:{prompt}"


class DummyPipeline:
    def __init__(self, *, fallback: bool = False, raise_on_run: bool = False):
        self._fallback = fallback
        self._raise_on_run = raise_on_run
        self._engine = DummyEngine()

    def available_slices(self):
        return [("hi", "communal_religious_hate")]

    def run(self, prompt: str, **_kwargs):
        if self._raise_on_run:
            raise RuntimeError("boom")
        result = DummyResult(prompt=prompt)
        if self._fallback:
            result.steering_category_used = "caste_discrimination"
        return result

    @property
    def engine(self):
        return self._engine


class BrokenPipeline(DummyPipeline):
    def available_slices(self):
        raise RuntimeError("no slices")


def _client(monkeypatch, pipeline) -> TestClient:
    monkeypatch.setattr(api, "_get_pipeline", lambda: pipeline)
    return TestClient(api.app)


def test_health_ok():
    client = TestClient(api.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_ok(monkeypatch):
    client = _client(monkeypatch, DummyPipeline())
    response = client.get("/ready")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["loaded_slices"] == 1


def test_ready_error(monkeypatch):
    client = _client(monkeypatch, BrokenPipeline())
    response = client.get("/ready")
    assert response.status_code == 503


def test_metrics_ok():
    client = TestClient(api.app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "safesteer_requests_total" in response.text


def test_slices_ok(monkeypatch):
    client = _client(monkeypatch, DummyPipeline())
    response = client.get("/slices")
    assert response.status_code == 200
    assert response.json()["slices"] == [
        {"language": "hi", "category": "communal_religious_hate"}
    ]


def test_slices_error(monkeypatch):
    client = _client(monkeypatch, BrokenPipeline())
    response = client.get("/slices")
    assert response.status_code == 500


def test_generate_ok(monkeypatch):
    client = _client(monkeypatch, DummyPipeline())
    response = client.post("/generate", json={"prompt": "hello"})
    assert response.status_code == 200
    assert response.json()["output"] == "generated:hello"


def test_generate_error(monkeypatch):
    pipeline = DummyPipeline()
    pipeline._engine.generate = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("bad")
    )
    client = _client(monkeypatch, pipeline)
    response = client.post("/generate", json={"prompt": "hello"})
    assert response.status_code == 500


def test_steer_ok(monkeypatch):
    client = _client(monkeypatch, DummyPipeline())
    response = client.post("/steer", json={"prompt": "test"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["prompt"] == "test"
    assert payload["steering_applied"] is True
    assert payload["steered_output"] == "steered-output"


def test_steer_fallback(monkeypatch):
    client = _client(monkeypatch, DummyPipeline(fallback=True))
    response = client.post("/steer", json={"prompt": "test"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["detected_category"] == "communal_religious_hate"


def test_steer_error(monkeypatch):
    client = _client(monkeypatch, DummyPipeline(raise_on_run=True))
    response = client.post("/steer", json={"prompt": "test"})
    assert response.status_code == 500
