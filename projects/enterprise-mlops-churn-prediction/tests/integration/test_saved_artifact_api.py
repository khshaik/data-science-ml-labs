"""Exercise FastAPI with the real champion and persisted preprocessing artifacts."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def saved_artifact_client():
    """Start the application lifespan so the real saved artifacts are loaded."""
    from src.serving.api import app

    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="module")
def sample_request():
    return json.loads(Path("sample_request.json").read_text())


def test_saved_artifact_contract_exists():
    """The evaluated serving bundle must contain every startup dependency."""
    champion = json.loads(Path("models/current_best.json").read_text())

    assert Path(champion["model_path"]).is_file()
    assert Path("artifacts/preprocessor.pkl").is_file()
    assert Path("artifacts/feature_threshold.json").is_file()
    assert Path("artifacts/eval/model_comparison.json").is_file()


def test_health_reports_loaded_champion(saved_artifact_client):
    response = saved_artifact_client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True

    champion = json.loads(Path("models/current_best.json").read_text())
    assert body["model_version"] == champion["model_version"]


def test_real_prediction_uses_champion(saved_artifact_client, sample_request):
    response = saved_artifact_client.post("/predict", json=sample_request)

    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in {"Yes", "No"}
    assert body["risk_level"] in {"Low", "Medium", "High"}
    assert body["latency_ms"] > 0

    champion = json.loads(Path("models/current_best.json").read_text())
    assert body["model_version"] == champion["model_version"]


def test_metrics_include_prediction_observability(saved_artifact_client):
    response = saved_artifact_client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "predictions_total" in response.text
    assert "prediction_latency_seconds" in response.text
