"""Static contract tests for the local Prometheus/Grafana stack."""

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _yaml(path: str):
    return yaml.safe_load((ROOT / path).read_text())


def test_compose_mounts_prometheus_rules_and_grafana_provisioning():
    compose = _yaml("docker/docker-compose.yml")
    prometheus = compose["services"]["prometheus"]
    grafana = compose["services"]["grafana"]
    api = compose["services"]["api"]

    assert api["healthcheck"]["test"][:3] == ["CMD", "python", "-c"]
    assert prometheus["image"] != "prom/prometheus:latest"
    assert "../monitoring/alerts.yml:/etc/prometheus/alerts.yml:ro" in prometheus["volumes"]
    assert grafana["image"] != "grafana/grafana:latest"
    assert any("/etc/grafana/provisioning/datasources:ro" in item for item in grafana["volumes"])
    assert any("/etc/grafana/provisioning/dashboards:ro" in item for item in grafana["volumes"])
    assert any("/var/lib/grafana/dashboards:ro" in item for item in grafana["volumes"])


def test_api_image_has_a_tensorflow_only_serving_dependency_boundary():
    dockerfile = (ROOT / "docker/Dockerfile.api").read_text()
    requirements = (ROOT / "docker/requirements.api.txt").read_text()

    assert "COPY docker/requirements.api.txt" in dockerfile
    assert "tensorflow==2.13.0" in requirements
    assert "fastapi==0.103.1" in requirements
    assert "mlflow" not in requirements
    assert "streamlit" not in requirements
    assert "shap" not in requirements


def test_prometheus_loads_rules_and_uses_request_error_ratio():
    prometheus = _yaml("monitoring/prometheus.yml")
    alerts = _yaml("monitoring/alerts.yml")

    assert prometheus["rule_files"] == ["/etc/prometheus/alerts.yml"]
    rules = alerts["groups"][0]["rules"]
    high_error = next(rule for rule in rules if rule["alert"] == "HighErrorRate")
    assert "prediction_errors_total" in high_error["expr"]
    assert "prediction_requests_total" in high_error["expr"]


def test_grafana_datasource_provider_and_dashboard_share_stable_uid():
    datasource = _yaml("monitoring/grafana/provisioning/datasources/prometheus.yml")
    provider = _yaml("monitoring/grafana/provisioning/dashboards/churn.yml")
    dashboard = json.loads(
        (ROOT / "monitoring/grafana/dashboards/model_performance.json").read_text()
    )

    configured = datasource["datasources"][0]
    assert configured["uid"] == "prometheus"
    assert configured["url"] == "http://prometheus:9090"
    assert provider["providers"][0]["options"]["path"] == "/var/lib/grafana/dashboards"
    assert dashboard["uid"] == "churn-model-performance"
    assert all(panel["datasource"]["uid"] == "prometheus" for panel in dashboard["panels"])
