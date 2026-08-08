"""Validate the notification path across Prometheus, Compose, and Alertmanager."""

import importlib.util
import io
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _yaml(path: str):
    return yaml.safe_load((ROOT / path).read_text())


def test_prometheus_and_compose_wire_alertmanager_to_internal_receiver():
    prometheus = _yaml("monitoring/prometheus.yml")
    compose = _yaml("docker/docker-compose.yml")
    alertmanager = _yaml("monitoring/alertmanager/alertmanager.yml")

    targets = prometheus["alerting"]["alertmanagers"][0]["static_configs"][0]["targets"]
    assert targets == ["alertmanager:9093"]
    assert compose["services"]["prometheus"]["depends_on"]["alertmanager"]["condition"] == "service_healthy"
    assert "notification-sink" in compose["services"]
    assert alertmanager["route"]["receiver"] == "internal-audit"
    internal = next(item for item in alertmanager["receivers"] if item["name"] == "internal-audit")
    assert internal["webhook_configs"][0]["url"] == "http://notification-sink:8080/alerts"


def test_external_template_fans_out_without_committed_credentials():
    external = _yaml("monitoring/alertmanager/alertmanager.external.example.yml")
    routes = external["route"]["routes"]
    receivers = {item["name"]: item for item in external["receivers"]}

    assert routes[0]["receiver"] == "internal-audit"
    assert routes[0]["continue"] is True
    assert routes[1]["receiver"] == "external-channels"
    assert routes[1]["matchers"] == ['severity=~"warning|critical"']
    channels = receivers["external-channels"]
    assert channels["webhook_configs"][0]["url_file"].startswith("/run/secrets/")
    assert channels["slack_configs"][0]["api_url_file"].startswith("/run/secrets/")
    assert external["global"]["smtp_auth_password_file"].startswith("/run/secrets/")

    template_text = (ROOT / "monitoring/alertmanager/alertmanager.external.example.yml").read_text()
    assert "hooks.slack.com/services/" not in template_text
    assert "smtp_auth_password:" not in template_text


def test_notification_secrets_and_local_config_are_git_ignored():
    ignore_rules = (ROOT / ".gitignore").read_text()
    compose_text = (ROOT / "docker/docker-compose.yml").read_text()

    assert "monitoring/alertmanager/alertmanager.local.yml" in ignore_rules
    assert "monitoring/alertmanager/secrets/*" in ignore_rules
    assert "ALERTMANAGER_CONFIG_FILE" in compose_text
    assert "/run/secrets:ro" in compose_text

    evidence = json.loads(
        (ROOT / "artifacts/monitoring/notification_routing_verification.json").read_text()
    )
    assert evidence["internal_delivery"]["verified"] is True
    assert evidence["external_delivery"]["verified"] is False
    assert evidence["external_delivery"]["reason"] == "credentials_not_supplied"


def test_internal_receiver_accepts_and_persists_alertmanager_payload(tmp_path):
    """Exercise Alertmanager payload validation and durable internal delivery."""
    module_path = ROOT / "monitoring/notification_sink.py"
    spec = importlib.util.spec_from_file_location("notification_sink", module_path)
    sink = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sink)
    sink.LOG_PATH = tmp_path / "notifications.jsonl"

    payload = {
        "status": "firing",
        "alerts": [{"labels": {"alertname": "APIDown", "severity": "critical"}}],
    }
    encoded = json.dumps(payload).encode("utf-8")
    handler = sink.NotificationHandler.__new__(sink.NotificationHandler)
    handler.path = "/alerts"
    handler.headers = {"Content-Length": str(len(encoded))}
    handler.rfile = io.BytesIO(encoded)
    response = {}
    handler._respond = lambda status, body: response.update(status=status, body=body)

    handler.do_POST()

    assert response == {"status": 202, "body": {"accepted": 1}}

    records = sink.LOG_PATH.read_text().splitlines()
    assert len(records) == 1
    record = json.loads(records[0])
    assert record["source"] == "alertmanager"
    assert record["payload"] == payload
