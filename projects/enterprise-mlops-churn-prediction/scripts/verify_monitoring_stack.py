"""Verify the local API -> Prometheus -> Grafana monitoring path."""

import argparse
import base64
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PREDICTION_PAYLOAD = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 65.50,
    "TotalCharges": 786.00,
}


def request_json(url, *, method="GET", payload=None, auth=None, timeout=10):
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if auth:
        token = base64.b64encode(f"{auth[0]}:{auth[1]}".encode()).decode()
        headers["Authorization"] = f"Basic {token}"
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                parsed = body
            return response.status, parsed
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Request failed for {url}: {exc}") from exc


def wait_for_prometheus_target(timeout_seconds=45):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        _, payload = request_json("http://127.0.0.1:9090/api/v1/targets")
        targets = payload["data"]["activeTargets"]
        api_targets = [
            target for target in targets
            if target.get("labels", {}).get("job") == "churn-prediction-api"
        ]
        if api_targets and all(target.get("health") == "up" for target in api_targets):
            return api_targets
        time.sleep(2)
    raise RuntimeError("Prometheus did not report the API target as up")


def wait_for_prometheus_metric(metric_name, timeout_seconds=45):
    deadline = time.monotonic() + timeout_seconds
    query_url = "http://127.0.0.1:9090/api/v1/query?" + urlencode({"query": metric_name})
    while time.monotonic() < deadline:
        _, payload = request_json(query_url)
        results = payload["data"]["result"]
        if results:
            return float(results[0]["value"][1])
        time.sleep(2)
    raise RuntimeError(f"Prometheus did not scrape {metric_name}")


def verify(admin_user, admin_password):
    api_health_status, api_health = request_json("http://127.0.0.1:8000/health")
    prediction_status, prediction = request_json(
        "http://127.0.0.1:8000/predict",
        method="POST",
        payload=PREDICTION_PAYLOAD,
    )
    prometheus_ready_status, _ = request_json("http://127.0.0.1:9090/-/ready")
    api_targets = wait_for_prometheus_target()
    request_metric_value = wait_for_prometheus_metric("prediction_requests_total")

    _, rules_payload = request_json("http://127.0.0.1:9090/api/v1/rules")
    alert_rules = [
        rule
        for group in rules_payload["data"]["groups"]
        for rule in group["rules"]
        if rule.get("type") == "alerting"
    ]
    loaded_rules = sorted(rule["name"] for rule in alert_rules)
    rule_states = {rule["name"]: rule.get("state") for rule in alert_rules}
    expected_rules = sorted(["APIDown", "HighErrorRate", "HighLatency", "LowThroughput"])
    if loaded_rules != expected_rules:
        raise RuntimeError(f"Unexpected Prometheus rules: {loaded_rules}")

    grafana_auth = (admin_user, admin_password)
    grafana_health_status, grafana_health = request_json("http://127.0.0.1:3000/api/health")
    datasource_status, datasource = request_json(
        "http://127.0.0.1:3000/api/datasources/uid/prometheus", auth=grafana_auth
    )
    dashboard_status, dashboard = request_json(
        "http://127.0.0.1:3000/api/dashboards/uid/churn-model-performance",
        auth=grafana_auth,
    )

    return {
        "verification_timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "success",
        "scope": "local Docker Compose monitoring path; external notifications excluded",
        "api": {
            "health_http_status": api_health_status,
            "model_loaded": api_health.get("model_loaded"),
            "model_version": api_health.get("model_version"),
            "prediction_http_status": prediction_status,
            "prediction_model_version": prediction.get("model_version"),
        },
        "prometheus": {
            "ready_http_status": prometheus_ready_status,
            "api_target_health": [target["health"] for target in api_targets],
            "prediction_requests_total": request_metric_value,
            "loaded_alert_rules": loaded_rules,
            "alert_rule_states": rule_states,
        },
        "grafana": {
            "health_http_status": grafana_health_status,
            "database": grafana_health.get("database"),
            "datasource_http_status": datasource_status,
            "datasource_uid": datasource.get("uid"),
            "datasource_url": datasource.get("url"),
            "dashboard_http_status": dashboard_status,
            "dashboard_uid": dashboard.get("dashboard", {}).get("uid"),
            "dashboard_title": dashboard.get("dashboard", {}).get("title"),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admin-user", default="admin")
    parser.add_argument("--admin-password", default="change-me-local")
    parser.add_argument(
        "--output", default="artifacts/monitoring/stack_verification.json"
    )
    args = parser.parse_args()

    result = verify(args.admin_user, args.admin_password)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
