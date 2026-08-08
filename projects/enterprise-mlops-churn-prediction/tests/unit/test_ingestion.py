"""Tests for the public batch-ingestion workflow."""

import yaml
import pandas as pd

from src.data.ingestion import DataIngestion


def _write_config(path):
    path.write_text(yaml.safe_dump({"monitoring": {
        "missing_rate_threshold": 0.05,
        "consistency_is_blocking": False,
    }}))


def _valid_rows():
    return pd.DataFrame({
        "customerID": ["C001", "C002"], "gender": ["Male", "Female"],
        "SeniorCitizen": [0, 1], "Partner": ["Yes", "No"],
        "Dependents": ["No", "No"], "tenure": [12, 24],
        "PhoneService": ["Yes", "Yes"], "MultipleLines": ["No", "Yes"],
        "InternetService": ["DSL", "Fiber optic"],
        "OnlineSecurity": ["Yes", "No"], "OnlineBackup": ["No", "Yes"],
        "DeviceProtection": ["No", "No"], "TechSupport": ["Yes", "No"],
        "StreamingTV": ["No", "Yes"], "StreamingMovies": ["No", "Yes"],
        "Contract": ["One year", "Month-to-month"],
        "PaperlessBilling": ["Yes", "Yes"],
        "PaymentMethod": ["Electronic check", "Mailed check"],
        "MonthlyCharges": [50.0, 80.0], "TotalCharges": ["600.0", "1920.0"],
        "Churn": ["No", "Yes"],
    })


def test_ingest_batch_creates_training_file_and_summary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    incoming = tmp_path / "incoming.csv"
    output = tmp_path / "training.csv"
    _valid_rows().to_csv(incoming, index=False)

    summary = DataIngestion(str(config_path)).ingest_batch(str(incoming), str(output))

    assert summary["status"] == "success"
    assert summary["new_rows"] == 2
    assert output.exists()
    assert len(pd.read_csv(output)) == 2
    assert list((tmp_path / "artifacts/logs").glob("ingestion_*.json"))


def test_ingest_batch_incoming_row_wins_by_ingestion_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    output = tmp_path / "training.csv"
    existing = _valid_rows().iloc[[0]].copy()
    existing.to_csv(output, index=False)
    incoming_data = _valid_rows()
    incoming_data.loc[0, "MonthlyCharges"] = 55.0
    incoming = tmp_path / "incoming.csv"
    incoming_data.to_csv(incoming, index=False)

    summary = DataIngestion(str(config_path)).ingest_batch(str(incoming), str(output))
    merged = pd.read_csv(output)

    assert summary["duplicates_removed"] == 1
    assert summary["existing_rows"] == 1
    assert summary["new_rows"] == 2
    assert summary["total_rows"] == 2
    assert summary["deduplication_policy"] == (
        "last_row_by_ingestion_order_incoming_after_existing"
    )
    assert len(merged) == 2
    assert merged.loc[merged.customerID == "C001", "MonthlyCharges"].iloc[0] == 55.0


def test_ingest_batch_rejects_invalid_schema(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    incoming = tmp_path / "invalid.csv"
    pd.DataFrame({"customerID": ["C001"]}).to_csv(incoming, index=False)

    summary = DataIngestion(str(config_path)).ingest_batch(
        str(incoming), str(tmp_path / "training.csv")
    )

    assert summary["status"] == "failed"
    assert summary["reason"] == "data_quality_checks_failed"
    assert not (tmp_path / "training.csv").exists()
