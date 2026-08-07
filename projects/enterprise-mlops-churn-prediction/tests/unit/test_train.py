"""Focused tests for the repeatable training pipeline."""

import json
from unittest.mock import patch

import yaml

from src.training.train import TrainingPipeline


def _config():
    return {
        "data": {
            "raw_path": "data.csv", "train_split": 0.6,
            "val_split": 0.2, "test_split": 0.2, "random_seed": 42,
        },
        "models": {
            "baseline": {"path": "models/baseline", "hyperparameters": {}},
            "candidate": {"path": "models/candidate", "hyperparameters": {}},
        },
        "mlflow": {"tracking_uri": "sqlite:///test.db", "experiment_name": "test"},
    }


@patch("src.training.train.mlflow.set_experiment")
@patch("src.training.train.mlflow.set_tracking_uri")
def test_training_pipeline_initialization(mock_uri, mock_experiment, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_config()))

    pipeline = TrainingPipeline(str(config_path))

    assert pipeline.config["data"]["train_split"] == 0.6
    mock_uri.assert_called_once_with("sqlite:///test.db")
    mock_experiment.assert_called_once_with("test")


@patch("src.training.train.mlflow.log_artifact")
def test_save_evaluation_report_writes_reproducible_json(mock_log, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pipeline = TrainingPipeline.__new__(TrainingPipeline)
    validation = {"auc": 0.82, "recall": 0.76}
    test = {"auc": 0.83, "recall": 0.77}

    pipeline._save_evaluation_report("baseline", validation, test)

    output = tmp_path / "artifacts/eval/baseline_evaluation.json"
    report = json.loads(output.read_text())
    assert report["model_type"] == "baseline"
    assert report["validation_metrics"] == validation
    assert report["test_metrics"] == test
    mock_log.assert_called_once()
