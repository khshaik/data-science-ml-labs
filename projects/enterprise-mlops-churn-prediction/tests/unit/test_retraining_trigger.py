"""Tests for the implemented multi-signal retraining decision."""

import json

from src.retraining.trigger import RetrainingTrigger


def _trigger():
    return RetrainingTrigger({"retraining": {
        "min_new_data_count": 1000,
        "max_auc_drop": 0.05,
        "max_drift_score": 0.3,
        "max_days_since_training": 30,
    }})


def test_no_signal_does_not_retrain():
    decision, reason, details = _trigger().should_retrain({
        "new_labeled_data_count": 200, "current_auc": 0.82,
        "baseline_auc": 0.83, "drift_score": 0.1, "days_since_training": 5,
    })
    assert decision is False
    assert reason == "No retraining signals triggered"
    assert details["triggered_signals"] == 0


def test_each_supported_signal_can_trigger_retraining():
    cases = [
        {"new_labeled_data_count": 1000},
        {"current_auc": 0.70, "baseline_auc": 0.82},
        {"drift_score": 0.31},
        {"days_since_training": 30},
    ]
    for metrics in cases:
        decision, _, details = _trigger().should_retrain(metrics)
        assert decision is True
        assert details["triggered_signals"] >= 1


def test_save_trigger_log(tmp_path):
    trigger = _trigger()
    _, _, details = trigger.should_retrain({"new_labeled_data_count": 1000})
    output = tmp_path / "trigger.json"
    trigger.save_trigger_log(details, str(output))
    assert json.loads(output.read_text())["should_retrain"] is True
