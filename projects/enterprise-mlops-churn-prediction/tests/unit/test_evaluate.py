"""Tests for model comparison and promotion guardrails."""

import json
import pytest

from src.training.evaluate import ModelEvaluator


def _report(auc, recall, precision=0.6, accuracy=0.75, f1=0.65):
    metrics = {
        "dataset": "validation", "auc": auc, "recall": recall,
        "precision": precision, "accuracy": accuracy, "f1": f1,
        "confusion_matrix": [[10, 2], [3, 5]],
    }
    return {"validation_metrics": metrics, "test_metrics": metrics}


def _evaluator():
    return ModelEvaluator({"models": {"promotion": {
        "min_auc": 0.80, "min_recall": 0.75, "min_auc_gain": 0.0,
    }}})


def test_compare_models_calculates_metric_differences():
    comparison = _evaluator().compare_models(_report(0.82, 0.76), _report(0.85, 0.79))
    assert comparison["differences"]["auc"] == pytest.approx(0.03)
    assert comparison["candidate"]["recall"] == 0.79


def test_candidate_passes_all_guardrails():
    promote, reason = _evaluator().check_promotion_guardrails(
        _report(0.82, 0.76), _report(0.85, 0.79)
    )
    assert promote is True
    assert reason == "All promotion guardrails passed"


def test_candidate_fails_minimum_auc():
    promote, reason = _evaluator().check_promotion_guardrails(
        _report(0.82, 0.76), _report(0.79, 0.80)
    )
    assert promote is False
    assert "AUC below threshold" in reason


def test_more_complex_candidate_is_not_promoted_when_auc_is_lower():
    promote, reason = _evaluator().check_promotion_guardrails(
        _report(0.835, 0.79), _report(0.830, 0.78)
    )
    assert promote is False
    assert "does not improve baseline" in reason


def test_save_comparison_report_writes_json_and_markdown(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    evaluator = _evaluator()
    comparison = evaluator.compare_models(_report(0.82, 0.76), _report(0.85, 0.79))
    evaluator.save_comparison_report(comparison, True, "All promotion guardrails passed")
    json_path = tmp_path / "artifacts/eval/model_comparison.json"
    assert json_path.exists()
    assert (tmp_path / "artifacts/eval/model_comparison.md").exists()
    assert json.loads(json_path.read_text())["promotion_decision"]["should_promote"] is True
