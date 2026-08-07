"""Tests for the implemented SHAP/LIME explainer interface."""

import json
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src.models.explainer import ModelExplainer


def _data():
    return pd.DataFrame({"feature_1": [1.0, 1.5], "feature_2": [2.0, 2.5]})


@patch("src.models.explainer.LimeTabularExplainer")
@patch("src.models.explainer.shap.Explainer")
def test_initialization_uses_supplied_model_and_background(mock_shap, mock_lime):
    model = Mock()
    data = _data()
    explainer = ModelExplainer(model, data, list(data.columns), "baseline")
    assert explainer.model is model
    assert explainer.feature_names == ["feature_1", "feature_2"]
    mock_shap.assert_called_once()
    mock_lime.assert_called_once()


@patch("src.models.explainer.LimeTabularExplainer")
@patch("src.models.explainer.shap.Explainer")
def test_single_shap_explanation_returns_ranked_contributions(mock_shap, mock_lime):
    values = Mock()
    values.values = np.array([[0.1, -0.4]])
    values.base_values = np.array([0.3])
    values.shape = values.values.shape
    mock_shap.return_value.return_value = values
    data = _data()
    explainer = ModelExplainer(Mock(), data, list(data.columns), "baseline")

    result = explainer.explain_single_prediction_shap(data.iloc[[0]], "C1")

    assert result["customer_id"] == "C1"
    assert result["top_contributions"][0][0] == "feature_2"
    assert result["method"] == "SHAP"


@patch("src.models.explainer.LimeTabularExplainer")
@patch("src.models.explainer.shap.Explainer")
def test_lime_explanation_uses_positive_class_probability(mock_shap, mock_lime):
    lime_result = Mock()
    lime_result.as_list.return_value = [("feature_1", 0.2), ("feature_2", -0.1)]
    lime_result.predict_proba = np.array([0.3, 0.7])
    mock_lime.return_value.explain_instance.return_value = lime_result
    model = Mock()
    model.predict_proba.return_value = np.array([[0.3, 0.7]])
    data = _data()
    explainer = ModelExplainer(model, data, list(data.columns), "baseline")

    result = explainer.explain_with_lime(data.iloc[[0]], "C1")

    assert result["prediction_probability"] == 0.7
    assert result["method"] == "LIME"


@patch("src.models.explainer.LimeTabularExplainer")
@patch("src.models.explainer.shap.Explainer")
def test_save_explanation_writes_json(mock_shap, mock_lime, tmp_path):
    data = _data()
    explainer = ModelExplainer(Mock(), data, list(data.columns), "baseline")
    output = tmp_path / "explanation.json"
    explainer.save_explanation({"prediction_probability": 0.7}, str(output))
    assert json.loads(output.read_text())["prediction_probability"] == 0.7
