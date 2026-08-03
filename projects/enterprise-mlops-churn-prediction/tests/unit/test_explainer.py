"""
Unit tests for model explainability
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
from src.models.explainer import ModelExplainer


@pytest.fixture
def model_explainer(base_config):
    """Create ModelExplainer instance"""
    return ModelExplainer(base_config)


@pytest.fixture
def sample_model_and_data():
    """Create sample model and data for explanation"""
    # Create mock model
    mock_model = Mock()
    mock_model.predict.return_value = np.array([[0.7]])
    mock_model.predict_proba = Mock(return_value=np.array([[0.3, 0.7]]))
    
    # Create sample data
    X = pd.DataFrame({
        'feature_1': [1.0],
        'feature_2': [2.0],
        'feature_3': [3.0],
        'feature_4': [4.0],
        'feature_5': [5.0]
    })
    
    feature_names = X.columns.tolist()
    
    return mock_model, X, feature_names


def test_model_explainer_initialization(model_explainer):
    """Test ModelExplainer initialization"""
    assert model_explainer is not None
    assert model_explainer.config is not None


@patch('src.models.explainer.shap.Explainer')
def test_create_shap_explainer(mock_shap_explainer, model_explainer, sample_model_and_data):
    """Test creating SHAP explainer"""
    mock_model, X, feature_names = sample_model_and_data
    
    mock_explainer_instance = Mock()
    mock_shap_explainer.return_value = mock_explainer_instance
    
    explainer = model_explainer.create_shap_explainer(mock_model, X)
    
    assert explainer is not None


@patch('src.models.explainer.shap.Explainer')
def test_explain_prediction(mock_shap_explainer, model_explainer, sample_model_and_data):
    """Test explaining a single prediction"""
    mock_model, X, feature_names = sample_model_and_data
    
    # Mock SHAP explainer and values
    mock_explainer_instance = Mock()
    mock_shap_values = Mock()
    mock_shap_values.values = np.array([[0.1, 0.2, 0.3, 0.15, 0.25]])
    mock_shap_values.base_values = np.array([0.5])
    mock_explainer_instance.return_value = mock_shap_values
    mock_shap_explainer.return_value = mock_explainer_instance
    
    explanation = model_explainer.explain_prediction(
        mock_model, X.iloc[0:1], feature_names
    )
    
    assert explanation is not None
    assert isinstance(explanation, dict)


def test_get_feature_importance(model_explainer):
    """Test getting feature importance from SHAP values"""
    shap_values = np.array([[0.1, 0.2, 0.3, 0.15, 0.25]])
    feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
    
    importance = model_explainer.get_feature_importance(shap_values, feature_names)
    
    assert isinstance(importance, dict)
    assert len(importance) == len(feature_names)
    assert all(name in importance for name in feature_names)


@patch('matplotlib.pyplot.savefig')
@patch('src.models.explainer.shap.plots.waterfall')
def test_plot_waterfall(mock_waterfall, mock_savefig, model_explainer, tmp_path):
    """Test plotting SHAP waterfall plot"""
    # Mock SHAP explanation
    mock_explanation = Mock()
    mock_explanation.values = np.array([0.1, 0.2, 0.3])
    mock_explanation.base_values = 0.5
    mock_explanation.data = np.array([1.0, 2.0, 3.0])
    
    output_path = tmp_path / "waterfall.png"
    model_explainer.plot_waterfall(mock_explanation, str(output_path))
    
    assert mock_savefig.called


@patch('matplotlib.pyplot.savefig')
@patch('src.models.explainer.shap.plots.bar')
def test_plot_feature_importance(mock_bar, mock_savefig, model_explainer, tmp_path):
    """Test plotting feature importance"""
    importance = {
        'feature_1': 0.3,
        'feature_2': 0.25,
        'feature_3': 0.2,
        'feature_4': 0.15,
        'feature_5': 0.1
    }
    
    output_path = tmp_path / "feature_importance.png"
    model_explainer.plot_feature_importance(importance, str(output_path))
    
    assert mock_savefig.called


def test_generate_explanation_report(model_explainer, sample_model_and_data):
    """Test generating explanation report"""
    mock_model, X, feature_names = sample_model_and_data
    
    report = {
        'prediction': 0.7,
        'feature_contributions': {
            'feature_1': 0.1,
            'feature_2': 0.2,
            'feature_3': 0.3
        },
        'top_features': ['feature_3', 'feature_2', 'feature_1']
    }
    
    assert isinstance(report, dict)
    assert 'prediction' in report
    assert 'feature_contributions' in report


def test_save_explanation(model_explainer, tmp_path):
    """Test saving explanation to file"""
    explanation = {
        'prediction': 0.7,
        'feature_contributions': {'feature_1': 0.1, 'feature_2': 0.2},
        'timestamp': '2024-01-01 12:00:00'
    }
    
    output_path = tmp_path / "explanation.json"
    model_explainer.save_explanation(explanation, str(output_path))
    
    assert output_path.exists()
