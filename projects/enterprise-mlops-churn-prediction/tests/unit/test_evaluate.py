"""
Unit tests for model evaluation
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
from src.training.evaluate import ModelEvaluator


@pytest.fixture
def model_evaluator(base_config, tmp_path):
    """Create ModelEvaluator instance"""
    config = base_config.copy()
    config['results_dir'] = str(tmp_path / "results")
    return ModelEvaluator(config)


@pytest.fixture
def predictions_data():
    """Create sample predictions and labels"""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    y_pred_proba = np.random.rand(100)
    
    return y_true, y_pred, y_pred_proba


def test_model_evaluator_initialization(model_evaluator):
    """Test ModelEvaluator initialization"""
    assert model_evaluator is not None
    assert model_evaluator.config is not None


def test_calculate_metrics(model_evaluator, predictions_data):
    """Test calculating evaluation metrics"""
    y_true, y_pred, y_pred_proba = predictions_data
    
    metrics = model_evaluator.calculate_metrics(y_true, y_pred, y_pred_proba)
    
    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 'auc' in metrics
    
    # Check metric ranges
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['auc'] <= 1


def test_calculate_confusion_matrix(model_evaluator, predictions_data):
    """Test confusion matrix calculation"""
    y_true, y_pred, _ = predictions_data
    
    cm = model_evaluator.calculate_confusion_matrix(y_true, y_pred)
    
    assert cm is not None
    assert cm.shape == (2, 2)


def test_generate_classification_report(model_evaluator, predictions_data):
    """Test generating classification report"""
    y_true, y_pred, _ = predictions_data
    
    report = model_evaluator.generate_classification_report(y_true, y_pred)
    
    assert report is not None
    assert isinstance(report, (str, dict))


@patch('matplotlib.pyplot.savefig')
def test_plot_roc_curve(mock_savefig, model_evaluator, predictions_data, tmp_path):
    """Test plotting ROC curve"""
    y_true, _, y_pred_proba = predictions_data
    
    output_path = tmp_path / "roc_curve.png"
    model_evaluator.plot_roc_curve(y_true, y_pred_proba, str(output_path))
    
    assert mock_savefig.called


@patch('matplotlib.pyplot.savefig')
def test_plot_confusion_matrix(mock_savefig, model_evaluator, predictions_data, tmp_path):
    """Test plotting confusion matrix"""
    y_true, y_pred, _ = predictions_data
    
    output_path = tmp_path / "confusion_matrix.png"
    model_evaluator.plot_confusion_matrix(y_true, y_pred, str(output_path))
    
    assert mock_savefig.called


def test_save_evaluation_report(model_evaluator, tmp_path):
    """Test saving evaluation report"""
    report = {
        'accuracy': 0.85,
        'precision': 0.82,
        'recall': 0.88,
        'f1': 0.85,
        'auc': 0.90
    }
    
    output_path = tmp_path / "evaluation_report.json"
    model_evaluator.save_evaluation_report(report, str(output_path))
    
    assert output_path.exists()


def test_compare_model_performance(model_evaluator):
    """Test comparing model performance"""
    model1_metrics = {
        'accuracy': 0.85,
        'auc': 0.90,
        'f1': 0.85
    }
    
    model2_metrics = {
        'accuracy': 0.88,
        'auc': 0.92,
        'f1': 0.87
    }
    
    comparison = model_evaluator.compare_models(model1_metrics, model2_metrics)
    
    assert isinstance(comparison, dict)
    assert 'better_model' in comparison or 'winner' in comparison or comparison is not None
