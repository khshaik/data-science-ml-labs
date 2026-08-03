"""
Unit tests for model training pipeline
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.training.train import ModelTrainer


@pytest.fixture
def model_trainer(base_config, tmp_path):
    """Create ModelTrainer instance"""
    config = base_config.copy()
    config['model_dir'] = str(tmp_path / "models")
    config['data_path'] = str(tmp_path / "data.csv")
    return ModelTrainer(config)


@pytest.fixture
def training_data():
    """Create sample training data"""
    np.random.seed(42)
    n_samples = 100
    
    X_train = pd.DataFrame(
        np.random.randn(n_samples, 10),
        columns=[f'feature_{i}' for i in range(10)]
    )
    y_train = pd.Series(np.random.randint(0, 2, n_samples))
    
    X_val = pd.DataFrame(
        np.random.randn(20, 10),
        columns=[f'feature_{i}' for i in range(10)]
    )
    y_val = pd.Series(np.random.randint(0, 2, 20))
    
    X_test = pd.DataFrame(
        np.random.randn(20, 10),
        columns=[f'feature_{i}' for i in range(10)]
    )
    y_test = pd.Series(np.random.randint(0, 2, 20))
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def test_model_trainer_initialization(model_trainer):
    """Test ModelTrainer initialization"""
    assert model_trainer is not None
    assert model_trainer.config is not None


@patch('src.training.train.BaselineModel')
def test_train_baseline_model(mock_baseline, model_trainer, training_data):
    """Test training baseline model"""
    X_train, y_train, X_val, y_val, X_test, y_test = training_data
    
    # Mock the baseline model
    mock_model_instance = Mock()
    mock_model_instance.train.return_value = None
    mock_model_instance.evaluate.return_value = {
        'accuracy': 0.85,
        'auc': 0.90,
        'precision': 0.82,
        'recall': 0.88,
        'f1': 0.85
    }
    mock_baseline.return_value = mock_model_instance
    
    metrics = model_trainer.train_baseline(X_train, y_train, X_val, y_val)
    
    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert mock_model_instance.train.called


@patch('src.training.train.CandidateModel')
def test_train_candidate_model(mock_candidate, model_trainer, training_data):
    """Test training candidate model"""
    X_train, y_train, X_val, y_val, X_test, y_test = training_data
    
    # Mock the candidate model
    mock_model_instance = Mock()
    mock_model_instance.build_model.return_value = None
    mock_model_instance.train.return_value = {
        'final_epoch': 10,
        'train_loss': 0.3,
        'val_loss': 0.35,
        'val_auc': 0.92
    }
    mock_model_instance.evaluate.return_value = {
        'accuracy': 0.88,
        'auc': 0.92,
        'precision': 0.85,
        'recall': 0.90,
        'f1': 0.87
    }
    mock_candidate.return_value = mock_model_instance
    
    metrics = model_trainer.train_candidate(X_train, y_train, X_val, y_val)
    
    assert isinstance(metrics, dict)
    assert mock_model_instance.build_model.called
    assert mock_model_instance.train.called


def test_compare_models(model_trainer):
    """Test model comparison"""
    baseline_metrics = {
        'accuracy': 0.85,
        'auc': 0.90,
        'precision': 0.82,
        'recall': 0.88,
        'f1': 0.85
    }
    
    candidate_metrics = {
        'accuracy': 0.88,
        'auc': 0.92,
        'precision': 0.85,
        'recall': 0.90,
        'f1': 0.87
    }
    
    comparison = model_trainer.compare_models(baseline_metrics, candidate_metrics)
    
    assert isinstance(comparison, dict)
    assert 'baseline' in comparison
    assert 'candidate' in comparison
    assert 'winner' in comparison


def test_save_training_report(model_trainer, tmp_path):
    """Test saving training report"""
    report = {
        'baseline_metrics': {'accuracy': 0.85, 'auc': 0.90},
        'candidate_metrics': {'accuracy': 0.88, 'auc': 0.92},
        'winner': 'candidate'
    }
    
    output_path = tmp_path / "training_report.json"
    model_trainer.save_training_report(report, str(output_path))
    
    assert output_path.exists()
