"""
Unit tests for model training and evaluation
Bonus Feature: Comprehensive Testing (+0.5 mark)
"""

import pytest
import pandas as pd
import numpy as np
from src.models.baseline import BaselineModel
from src.models.candidate import CandidateModel


@pytest.fixture
def sample_training_data():
    """Create sample training data"""
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    X_train = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y_train = pd.Series(np.random.randint(0, 2, n_samples))
    
    X_val = pd.DataFrame(
        np.random.randn(20, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y_val = pd.Series(np.random.randint(0, 2, 20))
    
    return X_train, y_train, X_val, y_val


def test_baseline_model_initialization():
    """Test BaselineModel initialization"""
    config = {
        'models': {
            'baseline': {
                'hyperparameters': {
                    'C': 1.0,
                    'max_iter': 1000,
                    'random_state': 42
                }
            }
        }
    }
    model = BaselineModel(config)
    assert model is not None
    assert model.model is None  # Not trained yet


def test_baseline_model_training(sample_training_data):
    """Test baseline model training"""
    X_train, y_train, X_val, y_val = sample_training_data
    
    config = {
        'models': {
            'baseline': {
                'hyperparameters': {
                    'C': 1.0,
                    'max_iter': 1000,
                    'random_state': 42
                }
            }
        }
    }
    model = BaselineModel(config)
    model.build_model()
    model.train(X_train, y_train)
    
    # Check model is trained
    assert model.model is not None
    assert hasattr(model.model, 'predict')


def test_baseline_model_prediction(sample_training_data):
    """Test baseline model prediction"""
    X_train, y_train, X_val, y_val = sample_training_data
    
    config = {
        'models': {
            'baseline': {
                'hyperparameters': {
                    'C': 1.0,
                    'max_iter': 1000,
                    'random_state': 42
                }
            }
        }
    }
    model = BaselineModel(config)
    model.build_model()
    model.train(X_train, y_train)
    
    # Test prediction
    predictions = model.predict(X_val)
    
    # Check predictions
    assert len(predictions) == len(X_val)
    assert all(pred in [0, 1] for pred in predictions)


def test_baseline_model_predict_proba(sample_training_data, base_config):
    """Test baseline model probability prediction"""
    X_train, y_train, X_val, y_val = sample_training_data
    
    model = BaselineModel(base_config)
    model.train(X_train, y_train)
    
    # Test probability prediction
    probabilities = model.predict_proba(X_val)
    
    # Check probabilities
    assert len(probabilities) == len(X_val)
    assert all(0 <= prob <= 1 for prob in probabilities)


def test_baseline_model_evaluation(sample_training_data, base_config):
    """Test baseline model evaluation"""
    X_train, y_train, X_val, y_val = sample_training_data
    
    model = BaselineModel(base_config)
    model.train(X_train, y_train)
    
    # Evaluate
    metrics = model.evaluate(X_val, y_val)
    
    # Check metrics exist
    assert 'accuracy' in metrics
    assert 'auc' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    
    # Check metrics are valid
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['auc'] <= 1


def test_baseline_feature_importance(sample_training_data, base_config):
    """Test baseline model feature importance"""
    X_train, y_train, X_val, y_val = sample_training_data
    
    model = BaselineModel(base_config)
    model.train(X_train, y_train)
    
    # Get feature importance
    feature_names = X_train.columns.tolist()
    importance = model.get_feature_importance(feature_names)
    
    # Check importance
    assert len(importance) == X_train.shape[1]
    assert all(isinstance(imp, (int, float)) for imp in importance.values())


def test_baseline_save_load(sample_training_data, tmp_path, base_config):
    """Test baseline model save and load"""
    X_train, y_train, X_val, y_val = sample_training_data
    
    model = BaselineModel(base_config)
    model.train(X_train, y_train)
    
    # Save
    save_path = tmp_path / "baseline_model.pkl"
    model.save_model(str(save_path))
    
    # Load
    new_model = BaselineModel(base_config)
    new_model.load_model(str(save_path))
    
    # Test loaded model
    predictions = new_model.predict(X_val)
    assert len(predictions) == len(X_val)


def test_candidate_model_initialization():
    """Test CandidateModel initialization"""
    config = {
        'models': {
            'candidate': {
                'architecture': {
                    'hidden_layers': [64, 32, 16],
                    'dropout_rate': 0.3,
                    'activation': 'relu'
                },
                'training': {
                    'epochs': 100,
                    'batch_size': 32,
                    'learning_rate': 0.001,
                    'early_stopping_patience': 5
                }
            }
        }
    }
    model = CandidateModel(config)
    assert model is not None


def test_candidate_model_training(sample_training_data):
    """Test candidate model training"""
    X_train, y_train, X_val, y_val = sample_training_data
    
    config = {
        'models': {
            'candidate': {
                'architecture': {
                    'hidden_layers': [64, 32, 16],
                    'dropout_rate': 0.3,
                    'activation': 'relu'
                },
                'training': {
                    'epochs': 5,
                    'batch_size': 32,
                    'learning_rate': 0.001,
                    'early_stopping_patience': 5
                },
                'hyperparameters': {
                    'hidden_layers': [64, 32, 16],
                    'dropout_rate': 0.3,
                    'activation': 'relu',
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 5,
                    'early_stopping_patience': 5
                }
            }
        }
    }
    model = CandidateModel(config)
    model.build_model(X_train.shape[1])
    metrics = model.train(X_train, y_train, X_val, y_val)
    
    # Check training metrics
    assert 'train_loss' in metrics
    assert 'val_loss' in metrics
    assert 'final_epoch' in metrics
    assert metrics['final_epoch'] > 0


def test_candidate_model_prediction(sample_training_data, base_config):
    """Test candidate model prediction"""
    X_train, y_train, X_val, y_val = sample_training_data
    
    model = CandidateModel(base_config)
    model.build_model(X_train.shape[1])
    model.train(X_train, y_train, X_val, y_val)
    
    # Test prediction
    predictions = model.predict(X_val)
    
    # Check predictions
    assert len(predictions) == len(X_val)
    assert all(pred in [0, 1] for pred in predictions)


def test_candidate_model_predict_proba(sample_training_data, base_config):
    """Test candidate model probability prediction"""
    X_train, y_train, X_val, y_val = sample_training_data
    
    model = CandidateModel(base_config)
    model.build_model(X_train.shape[1])
    model.train(X_train, y_train, X_val, y_val)
    
    # Test probability prediction
    probabilities = model.predict_proba(X_val)
    
    # Check probabilities
    assert len(probabilities) == len(X_val)
    assert all(0 <= prob <= 1 for prob in probabilities)


def test_candidate_model_evaluation(sample_training_data, base_config):
    """Test candidate model evaluation"""
    X_train, y_train, X_val, y_val = sample_training_data
    
    model = CandidateModel(base_config)
    model.build_model(X_train.shape[1])
    model.train(X_train, y_train, X_val, y_val)
    
    # Evaluate
    metrics = model.evaluate(X_val, y_val)
    
    # Check metrics exist
    assert 'accuracy' in metrics
    assert 'auc' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics


def test_candidate_early_stopping(sample_training_data, base_config):
    """Test candidate model early stopping"""
    X_train, y_train, X_val, y_val = sample_training_data
    
    model = CandidateModel(base_config)
    model.build_model(X_train.shape[1])
    metrics = model.train(X_train, y_train, X_val, y_val)
    
    # Check early stopping worked (should stop before 100 epochs)
    assert metrics['final_epoch'] < 100


def test_candidate_save_load(sample_training_data, tmp_path, base_config):
    """Test candidate model save and load"""
    X_train, y_train, X_val, y_val = sample_training_data
    
    model = CandidateModel(base_config)
    model.build_model(X_train.shape[1])
    model.train(X_train, y_train, X_val, y_val)
    
    # Save
    save_path = tmp_path / "candidate_model.h5"
    model.save_model(str(save_path))
    
    # Load
    new_model = CandidateModel(base_config)
    new_model.load_model(str(save_path))
    
    # Test loaded model
    predictions = new_model.predict(X_val)
    assert len(predictions) == len(X_val)


def test_model_comparison(sample_training_data, base_config):
    """Test comparison between baseline and candidate models"""
    X_train, y_train, X_val, y_val = sample_training_data
    
    # Train baseline
    baseline = BaselineModel(base_config)
    baseline.train(X_train, y_train)
    baseline_metrics = baseline.evaluate(X_val, y_val)
    
    candidate = CandidateModel(base_config)
    candidate.build_model(X_train.shape[1])
    candidate.train(X_train, y_train, X_val, y_val)
    candidate_metrics = candidate.evaluate(X_val, y_val)
    
    # Both should have valid metrics
    assert 0 <= baseline_metrics['auc'] <= 1
    assert 0 <= candidate_metrics['auc'] <= 1
    
    # Metrics should be comparable
    assert isinstance(baseline_metrics['auc'], (int, float))
    assert isinstance(candidate_metrics['auc'], (int, float))
