"""
Pytest configuration and shared fixtures
"""

import pytest
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def project_root_path():
    """Return project root path"""
    return project_root


@pytest.fixture
def base_config():
    """Base configuration for all tests"""
    return {
        'data': {
            'train_split': 0.6,
            'val_split': 0.2,
            'test_split': 0.2,
            'random_state': 42,
            'random_seed': 42  # Added for preprocessing compatibility
        },
        'models': {
            'baseline': {
                'hyperparameters': {
                    'C': 1.0,
                    'max_iter': 1000,
                    'random_state': 42
                }
            },
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
                },
                'hyperparameters': {  # Added for candidate model compatibility
                    'hidden_layers': [64, 32, 16],
                    'dropout_rate': 0.3,
                    'activation': 'relu',
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 100,
                    'early_stopping_patience': 5
                }
            }
        },
        'monitoring': {
            'drift_threshold_psi': 0.2,
            'drift_threshold_ks_pvalue': 0.05
        }
    }


@pytest.fixture
def sample_data_large():
    """Create larger sample data for testing with train/val/test splits"""
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    n_samples = 100  # Large enough for stratified splits
    
    # Generate diverse data
    data = {
        'customerID': [f'C{i:04d}' for i in range(n_samples)],
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples),
        'Partner': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples),
        'tenure': np.random.randint(1, 72, n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
        'MultipleLines': np.random.choice(['No', 'Yes', 'No phone service'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'MonthlyCharges': np.random.uniform(20, 120, n_samples),
        'TotalCharges': [str(np.random.uniform(20, 8000)) for _ in range(n_samples)],
        'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])  # Balanced for stratification
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def baseline_model(base_config):
    """Create a baseline model instance with config"""
    from src.models.baseline import BaselineModel
    return BaselineModel(base_config)


@pytest.fixture
def candidate_model(base_config):
    """Create a candidate model instance with config"""
    from src.models.candidate import CandidateModel
    return CandidateModel(base_config)


@pytest.fixture
def config(base_config):
    """Alias for base_config for backward compatibility"""
    return base_config


@pytest.fixture
def drift_detector(base_config):
    """Create a drift detector instance with config"""
    from src.monitoring.drift_detector import DriftDetector
    return DriftDetector(base_config)


@pytest.fixture(autouse=True)
def suppress_warnings():
    """Suppress warnings during tests"""
    import warnings
    warnings.filterwarnings("ignore")
