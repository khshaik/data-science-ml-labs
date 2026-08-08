"""
Unit tests for data preprocessing
Bonus Feature: Comprehensive Testing (+0.5 mark)
"""

import pytest
import pandas as pd
import numpy as np
from src.data.preprocessing import DataPreprocessor


# Config is now provided by conftest.py base_config fixture


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return pd.DataFrame({
        'customerID': ['C001', 'C002', 'C003', 'C004', 'C005'],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'SeniorCitizen': [0, 1, 0, 1, 0],
        'Partner': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'Dependents': ['No', 'No', 'Yes', 'No', 'Yes'],
        'tenure': [12, 24, 36, 1, 48],
        'PhoneService': ['Yes', 'Yes', 'No', 'Yes', 'Yes'],
        'MultipleLines': ['No', 'Yes', 'No phone service', 'No', 'Yes'],
        'InternetService': ['DSL', 'Fiber optic', 'DSL', 'No', 'Fiber optic'],
        'OnlineSecurity': ['Yes', 'No', 'No', 'No internet service', 'Yes'],
        'OnlineBackup': ['No', 'Yes', 'No', 'No internet service', 'Yes'],
        'DeviceProtection': ['No', 'No', 'No', 'No internet service', 'Yes'],
        'TechSupport': ['Yes', 'No', 'No', 'No internet service', 'No'],
        'StreamingTV': ['No', 'Yes', 'No', 'No internet service', 'Yes'],
        'StreamingMovies': ['No', 'Yes', 'No', 'No internet service', 'Yes'],
        'Contract': ['One year', 'Month-to-month', 'Two year', 'Month-to-month', 'Two year'],
        'PaperlessBilling': ['Yes', 'Yes', 'No', 'Yes', 'No'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card', 'Bank transfer'],
        'MonthlyCharges': [50.0, 80.0, 60.0, 30.0, 90.0],
        'TotalCharges': ['600.0', '1920.0', '2160.0', '30.0', '4320.0'],
        'Churn': ['No', 'Yes', 'No', 'Yes', 'No']
    })


def test_preprocessor_initialization(base_config):
    """Test DataPreprocessor initialization"""
    preprocessor = DataPreprocessor(base_config)
    assert preprocessor is not None
    assert preprocessor.config == base_config


def test_clean_data(sample_data, base_config):
    """Test data cleaning"""
    preprocessor = DataPreprocessor(base_config)
    cleaned = preprocessor.clean_data(sample_data.copy())
    
    # Check TotalCharges converted to numeric
    assert cleaned['TotalCharges'].dtype in [np.float64, np.float32]
    
    # Check no missing values in TotalCharges
    assert not cleaned['TotalCharges'].isna().any()


def test_encode_categorical(sample_data, base_config):
    """Test categorical encoding"""
    preprocessor = DataPreprocessor(base_config)
    cleaned = preprocessor.clean_data(sample_data.copy())
    encoded = preprocessor.encode_categorical(cleaned)
    
    # Check binary columns are encoded
    assert encoded['gender'].isin([0, 1]).all()
    assert encoded['Partner'].isin([0, 1]).all()
    
    # Multi-class inputs retain one integer-encoded column; this is not one-hot.
    assert 'Contract' in encoded.columns
    assert pd.api.types.is_integer_dtype(encoded['Contract'])


def test_scale_numerical(sample_data, base_config):
    """Test numerical scaling"""
    preprocessor = DataPreprocessor(base_config)
    cleaned = preprocessor.clean_data(sample_data.copy())
    encoded = preprocessor.encode_categorical(cleaned)
    
    # Fit scaler
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    if all(col in encoded.columns for col in numerical_cols):
        preprocessor.scaler.fit(encoded[numerical_cols])
        
        # Check scaler is fitted
        assert hasattr(preprocessor.scaler, 'mean_')
        assert len(preprocessor.scaler.mean_) == len(numerical_cols)


def test_train_test_split(sample_data_large, base_config):
    """Test train/val/test split"""
    preprocessor = DataPreprocessor(base_config)
    
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_for_training(sample_data_large)
    
    # Check split sizes
    total_size = len(sample_data_large)
    assert len(X_train) == int(total_size * 0.6)
    assert len(X_val) == int(total_size * 0.2)
    assert len(X_test) <= int(total_size * 0.2) + 1  # Allow for rounding
    
    # Check no data leakage
    assert len(X_train) + len(X_val) + len(X_test) == total_size


def test_preprocess_for_serving(sample_data_large, base_config):
    """Test preprocessing for serving (online inference)"""
    preprocessor = DataPreprocessor(base_config)
    
    # First train to fit preprocessor
    preprocessor.preprocess_for_training(sample_data_large)
    
    # Test single row preprocessing
    single_row = sample_data_large.iloc[[0]].copy()
    processed = preprocessor.preprocess_for_serving(single_row)
    
    # Check output is DataFrame
    assert isinstance(processed, pd.DataFrame)
    
    # Check has correct features
    assert len(processed.columns) > 0


def test_save_load_preprocessor(sample_data_large, base_config, tmp_path):
    """Test saving and loading preprocessor"""
    preprocessor = DataPreprocessor(base_config)
    
    # Train preprocessor
    preprocessor.preprocess_for_training(sample_data_large)
    
    # Save
    save_path = tmp_path / "preprocessor.pkl"
    preprocessor.save_preprocessor(str(save_path))
    
    # Load
    new_preprocessor = DataPreprocessor(base_config)
    new_preprocessor.load_preprocessor(str(save_path))
    
    # Check loaded preprocessor works
    single_row = sample_data_large.iloc[[0]].copy()
    processed = new_preprocessor.preprocess_for_serving(single_row)
    assert isinstance(processed, pd.DataFrame)


def test_feature_names_consistency(sample_data_large, base_config):
    """Test that feature names are consistent across preprocessing"""
    preprocessor = DataPreprocessor(base_config)
    
    X_train, X_val, X_test, _, _, _ = preprocessor.preprocess_for_training(sample_data_large)
    
    # Check all splits have same features
    assert list(X_train.columns) == list(X_val.columns)
    assert list(X_train.columns) == list(X_test.columns)
    
    # Check feature_names attribute is set
    assert hasattr(preprocessor, 'feature_names')
    assert len(preprocessor.feature_names) > 0


def test_handle_missing_values(base_config):
    """Test handling of missing values"""
    preprocessor = DataPreprocessor(base_config)
    
    # Create data with missing values
    data_with_missing = pd.DataFrame({
        'customerID': ['C001', 'C002'],
        'gender': ['Male', None],
        'SeniorCitizen': [0, 1],
        'tenure': [12, None],
        'MonthlyCharges': [50.0, 80.0],
        'TotalCharges': ['600.0', None],
        'Churn': ['No', 'Yes']
    })
    
    cleaned = preprocessor.clean_data(data_with_missing)
    
    # Check missing values are handled
    assert not cleaned['TotalCharges'].isna().any()


def test_target_encoding(sample_data_large, base_config):
    """Test target variable encoding"""
    preprocessor = DataPreprocessor(base_config)
    
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_for_training(sample_data_large)
    
    # Check target is binary
    assert y_train.isin([0, 1]).all()
    assert y_val.isin([0, 1]).all()
    assert y_test.isin([0, 1]).all()
