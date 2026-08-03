"""
Unit tests for batch prediction
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
from src.serving.batch_predict import BatchPredictor


@pytest.fixture
def batch_predictor(base_config, tmp_path):
    """Create BatchPredictor instance"""
    config = base_config.copy()
    config['model_path'] = str(tmp_path / "model.pkl")
    config['batch_output_dir'] = str(tmp_path / "predictions")
    return BatchPredictor(config)


@pytest.fixture
def batch_data():
    """Create sample batch data"""
    return pd.DataFrame({
        'customerID': [f'C{i:04d}' for i in range(10)],
        'gender': np.random.choice(['Male', 'Female'], 10),
        'SeniorCitizen': np.random.choice([0, 1], 10),
        'Partner': np.random.choice(['Yes', 'No'], 10),
        'Dependents': np.random.choice(['Yes', 'No'], 10),
        'tenure': np.random.randint(1, 72, 10),
        'PhoneService': np.random.choice(['Yes', 'No'], 10),
        'MultipleLines': np.random.choice(['No', 'Yes', 'No phone service'], 10),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], 10),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], 10),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], 10),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], 10),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], 10),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], 10),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], 10),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], 10),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], 10),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], 10),
        'MonthlyCharges': np.random.uniform(20, 120, 10),
        'TotalCharges': [str(np.random.uniform(20, 8000)) for _ in range(10)]
    })


def test_batch_predictor_initialization(batch_predictor):
    """Test BatchPredictor initialization"""
    assert batch_predictor is not None
    assert batch_predictor.config is not None


@patch('src.serving.batch_predict.joblib.load')
@patch('src.serving.batch_predict.DataPreprocessor')
@patch('src.serving.batch_predict.FeatureEngineer')
def test_load_model(mock_feature_eng, mock_preprocessor, mock_joblib, batch_predictor):
    """Test loading model for batch prediction"""
    mock_joblib.return_value = Mock()
    mock_preprocessor.return_value = Mock()
    mock_feature_eng.return_value = Mock()
    
    batch_predictor.load_model()
    
    assert mock_joblib.called


@patch('src.serving.batch_predict.joblib.load')
@patch('src.serving.batch_predict.DataPreprocessor')
@patch('src.serving.batch_predict.FeatureEngineer')
def test_predict_batch(mock_feature_eng, mock_preprocessor, mock_joblib, batch_predictor, batch_data):
    """Test batch prediction"""
    # Mock model and preprocessor
    mock_model = Mock()
    mock_model.predict.return_value = np.array([[0.3], [0.7], [0.5], [0.2], [0.8],
                                                  [0.4], [0.6], [0.1], [0.9], [0.35]])
    mock_joblib.return_value = mock_model
    
    mock_prep = Mock()
    mock_prep.preprocess_for_serving.return_value = batch_data
    mock_prep.feature_names = batch_data.columns.tolist()
    mock_preprocessor.return_value = mock_prep
    
    mock_fe = Mock()
    mock_fe.create_features.return_value = batch_data
    mock_feature_eng.return_value = mock_fe
    
    batch_predictor.load_model()
    predictions = batch_predictor.predict_batch(batch_data)
    
    assert isinstance(predictions, (pd.DataFrame, np.ndarray, list))
    assert len(predictions) == len(batch_data)


def test_save_predictions(batch_predictor, batch_data, tmp_path):
    """Test saving batch predictions"""
    predictions = pd.DataFrame({
        'customerID': batch_data['customerID'],
        'churn_probability': np.random.rand(len(batch_data)),
        'churn_prediction': np.random.choice(['Yes', 'No'], len(batch_data))
    })
    
    output_path = tmp_path / "batch_predictions.csv"
    batch_predictor.save_predictions(predictions, str(output_path))
    
    assert output_path.exists()


@patch('src.serving.batch_predict.joblib.load')
@patch('src.serving.batch_predict.DataPreprocessor')
@patch('src.serving.batch_predict.FeatureEngineer')
def test_process_batch_file(mock_feature_eng, mock_preprocessor, mock_joblib, 
                            batch_predictor, batch_data, tmp_path):
    """Test processing batch file"""
    # Save batch data to file
    input_path = tmp_path / "input_batch.csv"
    batch_data.to_csv(input_path, index=False)
    
    # Mock model and preprocessor
    mock_model = Mock()
    mock_model.predict.return_value = np.array([[0.5]] * len(batch_data))
    mock_joblib.return_value = mock_model
    
    mock_prep = Mock()
    mock_prep.preprocess_for_serving.return_value = batch_data
    mock_prep.feature_names = batch_data.columns.tolist()
    mock_preprocessor.return_value = mock_prep
    
    mock_fe = Mock()
    mock_fe.create_features.return_value = batch_data
    mock_feature_eng.return_value = mock_fe
    
    output_path = tmp_path / "output_predictions.csv"
    
    batch_predictor.load_model()
    batch_predictor.process_batch_file(str(input_path), str(output_path))
    
    assert output_path.exists()
