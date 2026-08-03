"""
Unit tests for data ingestion
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.ingestion import DataIngestion


@pytest.fixture
def data_ingestion(base_config, tmp_path):
    """Create DataIngestion instance with temp directory"""
    config = base_config.copy()
    config['data_path'] = str(tmp_path / "test_data.csv")
    return DataIngestion(config)


@pytest.fixture
def sample_csv_file(tmp_path):
    """Create a sample CSV file"""
    data = pd.DataFrame({
        'customerID': ['C001', 'C002', 'C003'],
        'gender': ['Male', 'Female', 'Male'],
        'SeniorCitizen': [0, 1, 0],
        'Partner': ['Yes', 'No', 'Yes'],
        'Dependents': ['No', 'No', 'Yes'],
        'tenure': [12, 24, 36],
        'PhoneService': ['Yes', 'Yes', 'No'],
        'MultipleLines': ['No', 'Yes', 'No phone service'],
        'InternetService': ['DSL', 'Fiber optic', 'DSL'],
        'OnlineSecurity': ['Yes', 'No', 'No'],
        'OnlineBackup': ['No', 'Yes', 'No'],
        'DeviceProtection': ['No', 'No', 'No'],
        'TechSupport': ['Yes', 'No', 'No'],
        'StreamingTV': ['No', 'Yes', 'No'],
        'StreamingMovies': ['No', 'Yes', 'No'],
        'Contract': ['One year', 'Month-to-month', 'Two year'],
        'PaperlessBilling': ['Yes', 'Yes', 'No'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer'],
        'MonthlyCharges': [50.0, 80.0, 60.0],
        'TotalCharges': ['600.0', '1920.0', '2160.0'],
        'Churn': ['No', 'Yes', 'No']
    })
    
    csv_path = tmp_path / "test_data.csv"
    data.to_csv(csv_path, index=False)
    return csv_path


def test_data_ingestion_initialization(data_ingestion):
    """Test DataIngestion initialization"""
    assert data_ingestion is not None
    assert data_ingestion.config is not None


def test_load_data(data_ingestion, sample_csv_file):
    """Test loading data from CSV"""
    data_ingestion.config['data_path'] = str(sample_csv_file)
    df = data_ingestion.load_data()
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert 'customerID' in df.columns
    assert 'Churn' in df.columns


def test_validate_schema(data_ingestion, sample_csv_file):
    """Test schema validation"""
    data_ingestion.config['data_path'] = str(sample_csv_file)
    df = data_ingestion.load_data()
    
    # Should not raise exception for valid schema
    is_valid = data_ingestion.validate_schema(df)
    assert is_valid is True


def test_validate_schema_missing_column(data_ingestion, tmp_path):
    """Test schema validation with missing column"""
    # Create data with missing column
    data = pd.DataFrame({
        'customerID': ['C001', 'C002'],
        'gender': ['Male', 'Female']
        # Missing other required columns
    })
    
    csv_path = tmp_path / "invalid_data.csv"
    data.to_csv(csv_path, index=False)
    
    data_ingestion.config['data_path'] = str(csv_path)
    df = data_ingestion.load_data()
    
    is_valid = data_ingestion.validate_schema(df)
    assert is_valid is False


def test_get_data_summary(data_ingestion, sample_csv_file):
    """Test getting data summary statistics"""
    data_ingestion.config['data_path'] = str(sample_csv_file)
    df = data_ingestion.load_data()
    
    summary = data_ingestion.get_data_summary(df)
    
    assert isinstance(summary, dict)
    assert 'num_rows' in summary
    assert 'num_columns' in summary
    assert summary['num_rows'] == 3
    assert summary['num_columns'] > 0


def test_check_missing_values(data_ingestion, tmp_path):
    """Test checking for missing values"""
    # Create data with missing values
    data = pd.DataFrame({
        'customerID': ['C001', 'C002', 'C003'],
        'gender': ['Male', None, 'Male'],
        'tenure': [12, 24, None],
        'Churn': ['No', 'Yes', 'No']
    })
    
    csv_path = tmp_path / "data_with_missing.csv"
    data.to_csv(csv_path, index=False)
    
    data_ingestion.config['data_path'] = str(csv_path)
    df = data_ingestion.load_data()
    
    missing_report = data_ingestion.check_missing_values(df)
    
    assert isinstance(missing_report, dict)
    assert len(missing_report) > 0


def test_save_data(data_ingestion, sample_csv_file, tmp_path):
    """Test saving data to file"""
    data_ingestion.config['data_path'] = str(sample_csv_file)
    df = data_ingestion.load_data()
    
    output_path = tmp_path / "output_data.csv"
    data_ingestion.save_data(df, str(output_path))
    
    assert output_path.exists()
    
    # Verify saved data
    loaded_df = pd.read_csv(output_path)
    assert len(loaded_df) == len(df)
