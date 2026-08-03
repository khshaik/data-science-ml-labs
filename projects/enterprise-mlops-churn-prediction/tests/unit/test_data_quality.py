"""
Unit tests for data quality checks
Bonus Feature: Comprehensive Testing (+0.5 mark)
"""

import pytest
import pandas as pd
import numpy as np
from src.data.quality import DataQualityChecker


@pytest.fixture
def valid_data():
    """Create valid customer data"""
    return pd.DataFrame({
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


@pytest.fixture
def config():
    """Create test configuration"""
    return {
        'missing_rate_threshold': 0.05
    }


def test_data_quality_checker_initialization(config):
    """Test DataQualityChecker initialization"""
    checker = DataQualityChecker(config)
    assert checker is not None
    assert checker.config == config


def test_validate_schema_valid(valid_data, config):
    """Test schema validation with valid data"""
    checker = DataQualityChecker(config)
    is_valid, issues = checker.validate_schema(valid_data)
    
    assert is_valid
    assert len(issues) == 0


def test_validate_schema_missing_column(valid_data, config):
    """Test schema validation with missing column"""
    checker = DataQualityChecker(config)
    invalid_data = valid_data.drop('tenure', axis=1)
    
    is_valid, issues = checker.validate_schema(invalid_data)
    
    assert not is_valid
    assert len(issues) > 0
    assert any('Missing columns' in issue for issue in issues)


def test_check_missing_values_acceptable(valid_data, config):
    """Test missing value check with acceptable data"""
    checker = DataQualityChecker(config)
    is_acceptable, stats = checker.check_missing_values(valid_data)
    
    assert is_acceptable
    assert stats['total_missing_rate'] == 0.0


def test_check_missing_values_high(valid_data, config):
    """Test missing value check with high missing rate"""
    checker = DataQualityChecker(config)
    
    # Add missing values
    data_with_missing = valid_data.copy()
    data_with_missing.loc[0, 'MonthlyCharges'] = np.nan
    data_with_missing.loc[1, 'MonthlyCharges'] = np.nan
    
    is_acceptable, stats = checker.check_missing_values(data_with_missing)
    
    assert not is_acceptable
    assert 'MonthlyCharges' in stats['columns_with_high_missing']


def test_check_data_ranges_valid(valid_data, config):
    """Test data range check with valid data"""
    checker = DataQualityChecker(config)
    is_valid, issues = checker.check_data_ranges(valid_data)
    
    assert is_valid
    assert len(issues) == 0


def test_check_data_ranges_negative_tenure(valid_data, config):
    """Test data range check with negative tenure"""
    checker = DataQualityChecker(config)
    
    invalid_data = valid_data.copy()
    invalid_data.loc[0, 'tenure'] = -5
    
    is_valid, issues = checker.check_data_ranges(invalid_data)
    
    assert not is_valid
    assert any('Negative tenure' in issue for issue in issues)


def test_check_duplicates_none(valid_data, config):
    """Test duplicate check with no duplicates"""
    checker = DataQualityChecker(config)
    is_acceptable, num_dups = checker.check_duplicates(valid_data)
    
    assert is_acceptable
    assert num_dups == 0


def test_check_duplicates_present(valid_data, config):
    """Test duplicate check with duplicates"""
    checker = DataQualityChecker(config)
    
    # Add duplicate
    data_with_dup = pd.concat([valid_data, valid_data.iloc[[0]]], ignore_index=True)
    
    is_acceptable, num_dups = checker.check_duplicates(data_with_dup)
    
    assert not is_acceptable
    assert num_dups == 1


def test_run_all_checks_valid(valid_data, config):
    """Test running all checks with valid data"""
    checker = DataQualityChecker(config)
    report = checker.run_all_checks(valid_data)
    
    assert report['overall_passed']
    assert report['num_rows'] == 3
    assert all(check['passed'] for check in report['checks'].values())
