"""
Unit tests for feature engineering
Bonus Feature: Comprehensive Testing (+0.5 mark)
"""

import pytest
import pandas as pd
import numpy as np
from src.features.engineering import FeatureEngineer


@pytest.fixture
def sample_data():
    """Create sample customer data for testing"""
    return pd.DataFrame({
        'customerID': ['C001', 'C002', 'C003'],
        'gender': ['Male', 'Female', 'Male'],
        'SeniorCitizen': [0, 1, 0],
        'Partner': ['Yes', 'No', 'Yes'],
        'Dependents': ['No', 'No', 'Yes'],
        'tenure': [12, 24, 1],
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
        'MonthlyCharges': [50.0, 80.0, 30.0],
        'TotalCharges': [600.0, 1920.0, 30.0],
        'Churn': ['No', 'Yes', 'No']
    })


def test_feature_engineer_initialization():
    """Test FeatureEngineer initialization"""
    engineer = FeatureEngineer()
    assert engineer is not None
    assert len(engineer.feature_definitions) == 6


def test_avg_monthly_charge(sample_data):
    """Test avg_monthly_charge feature"""
    engineer = FeatureEngineer()
    result = engineer._create_avg_monthly_charge(sample_data)
    
    # Check calculations
    assert result[0] == pytest.approx(600.0 / 12, rel=1e-2)  # 50.0
    assert result[1] == pytest.approx(1920.0 / 24, rel=1e-2)  # 80.0
    assert result[2] == pytest.approx(30.0 / 1, rel=1e-2)  # 30.0


def test_service_adoption_score(sample_data):
    """Test service_adoption_score feature"""
    engineer = FeatureEngineer()
    result = engineer._create_service_adoption_score(sample_data)
    
    # Customer 1: OnlineSecurity=Yes, TechSupport=Yes = 2 services
    assert result[0] == 2
    # Customer 2: OnlineBackup=Yes, StreamingTV=Yes, StreamingMovies=Yes = 3 services
    assert result[1] == 3
    # Customer 3: No services = 0
    assert result[2] == 0


def test_tenure_category(sample_data):
    """Test tenure_category feature"""
    engineer = FeatureEngineer()
    result = engineer._create_tenure_category(sample_data)
    
    assert result[0] == '0-12m'  # tenure=12
    assert result[1] == '13-24m'  # tenure=24
    assert result[2] == '0-12m'  # tenure=1


def test_payment_risk_flag(sample_data):
    """Test payment_risk_flag feature"""
    engineer = FeatureEngineer()
    result = engineer._create_payment_risk_flag(sample_data)
    
    assert result[0] == 1  # Electronic check
    assert result[1] == 0  # Mailed check
    assert result[2] == 0  # Bank transfer


def test_contract_stability_score(sample_data):
    """Test contract_stability_score feature"""
    engineer = FeatureEngineer()
    result = engineer._create_contract_stability_score(sample_data)
    
    assert result[0] == 2  # One year
    assert result[1] == 1  # Month-to-month
    assert result[2] == 3  # Two year


def test_create_features_offline(sample_data):
    """Test complete feature creation in offline mode"""
    engineer = FeatureEngineer()
    result = engineer.create_features(sample_data, mode='offline')
    
    # Check all engineered features are created
    assert 'avg_monthly_charge' in result.columns
    assert 'service_adoption_score' in result.columns
    assert 'tenure_category' in result.columns
    assert 'payment_risk_flag' in result.columns
    assert 'contract_stability_score' in result.columns
    assert 'high_value_customer' in result.columns
    
    # Check no NaN values in engineered features
    assert not result['avg_monthly_charge'].isna().any()
    assert not result['service_adoption_score'].isna().any()


def test_create_features_online(sample_data):
    """Test feature creation in online mode"""
    engineer = FeatureEngineer()
    engineer.high_value_threshold = 70.0  # Set threshold
    
    result = engineer.create_features(sample_data, mode='online')
    
    # Should create same features as offline
    assert 'avg_monthly_charge' in result.columns
    assert 'high_value_customer' in result.columns


def test_training_serving_consistency(sample_data):
    """Test that offline and online modes produce same results"""
    engineer = FeatureEngineer()
    
    # Offline mode
    offline_result = engineer.create_features(sample_data.copy(), mode='offline')
    threshold = engineer.high_value_threshold
    
    # Online mode with same threshold
    engineer_online = FeatureEngineer()
    engineer_online.high_value_threshold = threshold
    online_result = engineer_online.create_features(sample_data.copy(), mode='online')
    
    # Check key features match
    pd.testing.assert_series_equal(
        offline_result['avg_monthly_charge'],
        online_result['avg_monthly_charge'],
        check_names=False
    )
    pd.testing.assert_series_equal(
        offline_result['service_adoption_score'],
        online_result['service_adoption_score'],
        check_names=False
    )
