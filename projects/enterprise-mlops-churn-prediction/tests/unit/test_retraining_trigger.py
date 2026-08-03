"""
Unit tests for retraining trigger
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
from src.retraining.trigger import RetrainingTrigger


@pytest.fixture
def retraining_trigger(base_config, tmp_path):
    """Create RetrainingTrigger instance"""
    config = base_config.copy()
    config['retraining'] = {
        'performance_threshold': 0.80,
        'drift_threshold': 0.2,
        'min_samples': 100
    }
    return RetrainingTrigger(config)


def test_retraining_trigger_initialization(retraining_trigger):
    """Test RetrainingTrigger initialization"""
    assert retraining_trigger is not None
    assert retraining_trigger.config is not None


def test_check_performance_degradation(retraining_trigger):
    """Test checking for performance degradation"""
    current_metrics = {'accuracy': 0.75, 'auc': 0.78}
    baseline_metrics = {'accuracy': 0.85, 'auc': 0.90}
    
    needs_retraining = retraining_trigger.check_performance_degradation(
        current_metrics, baseline_metrics
    )
    
    assert isinstance(needs_retraining, bool)
    assert needs_retraining is True  # Performance dropped significantly


def test_check_performance_no_degradation(retraining_trigger):
    """Test when performance is still good"""
    current_metrics = {'accuracy': 0.84, 'auc': 0.89}
    baseline_metrics = {'accuracy': 0.85, 'auc': 0.90}
    
    needs_retraining = retraining_trigger.check_performance_degradation(
        current_metrics, baseline_metrics
    )
    
    assert isinstance(needs_retraining, bool)
    assert needs_retraining is False  # Performance is still acceptable


def test_check_data_drift(retraining_trigger):
    """Test checking for data drift"""
    drift_report = {
        'features_with_drift': 5,
        'drift_rate': 0.25,
        'features_checked': 20
    }
    
    needs_retraining = retraining_trigger.check_data_drift(drift_report)
    
    assert isinstance(needs_retraining, bool)
    assert needs_retraining is True  # Drift rate exceeds threshold


def test_check_data_drift_no_drift(retraining_trigger):
    """Test when there's no significant drift"""
    drift_report = {
        'features_with_drift': 2,
        'drift_rate': 0.1,
        'features_checked': 20
    }
    
    needs_retraining = retraining_trigger.check_data_drift(drift_report)
    
    assert isinstance(needs_retraining, bool)
    assert needs_retraining is False  # Drift rate is acceptable


def test_check_sample_size(retraining_trigger):
    """Test checking if enough samples for retraining"""
    has_enough = retraining_trigger.check_sample_size(150)
    assert has_enough is True
    
    has_enough = retraining_trigger.check_sample_size(50)
    assert has_enough is False


def test_should_trigger_retraining_performance(retraining_trigger):
    """Test retraining trigger based on performance"""
    current_metrics = {'accuracy': 0.75, 'auc': 0.78}
    baseline_metrics = {'accuracy': 0.85, 'auc': 0.90}
    drift_report = {'drift_rate': 0.1}
    sample_count = 150
    
    should_retrain, reasons = retraining_trigger.should_trigger_retraining(
        current_metrics, baseline_metrics, drift_report, sample_count
    )
    
    assert isinstance(should_retrain, bool)
    assert isinstance(reasons, list)
    assert should_retrain is True
    assert len(reasons) > 0


def test_should_trigger_retraining_drift(retraining_trigger):
    """Test retraining trigger based on drift"""
    current_metrics = {'accuracy': 0.84, 'auc': 0.89}
    baseline_metrics = {'accuracy': 0.85, 'auc': 0.90}
    drift_report = {'drift_rate': 0.25, 'features_with_drift': 5}
    sample_count = 150
    
    should_retrain, reasons = retraining_trigger.should_trigger_retraining(
        current_metrics, baseline_metrics, drift_report, sample_count
    )
    
    assert isinstance(should_retrain, bool)
    assert isinstance(reasons, list)
    assert should_retrain is True


def test_should_not_trigger_retraining(retraining_trigger):
    """Test when retraining should not be triggered"""
    current_metrics = {'accuracy': 0.84, 'auc': 0.89}
    baseline_metrics = {'accuracy': 0.85, 'auc': 0.90}
    drift_report = {'drift_rate': 0.1}
    sample_count = 150
    
    should_retrain, reasons = retraining_trigger.should_trigger_retraining(
        current_metrics, baseline_metrics, drift_report, sample_count
    )
    
    assert isinstance(should_retrain, bool)
    assert should_retrain is False


def test_log_retraining_decision(retraining_trigger, tmp_path):
    """Test logging retraining decision"""
    decision = {
        'should_retrain': True,
        'reasons': ['Performance degradation', 'Data drift detected'],
        'timestamp': '2024-01-01 12:00:00'
    }
    
    log_path = tmp_path / "retraining_log.json"
    retraining_trigger.log_decision(decision, str(log_path))
    
    assert log_path.exists()
