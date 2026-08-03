"""
Drift detection module
Implements statistical tests for data and feature drift

Section D: Monitoring, Data Quality and Retraining Trigger (25%) - Drift Detection
"""

import pandas as pd
import numpy as np
from scipy import stats
import logging
from typing import Dict, Tuple
from datetime import datetime
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class DriftDetector:
    """
    Drift detection using statistical tests
    
    Implements:
    1. PSI (Population Stability Index) - for continuous features
    2. KS Test (Kolmogorov-Smirnov) - for continuous features
    3. Chi-Squared Test - for categorical features
    
    From Week 5 concepts
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.psi_threshold = config['monitoring']['drift_threshold_psi']
        self.ks_threshold = config['monitoring']['drift_threshold_ks_pvalue']
        self.baseline_stats = None
    
    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI)
        
        Formula: PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
        
        Thresholds (from Week 5):
        < 0.1: Stable ✅
        0.1-0.2: Watch ⚠️
        > 0.2: Act 🚨
        
        Args:
            expected: Baseline distribution (training data)
            actual: Current distribution (production data)
            bins: Number of bins for discretization
        
        Returns:
            psi: PSI score
        """
        # Create bins based on expected distribution
        breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)  # Remove duplicates
        
        # Calculate percentages in each bin
        expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        expected_percents = np.where(expected_percents == 0, epsilon, expected_percents)
        actual_percents = np.where(actual_percents == 0, epsilon, actual_percents)
        
        # Calculate PSI
        psi = np.sum((actual_percents - expected_percents) * 
                     np.log(actual_percents / expected_percents))
        
        return float(psi)
    
    def ks_test(self, baseline: np.ndarray, current: np.ndarray) -> Tuple[float, float, bool]:
        """
        Kolmogorov-Smirnov test for distribution difference
        
        Formula: D = max |F_ref(x) - F_prod(x)|
        
        Args:
            baseline: Baseline distribution
            current: Current distribution
        
        Returns:
            (statistic, p_value, drift_detected)
        """
        statistic, p_value = stats.ks_2samp(baseline, current)
        
        # Drift detected if p-value < threshold (reject null hypothesis)
        drift_detected = p_value < self.ks_threshold
        
        return float(statistic), float(p_value), drift_detected
    
    def chi_squared_test(self, baseline: pd.Series, current: pd.Series) -> Tuple[float, float, bool]:
        """
        Chi-squared test for categorical features
        
        Args:
            baseline: Baseline categorical distribution
            current: Current categorical distribution
        
        Returns:
            (statistic, p_value, drift_detected)
        """
        # Create contingency table
        baseline_counts = baseline.value_counts()
        current_counts = current.value_counts()
        
        # Align categories
        all_categories = set(baseline_counts.index) | set(current_counts.index)
        baseline_counts = baseline_counts.reindex(all_categories, fill_value=0)
        current_counts = current_counts.reindex(all_categories, fill_value=0)
        
        # Chi-squared test
        contingency_table = pd.DataFrame({
            'baseline': baseline_counts,
            'current': current_counts
        })
        
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table.T)
        
        drift_detected = p_value < self.ks_threshold
        
        return float(chi2), float(p_value), drift_detected
    
    def detect_feature_drift(self, baseline_df: pd.DataFrame, current_df: pd.DataFrame, 
                            feature_name: str, feature_type: str = 'continuous') -> Dict:
        """
        Detect drift for a single feature
        
        Args:
            baseline_df: Baseline data (training)
            current_df: Current data (production)
            feature_name: Name of feature
            feature_type: 'continuous' or 'categorical'
        
        Returns:
            drift_report: Dict with drift metrics
        """
        logger.info(f"Checking drift for feature: {feature_name} ({feature_type})")
        
        baseline_data = baseline_df[feature_name].dropna()
        current_data = current_df[feature_name].dropna()
        
        report = {
            'feature': feature_name,
            'feature_type': feature_type,
            'baseline_size': len(baseline_data),
            'current_size': len(current_data),
            'timestamp': datetime.now().isoformat()
        }
        
        if feature_type == 'continuous':
            # PSI
            psi = self.calculate_psi(baseline_data.values, current_data.values)
            report['psi'] = psi
            report['psi_threshold'] = self.psi_threshold
            report['psi_drift'] = psi > self.psi_threshold
            
            # KS Test
            ks_stat, ks_pvalue, ks_drift = self.ks_test(baseline_data.values, current_data.values)
            report['ks_statistic'] = ks_stat
            report['ks_pvalue'] = ks_pvalue
            report['ks_drift'] = ks_drift
            
            # Basic statistics
            report['baseline_mean'] = float(baseline_data.mean())
            report['current_mean'] = float(current_data.mean())
            report['mean_shift'] = float(current_data.mean() - baseline_data.mean())
            report['mean_shift_pct'] = float((current_data.mean() - baseline_data.mean()) / baseline_data.mean() * 100)
            
            # Overall drift decision
            report['drift_detected'] = report['psi_drift'] or report['ks_drift']
            
            # Log results
            if report['drift_detected']:
                logger.warning(f"  🚨 DRIFT DETECTED: PSI={psi:.4f}, KS p-value={ks_pvalue:.4f}")
            else:
                logger.info(f"  ✅ No drift: PSI={psi:.4f}, KS p-value={ks_pvalue:.4f}")
        
        else:  # categorical
            # Chi-squared test
            chi2, pvalue, drift = self.chi_squared_test(baseline_data, current_data)
            report['chi2_statistic'] = chi2
            report['chi2_pvalue'] = pvalue
            report['drift_detected'] = drift
            
            # Value counts
            report['baseline_distribution'] = baseline_data.value_counts().to_dict()
            report['current_distribution'] = current_data.value_counts().to_dict()
            
            # Log results
            if drift:
                logger.warning(f"  🚨 DRIFT DETECTED: Chi2={chi2:.4f}, p-value={pvalue:.4f}")
            else:
                logger.info(f"  ✅ No drift: Chi2={chi2:.4f}, p-value={pvalue:.4f}")
        
        return report
    
    def detect_dataset_drift(self, baseline_df: pd.DataFrame, current_df: pd.DataFrame, 
                            feature_types: Dict[str, str]) -> Dict:
        """
        Detect drift across entire dataset
        
        Args:
            baseline_df: Baseline data
            current_df: Current data
            feature_types: Dict mapping feature names to types ('continuous' or 'categorical')
        
        Returns:
            drift_report: Complete drift report
        """
        logger.info("=" * 80)
        logger.info("DRIFT DETECTION ANALYSIS")
        logger.info("=" * 80)
        logger.info(f"Baseline data: {len(baseline_df)} rows")
        logger.info(f"Current data: {len(current_df)} rows")
        logger.info(f"Features to check: {len(feature_types)}")
        
        feature_reports = []
        drift_count = 0
        
        for feature_name, feature_type in feature_types.items():
            if feature_name in baseline_df.columns and feature_name in current_df.columns:
                report = self.detect_feature_drift(baseline_df, current_df, feature_name, feature_type)
                feature_reports.append(report)
                
                if report['drift_detected']:
                    drift_count += 1
        
        # Overall report
        overall_report = {
            'timestamp': datetime.now().isoformat(),
            'baseline_size': len(baseline_df),
            'current_size': len(current_df),
            'features_checked': len(feature_reports),
            'features_with_drift': drift_count,
            'drift_rate': drift_count / len(feature_reports) if feature_reports else 0,
            'feature_reports': feature_reports
        }
        
        logger.info("\n" + "=" * 80)
        logger.info("DRIFT DETECTION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Features checked: {overall_report['features_checked']}")
        logger.info(f"Features with drift: {overall_report['features_with_drift']}")
        logger.info(f"Drift rate: {overall_report['drift_rate']*100:.1f}%")
        
        if drift_count > 0:
            logger.warning(f"\n⚠️  DRIFT DETECTED in {drift_count} features!")
            logger.warning("Features with drift:")
            for report in feature_reports:
                if report['drift_detected']:
                    logger.warning(f"  - {report['feature']}")
        else:
            logger.info("\n✅ No significant drift detected")
        
        logger.info("=" * 80)
        
        return overall_report
    
    def save_drift_report(self, report: Dict, output_file: str):
        """
        Save drift report to file
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"✅ Drift report saved to {output_file}")


def main():
    """
    Test drift detection
    """
    import yaml
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    df = pd.read_csv('data/raw/telco_customer_churn.csv')
    
    # Split into baseline and current (simulate drift)
    baseline_df = df.iloc[:5000].copy()
    current_df = df.iloc[5000:].copy()
    
    # Define feature types
    feature_types = {
        'tenure': 'continuous',
        'MonthlyCharges': 'continuous',
        'TotalCharges': 'continuous',
        'Contract': 'categorical',
        'PaymentMethod': 'categorical',
        'InternetService': 'categorical'
    }
    
    # Detect drift
    detector = DriftDetector(config)
    report = detector.detect_dataset_drift(baseline_df, current_df, feature_types)
    
    # Save report
    detector.save_drift_report(report, 'artifacts/drift_reports/drift_report_test.json')
    
    logger.info("\n✅ Drift detection test completed!")


if __name__ == "__main__":
    main()
