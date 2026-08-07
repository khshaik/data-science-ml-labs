"""
Data quality checks for telco churn dataset
Validates schema, checks for missing values, detects anomalies

Section A: Data & Features (25%) - Data Quality Component
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def _json_default(value):
    """Convert NumPy values used in quality reports to JSON-native values."""
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class DataQualityChecker:
    """
    Data quality validation for churn prediction
    Implements comprehensive checks for schema, missing values, ranges, and duplicates
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.expected_schema = {
            'customerID': 'object',
            'gender': 'object',
            'SeniorCitizen': 'int64',
            'Partner': 'object',
            'Dependents': 'object',
            'tenure': 'int64',
            'PhoneService': 'object',
            'MultipleLines': 'object',
            'InternetService': 'object',
            'OnlineSecurity': 'object',
            'OnlineBackup': 'object',
            'DeviceProtection': 'object',
            'TechSupport': 'object',
            'StreamingTV': 'object',
            'StreamingMovies': 'object',
            'Contract': 'object',
            'PaperlessBilling': 'object',
            'PaymentMethod': 'object',
            'MonthlyCharges': 'float64',
            'TotalCharges': 'object',  # Will convert to float
            'Churn': 'object'
        }
    
    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate dataframe schema against expected schema
        
        Args:
            df: Input dataframe
        
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        # Check column names
        expected_cols = set(self.expected_schema.keys())
        actual_cols = set(df.columns)
        
        missing_cols = expected_cols - actual_cols
        extra_cols = actual_cols - expected_cols
        
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        if extra_cols:
            issues.append(f"Extra columns: {extra_cols}")
        
        # Check data types (for columns that exist)
        for col in expected_cols.intersection(actual_cols):
            if col == 'TotalCharges':
                continue  # Will be converted
            
            expected_dtype = self.expected_schema[col]
            actual_dtype = str(df[col].dtype)
            
            if expected_dtype not in actual_dtype:
                issues.append(f"Column '{col}': expected {expected_dtype}, got {actual_dtype}")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def check_missing_values(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Check for missing values
        
        Args:
            df: Input dataframe
        
        Returns:
            (is_acceptable, missing_stats)
        """
        missing_rate = df.isnull().mean()
        threshold = self.config.get('missing_rate_threshold', 0.05)
        
        high_missing = missing_rate[missing_rate > threshold]
        
        is_acceptable = len(high_missing) == 0
        
        missing_stats = {
            'total_missing_rate': float(missing_rate.mean()),
            'columns_with_high_missing': high_missing.to_dict(),
            'threshold': threshold,
            'per_column_missing_rate': missing_rate.to_dict()
        }
        
        return is_acceptable, missing_stats
    
    def check_data_ranges(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Check if data values are within expected ranges
        
        Args:
            df: Input dataframe
        
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        # Numeric range checks
        if 'tenure' in df.columns:
            if (df['tenure'] < 0).any():
                count = (df['tenure'] < 0).sum()
                issues.append(f"Negative tenure values found: {count} rows")
            if (df['tenure'] > 200).any():
                count = (df['tenure'] > 200).sum()
                issues.append(f"Tenure > 200 months (unrealistic): {count} rows")
        
        if 'MonthlyCharges' in df.columns:
            if (df['MonthlyCharges'] < 0).any():
                count = (df['MonthlyCharges'] < 0).sum()
                issues.append(f"Negative MonthlyCharges found: {count} rows")
            if (df['MonthlyCharges'] > 500).any():
                count = (df['MonthlyCharges'] > 500).sum()
                issues.append(f"MonthlyCharges > $500 (check for outliers): {count} rows")
        
        if 'TotalCharges' in df.columns:
            # Convert to numeric
            df_copy = df.copy()
            df_copy['TotalCharges'] = pd.to_numeric(df_copy['TotalCharges'], errors='coerce')
            if (df_copy['TotalCharges'] < 0).any():
                count = (df_copy['TotalCharges'] < 0).sum()
                issues.append(f"Negative TotalCharges found: {count} rows")
        
        # Categorical value checks
        valid_contracts = ['Month-to-month', 'One year', 'Two year']
        if 'Contract' in df.columns:
            invalid_contracts = ~df['Contract'].isin(valid_contracts)
            if invalid_contracts.any():
                invalid_values = df.loc[invalid_contracts, 'Contract'].unique()
                issues.append(f"Invalid contract types: {invalid_values}")
        
        valid_churn = ['Yes', 'No']
        if 'Churn' in df.columns:
            invalid_churn = ~df['Churn'].isin(valid_churn)
            if invalid_churn.any():
                invalid_values = df.loc[invalid_churn, 'Churn'].unique()
                issues.append(f"Invalid churn values: {invalid_values}")
        
        valid_gender = ['Male', 'Female']
        if 'gender' in df.columns:
            invalid_gender = ~df['gender'].isin(valid_gender)
            if invalid_gender.any():
                invalid_values = df.loc[invalid_gender, 'gender'].unique()
                issues.append(f"Invalid gender values: {invalid_values}")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def check_duplicates(self, df: pd.DataFrame) -> Tuple[bool, int]:
        """
        Check for duplicate customer IDs
        
        Args:
            df: Input dataframe
        
        Returns:
            (is_acceptable, num_duplicates)
        """
        if 'customerID' not in df.columns:
            return True, 0
        
        num_duplicates = df['customerID'].duplicated().sum()
        is_acceptable = num_duplicates == 0
        
        return is_acceptable, num_duplicates
    
    def check_data_consistency(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Check for logical consistency in data
        
        Args:
            df: Input dataframe
        
        Returns:
            (is_consistent, list_of_issues)
        """
        issues = []
        
        # Check: TotalCharges should be approximately tenure * MonthlyCharges
        if all(col in df.columns for col in ['tenure', 'MonthlyCharges', 'TotalCharges']):
            df_copy = df.copy()
            df_copy['TotalCharges'] = pd.to_numeric(df_copy['TotalCharges'], errors='coerce')
            
            # For customers with tenure > 0
            mask = df_copy['tenure'] > 0
            expected_total = df_copy.loc[mask, 'tenure'] * df_copy.loc[mask, 'MonthlyCharges']
            actual_total = df_copy.loc[mask, 'TotalCharges']
            
            # Allow 20% deviation (due to promotions, price changes, etc.)
            deviation = np.abs(expected_total - actual_total) / expected_total
            inconsistent = (deviation > 0.2) & (~actual_total.isna())
            
            if inconsistent.any():
                count = inconsistent.sum()
                issues.append(f"TotalCharges inconsistent with tenure*MonthlyCharges: {count} rows (>20% deviation)")
        
        is_consistent = len(issues) == 0
        return is_consistent, issues
    
    def run_all_checks(self, df: pd.DataFrame) -> Dict:
        """
        Run all data quality checks
        
        Args:
            df: Input dataframe
        
        Returns:
            quality_report: Dict with all check results
        """
        logger.info("=" * 60)
        logger.info("RUNNING DATA QUALITY CHECKS")
        logger.info("=" * 60)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'checks': {}
        }
        
        # Schema validation
        logger.info("1. Validating schema...")
        schema_valid, schema_issues = self.validate_schema(df)
        report['checks']['schema'] = {
            'passed': schema_valid,
            'issues': schema_issues
        }
        if schema_valid:
            logger.info("   ✅ Schema validation passed")
        else:
            logger.warning(f"   ⚠️  Schema issues: {schema_issues}")
        
        # Missing values
        logger.info("2. Checking missing values...")
        missing_acceptable, missing_stats = self.check_missing_values(df)
        report['checks']['missing_values'] = {
            'passed': missing_acceptable,
            'stats': missing_stats
        }
        if missing_acceptable:
            logger.info("   ✅ Missing values within acceptable range")
        else:
            logger.warning(f"   ⚠️  High missing rates: {missing_stats['columns_with_high_missing']}")
        
        # Data ranges
        logger.info("3. Checking data ranges...")
        ranges_valid, range_issues = self.check_data_ranges(df)
        report['checks']['data_ranges'] = {
            'passed': ranges_valid,
            'issues': range_issues
        }
        if ranges_valid:
            logger.info("   ✅ All data values within expected ranges")
        else:
            logger.warning(f"   ⚠️  Range issues: {range_issues}")
        
        # Duplicates
        logger.info("4. Checking for duplicates...")
        no_duplicates, num_dups = self.check_duplicates(df)
        report['checks']['duplicates'] = {
            'passed': no_duplicates,
            'num_duplicates': num_dups
        }
        if no_duplicates:
            logger.info("   ✅ No duplicate customer IDs found")
        else:
            logger.warning(f"   ⚠️  Found {num_dups} duplicate customer IDs")
        
        # Data consistency
        logger.info("5. Checking data consistency...")
        consistent, consistency_issues = self.check_data_consistency(df)
        consistency_is_blocking = self.config.get('consistency_is_blocking', False)
        report['checks']['consistency'] = {
            'passed': bool(consistent or not consistency_is_blocking),
            'warning': bool(not consistent),
            'blocking': bool(consistency_is_blocking),
            'issues': consistency_issues
        }
        if consistent:
            logger.info("   ✅ Data consistency checks passed")
        else:
            logger.warning(
                "   ⚠️  Consistency warning (non-blocking unless configured): "
                f"{consistency_issues}"
            )
        
        # Overall pass/fail
        all_passed = all(
            check['passed'] for check in report['checks'].values()
        )
        report['overall_passed'] = all_passed
        
        logger.info("=" * 60)
        if all_passed:
            logger.info("✅ ALL DATA QUALITY CHECKS PASSED!")
        else:
            logger.warning("⚠️  SOME DATA QUALITY CHECKS FAILED")
            failed_checks = [name for name, check in report['checks'].items() if not check['passed']]
            logger.warning(f"   Failed checks: {failed_checks}")
        logger.info("=" * 60)
        
        return report


def main():
    """
    Main function to run data quality checks
    """
    import yaml
    import json
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    config_path = Path('config/config.yaml')
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    data_path = Path(config['data']['raw_path'])
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return
    
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Run checks
    checker = DataQualityChecker(config['monitoring'])
    report = checker.run_all_checks(df)
    
    # Save report
    report_dir = Path('artifacts/logs')
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / f"data_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=_json_default)
    
    logger.info(f"Report saved to {report_file}")
    
    # Exit with appropriate code
    if report['overall_passed']:
        logger.info("✅ Data quality validation successful")
        return 0
    else:
        logger.error("❌ Data quality validation failed")
        return 1


if __name__ == "__main__":
    exit(main())
