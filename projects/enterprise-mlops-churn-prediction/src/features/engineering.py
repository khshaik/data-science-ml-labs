"""
Shared feature engineering module
Used by both training (offline) and serving (online) to prevent training-serving skew

Section A: Data & Features (25%) - Feature Engineering Component

Key Requirements:
1. At least 5 non-trivial features (aggregations, ratios, encodings, time-window features, etc.)
2. Document which features are offline vs online
3. Show awareness of training-serving skew
4. Explain how to ensure consistent features between training and serving
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering for churn prediction
    Ensures consistency between training and serving
    
    Implements 6 non-trivial engineered features:
    1. avg_monthly_charge (aggregation)
    2. service_adoption_score (aggregation)
    3. tenure_category (binning)
    4. payment_risk_flag (domain encoding)
    5. contract_stability_score (ordinal encoding)
    6. high_value_customer (threshold)
    """
    
    def __init__(self, config_path: str = 'config/feature_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.feature_definitions = self.config['features']['engineered']
        self.high_value_threshold = None  # Will be set from training data
    
    def create_features(self, df: pd.DataFrame, mode: str = 'offline') -> pd.DataFrame:
        """
        Create all engineered features
        
        Args:
            df: Input dataframe
            mode: 'offline' for training, 'online' for serving
        
        Returns:
            df_with_features: Dataframe with engineered features
        """
        logger.info(f"Creating features in {mode} mode...")
        
        df = df.copy()
        
        # Convert TotalCharges to numeric (handle empty strings from original data)
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'] = df['TotalCharges'].fillna(0)
        
        # Feature 1: avg_monthly_charge (aggregation)
        df['avg_monthly_charge'] = self._create_avg_monthly_charge(df)
        
        # Feature 2: service_adoption_score (aggregation)
        df['service_adoption_score'] = self._create_service_adoption_score(df)
        
        # Feature 3: tenure_category (binning)
        df['tenure_category'] = self._create_tenure_category(df)
        
        # Feature 4: payment_risk_flag (encoding)
        df['payment_risk_flag'] = self._create_payment_risk_flag(df)
        
        # Feature 5: contract_stability_score (ordinal encoding)
        df['contract_stability_score'] = self._create_contract_stability_score(df)
        
        # Feature 6: high_value_customer (threshold)
        df['high_value_customer'] = self._create_high_value_customer(df, mode)
        
        logger.info(f"✅ Created {len(self.feature_definitions)} engineered features")
        
        return df
    
    def _create_avg_monthly_charge(self, df: pd.DataFrame) -> pd.Series:
        """
        Feature 1: Average monthly charge over customer lifetime
        Formula: TotalCharges / tenure
        
        Training-Serving Skew Risk: Medium
        - Training: Uses historical TotalCharges from data warehouse
        - Serving: Must calculate from real-time data
        
        Mitigation: Shared calculation logic in this module
        """
        # Handle edge case: tenure = 0 (new customers)
        avg_charge = np.where(
            df['tenure'] > 0,
            df['TotalCharges'] / df['tenure'],
            df['MonthlyCharges']  # Use current charge if tenure = 0
        )
        return pd.Series(avg_charge, index=df.index)
    
    def _create_service_adoption_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Feature 2: Number of add-on services adopted
        Count: OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
        
        Training-Serving Skew Risk: Low
        - Simple count, consistent across environments
        
        From original notebook: Customers with more services tend to have lower churn
        """
        service_columns = [
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
        # Count 'Yes' values
        score = 0
        for col in service_columns:
            if col in df.columns:
                score += (df[col] == 'Yes').astype(int)
        
        return score
    
    def _create_tenure_category(self, df: pd.DataFrame) -> pd.Series:
        """
        Feature 3: Tenure bucketed into meaningful categories
        Bins: 0-12, 13-24, 25-48, 48+ months
        
        Training-Serving Skew Risk: Low
        - Fixed bin boundaries, deterministic
        
        From original notebook: Churn is highest in first 12 months
        """
        bins = [0, 12, 24, 48, np.inf]
        labels = ['0-12m', '13-24m', '25-48m', '48m+']
        
        return pd.cut(df['tenure'], bins=bins, labels=labels, include_lowest=True)
    
    def _create_payment_risk_flag(self, df: pd.DataFrame) -> pd.Series:
        """
        Feature 4: High-risk payment method indicator
        1 if Electronic check, 0 otherwise
        
        Training-Serving Skew Risk: Low
        - Simple boolean logic, no dependencies
        
        From original notebook: Electronic check has highest churn rate
        """
        return (df['PaymentMethod'] == 'Electronic check').astype(int)
    
    def _create_contract_stability_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Feature 5: Contract commitment level
        Month-to-month: 1, One year: 2, Two year: 3
        
        Training-Serving Skew Risk: Low
        - Fixed mapping, deterministic
        
        From original notebook: Month-to-month has highest churn, Two year has lowest
        """
        mapping = {
            'Month-to-month': 1,
            'One year': 2,
            'Two year': 3
        }
        return df['Contract'].map(mapping).fillna(1)
    
    def _create_high_value_customer(self, df: pd.DataFrame, mode: str) -> pd.Series:
        """
        Feature 6: High monthly charges indicator
        1 if MonthlyCharges > 75th percentile, 0 otherwise
        
        Training-Serving Skew Risk: Medium
        - Training: Calculate percentile from training data
        - Serving: Use fixed threshold from training
        
        Mitigation: Store training set percentile and use as fixed threshold
        """
        if mode == 'offline':
            # Training mode: calculate and store threshold
            self.high_value_threshold = df['MonthlyCharges'].quantile(0.75)
            logger.info(f"High value threshold (75th percentile): ${self.high_value_threshold:.2f}")
        else:
            # Serving mode: use stored threshold
            if self.high_value_threshold is None:
                # Fallback: use a reasonable default
                self.high_value_threshold = 70.0
                logger.warning(f"Using default high value threshold: ${self.high_value_threshold:.2f}")
        
        return (df['MonthlyCharges'] > self.high_value_threshold).astype(int)
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature names (original + engineered)
        """
        original_features = self.config['features']['original']
        engineered_features = [f['name'] for f in self.feature_definitions]
        drop_features = self.config['features']['drop']
        
        all_features = [f for f in original_features if f not in drop_features] + engineered_features
        
        # Remove target
        target = self.config['features']['target']
        if target in all_features:
            all_features.remove(target)
        
        return all_features
    
    def get_engineered_feature_names(self) -> List[str]:
        """
        Get list of engineered feature names only
        """
        return [f['name'] for f in self.feature_definitions]
    
    def save_threshold(self, filepath: str):
        """
        Save high value threshold for serving
        """
        import json
        threshold_data = {
            'high_value_threshold': self.high_value_threshold,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        with open(filepath, 'w') as f:
            json.dump(threshold_data, f, indent=2)
        logger.info(f"Saved threshold to {filepath}")
    
    def load_threshold(self, filepath: str):
        """
        Load high value threshold for serving
        """
        import json
        with open(filepath, 'r') as f:
            threshold_data = json.load(f)
        self.high_value_threshold = threshold_data['high_value_threshold']
        logger.info(f"Loaded threshold from {filepath}: ${self.high_value_threshold:.2f}")


def main():
    """
    Test feature engineering
    """
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    df = pd.read_csv('data/raw/telco_customer_churn.csv')
    logger.info(f"Loaded {len(df)} rows")
    
    # Create features
    engineer = FeatureEngineer()
    df_with_features = engineer.create_features(df, mode='offline')
    
    # Display engineered features
    engineered_cols = engineer.get_engineered_feature_names()
    print("\n" + "=" * 80)
    print("ENGINEERED FEATURES SAMPLE")
    print("=" * 80)
    print(df_with_features[engineered_cols].head(10))
    
    print("\n" + "=" * 80)
    print("ENGINEERED FEATURES STATISTICS")
    print("=" * 80)
    print(df_with_features[engineered_cols].describe())
    
    print("\n" + "=" * 80)
    print("FEATURE CORRELATION WITH CHURN")
    print("=" * 80)
    # Convert Churn to binary
    df_with_features['Churn_binary'] = (df_with_features['Churn'] == 'Yes').astype(int)
    correlations = df_with_features[engineered_cols + ['Churn_binary']].corr()['Churn_binary'].drop('Churn_binary')
    print(correlations.sort_values(ascending=False))
    
    # Save threshold
    engineer.save_threshold('artifacts/feature_threshold.json')
    
    print("\n✅ Feature engineering test completed successfully!")


if __name__ == "__main__":
    main()
