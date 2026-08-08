"""
Data preprocessing module
Handles data cleaning, encoding, and splitting

Leverages preprocessing logic from original notebook
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
from typing import Tuple, Dict
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Data preprocessing for churn prediction
    Based on preprocessing from original notebook
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw data
        Based on original notebook cleaning steps
        """
        df = df.copy()
        
        # Convert TotalCharges to numeric (from original notebook)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Fill missing TotalCharges with 0 (new customers)
        df['TotalCharges'] = df['TotalCharges'].fillna(0)
        
        # Remove customerID (not predictive)
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        
        logger.info(f"✅ Data cleaned: {len(df)} rows")
        return df
    
    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical variables with fitted integer mappings.

        Binary and multi-class input columns each retain a training-fitted
        LabelEncoder. During serving, an unseen multi-class value maps to that
        encoder's first known class for compatibility with the retained model;
        unseen binary values remain strict and raise from LabelEncoder. This is
        not one-hot encoding.
        """
        df = df.copy()
        
        # Binary categorical features (from original notebook)
        binary_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                          'PaperlessBilling']
        
        for col in binary_features:
            if col in df.columns:
                if fit:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        df[col] = self.label_encoders[col].transform(df[col])
        
        # Multi-class categorical features use fitted integer encoding. This
        # preserves the retained model's 25-column input contract; it is not
        # one-hot encoding.
        multi_class_features = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                               'OnlineBackup', 'DeviceProtection', 'TechSupport',
                               'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
        
        # Also encode engineered categorical feature
        if 'tenure_category' in df.columns:
            multi_class_features.append('tenure_category')
        
        for col in multi_class_features:
            if col in df.columns:
                if fit:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        # Retained-model compatibility policy: map an unseen
                        # multi-class value to the encoder's first known class.
                        df[col] = df[col].astype(str).apply(
                            lambda x: x if x in self.label_encoders[col].classes_ else self.label_encoders[col].classes_[0]
                        )
                        df[col] = self.label_encoders[col].transform(df[col])
        
        logger.info(f"✅ Encoded {len(binary_features) + len(multi_class_features)} categorical features")
        return df
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features
        Based on original notebook scaling strategy
        """
        df = df.copy()
        
        # Numerical features to scale (from original notebook)
        numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        
        # Add engineered numerical features
        engineered_numerical = ['avg_monthly_charge', 'service_adoption_score', 
                               'contract_stability_score']
        
        scale_features = [f for f in numerical_features + engineered_numerical if f in df.columns]
        
        if fit:
            df[scale_features] = self.scaler.fit_transform(df[scale_features])
        else:
            df[scale_features] = self.scaler.transform(df[scale_features])
        
        logger.info(f"✅ Scaled {len(scale_features)} numerical features")
        return df
    
    def split_data(self, df: pd.DataFrame, target_col: str = 'Churn') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data into train/val/test sets
        Based on configuration
        """
        # Separate features and target
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Convert target to binary (from original notebook)
        y = (y == 'Yes').astype(int)
        
        # Get split ratios from config
        train_size = self.config['data']['train_split']
        val_size = self.config['data']['val_split']
        test_size = self.config['data']['test_split']
        random_state = self.config['data']['random_seed']
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        # Second split: train vs val
        val_ratio = val_size / (train_size + val_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=random_state,
            stratify=y_temp
        )
        
        logger.info(f"✅ Data split:")
        logger.info(f"   Train: {len(X_train)} rows ({train_size*100:.0f}%)")
        logger.info(f"   Val:   {len(X_val)} rows ({val_size*100:.0f}%)")
        logger.info(f"   Test:  {len(X_test)} rows ({test_size*100:.0f}%)")
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_preprocessor(self, filepath: str):
        """
        Save preprocessor (encoders, scaler) for serving
        """
        preprocessor_data = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor_data, filepath)
        logger.info(f"✅ Saved preprocessor to {filepath}")
    
    def load_preprocessor(self, filepath: str):
        """
        Load preprocessor for serving
        """
        preprocessor_data = joblib.load(filepath)
        self.label_encoders = preprocessor_data['label_encoders']
        self.scaler = preprocessor_data['scaler']
        self.feature_names = preprocessor_data['feature_names']
        logger.info(f"✅ Loaded preprocessor from {filepath}")
    
    def preprocess_for_training(self, df: pd.DataFrame) -> Tuple:
        """
        Full preprocessing pipeline for training
        """
        logger.info("Starting preprocessing for training...")
        
        # Clean
        df = self.clean_data(df)
        
        # Split first (to avoid data leakage)
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(df)
        
        # Encode categorical (fit on train)
        X_train = self.encode_categorical(X_train, fit=True)
        X_val = self.encode_categorical(X_val, fit=False)
        X_test = self.encode_categorical(X_test, fit=False)
        
        # Scale numerical (fit on train)
        X_train = self.scale_features(X_train, fit=True)
        X_val = self.scale_features(X_val, fit=False)
        X_test = self.scale_features(X_test, fit=False)
        
        logger.info("✅ Preprocessing completed")
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def preprocess_feature_splits(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fit preprocessing on a training split and transform held-out splits.

        Feature engineering that learns values from data (for example a
        percentile threshold) must happen after the raw split. This method
        lets the training pipeline preserve that ordering while still fitting
        encoders and the scaler only on training rows.
        """
        X_train = self.clean_data(X_train)
        X_val = self.clean_data(X_val)
        X_test = self.clean_data(X_test)

        X_train = self.encode_categorical(X_train, fit=True)
        X_val = self.encode_categorical(X_val, fit=False)
        X_test = self.encode_categorical(X_test, fit=False)

        X_train = self.scale_features(X_train, fit=True)
        X_val = self.scale_features(X_val, fit=False)
        X_test = self.scale_features(X_test, fit=False)

        self.feature_names = X_train.columns.tolist()
        X_val = X_val.reindex(columns=self.feature_names)
        X_test = X_test.reindex(columns=self.feature_names)
        return X_train, X_val, X_test
    
    def preprocess_for_serving(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocessing pipeline for serving (inference)
        """
        # Clean
        df = self.clean_data(df)
        
        # Encode categorical (use fitted encoders)
        df = self.encode_categorical(df, fit=False)
        
        # Scale numerical (use fitted scaler)
        df = self.scale_features(df, fit=False)
        
        return df


def main():
    """
    Test preprocessing
    """
    import yaml
    
    logging.basicConfig(level=logging.INFO)
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data with features
    from src.features.engineering import FeatureEngineer
    
    df = pd.read_csv('data/raw/telco_customer_churn.csv')
    
    # Create features
    engineer = FeatureEngineer()
    df = engineer.create_features(df, mode='offline')
    
    # Preprocess
    preprocessor = DataPreprocessor(config)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_for_training(df)
    
    print("\n" + "=" * 80)
    print("PREPROCESSING RESULTS")
    print("=" * 80)
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"\nTarget distribution (train):")
    print(f"  No churn (0): {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.1f}%)")
    print(f"  Churn (1): {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")
    
    # Save preprocessor
    preprocessor.save_preprocessor('artifacts/preprocessor.pkl')
    
    print("\n✅ Preprocessing test completed successfully!")


if __name__ == "__main__":
    main()
