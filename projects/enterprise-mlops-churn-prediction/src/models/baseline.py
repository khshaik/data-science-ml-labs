"""
Baseline Model: Logistic Regression
Simple, interpretable model for churn prediction

Section B: Model Training & Offline Evaluation (25%) - Baseline Model
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import numpy as np
import logging
from typing import Dict, Tuple
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class BaselineModel:
    """
    Baseline Logistic Regression model for churn prediction
    
    Justification:
    - Simple and interpretable
    - Fast training and inference
    - Good baseline for comparison
    - Provides probability estimates
    - Works well with binary classification
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.metrics = {}
    
    def build_model(self):
        """
        Build Logistic Regression model
        """
        hyperparams = self.config['models']['baseline']['hyperparameters']
        
        self.model = LogisticRegression(
            C=hyperparams['C'],
            max_iter=hyperparams['max_iter'],
            random_state=hyperparams['random_state'],
            class_weight='balanced'  # Handle class imbalance
        )
        
        logger.info("✅ Baseline model (Logistic Regression) built")
        logger.info(f"   Hyperparameters: C={hyperparams['C']}, max_iter={hyperparams['max_iter']}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
        
        Returns:
            training_metrics: Dict with training metrics
        """
        logger.info("Training baseline model...")
        
        if self.model is None:
            self.build_model()
        
        # Train
        self.model.fit(X_train, y_train)
        
        # Evaluate on training set
        y_train_pred = self.model.predict(X_train)
        y_train_proba = self.model.predict_proba(X_train)[:, 1]
        
        training_metrics = {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred),
            'recall': recall_score(y_train, y_train_pred),
            'f1': f1_score(y_train, y_train_pred),
            'auc': roc_auc_score(y_train, y_train_proba)
        }
        
        logger.info("✅ Training completed")
        logger.info(f"   Training AUC: {training_metrics['auc']:.4f}")
        logger.info(f"   Training Recall: {training_metrics['recall']:.4f}")
        
        return training_metrics
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, dataset_name: str = "test") -> Dict:
        """
        Evaluate the model
        
        Args:
            X: Features
            y: True labels
            dataset_name: Name of dataset (for logging)
        
        Returns:
            metrics: Dict with evaluation metrics
        """
        logger.info(f"Evaluating on {dataset_name} set...")
        
        # Predictions
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]
        
        # Calculate metrics
        metrics = {
            'dataset': dataset_name,
            'accuracy': float(accuracy_score(y, y_pred)),
            'precision': float(precision_score(y, y_pred)),
            'recall': float(recall_score(y, y_pred)),
            'f1': float(f1_score(y, y_pred)),
            'auc': float(roc_auc_score(y, y_proba)),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist()
        }
        
        # Store metrics
        self.metrics[dataset_name] = metrics
        
        # Log results
        logger.info(f"✅ {dataset_name.capitalize()} Evaluation Results:")
        logger.info(f"   Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"   Precision: {metrics['precision']:.4f}")
        logger.info(f"   Recall:    {metrics['recall']:.4f}")
        logger.info(f"   F1-Score:  {metrics['f1']:.4f}")
        logger.info(f"   AUC-ROC:   {metrics['auc']:.4f}")
        
        return metrics
    
    def get_feature_importance(self, feature_names: list) -> Dict:
        """
        Get feature importance (coefficients for logistic regression)
        
        Args:
            feature_names: List of feature names
        
        Returns:
            importance_dict: Dict mapping feature names to importance scores
        """
        if self.model is None:
            logger.warning("Model not trained yet")
            return {}
        
        # Get coefficients
        coefficients = self.model.coef_[0]
        
        # Create importance dict
        importance = dict(zip(feature_names, np.abs(coefficients)))
        
        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        logger.info("Top 10 most important features:")
        for i, (feature, coef) in enumerate(list(importance.items())[:10], 1):
            logger.info(f"   {i}. {feature}: {coef:.4f}")
        
        return importance
    
    def save_model(self, filepath: str):
        """
        Save model to disk
        
        Args:
            filepath: Path to save model
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'metrics': self.metrics,
            'config': self.config
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load model from disk
        
        Args:
            filepath: Path to load model from
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.metrics = model_data['metrics']
        self.config = model_data['config']
        logger.info(f"✅ Model loaded from {filepath}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features
        
        Returns:
            predictions: Binary predictions (0 or 1)
        """
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities
        
        Args:
            X: Features
        
        Returns:
            probabilities: Probability of churn (class 1)
        """
        return self.model.predict_proba(X)[:, 1]


def main():
    """
    Test baseline model
    """
    import yaml
    from src.data.preprocessing import DataPreprocessor
    from src.features.engineering import FeatureEngineer
    import pandas as pd
    
    logging.basicConfig(level=logging.INFO)
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load and preprocess data
    df = pd.read_csv('data/raw/telco_customer_churn.csv')
    
    # Create features
    engineer = FeatureEngineer()
    df = engineer.create_features(df, mode='offline')
    
    # Preprocess
    preprocessor = DataPreprocessor(config)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_for_training(df)
    
    # Train baseline model
    baseline = BaselineModel(config)
    baseline.train(X_train, y_train)
    
    # Evaluate
    val_metrics = baseline.evaluate(X_val, y_val, "validation")
    test_metrics = baseline.evaluate(X_test, y_test, "test")
    
    # Feature importance
    importance = baseline.get_feature_importance(preprocessor.feature_names)
    
    # Save model
    baseline.save_model('models/baseline/logistic_regression_v1.pkl')
    
    print("\n✅ Baseline model test completed successfully!")


if __name__ == "__main__":
    main()
