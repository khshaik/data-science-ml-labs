"""
Model explainability using SHAP and LIME
Provides interpretable explanations for predictions

Bonus Feature: SHAP/LIME Explainability (+1 mark)
"""

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelExplainer:
    """
    Model explainability using SHAP and LIME
    
    From Week 11: Security, Compliance & Responsible AI
    """
    
    def __init__(self, model, X_train: pd.DataFrame, feature_names: List[str], model_type: str = 'baseline'):
        """
        Initialize explainer
        
        Args:
            model: Trained model
            X_train: Training data for background distribution
            feature_names: List of feature names
            model_type: 'baseline' or 'candidate'
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.model_type = model_type
        
        # Initialize SHAP explainer
        if model_type == 'baseline':
            # For tree-based or linear models
            self.shap_explainer = shap.Explainer(model, X_train)
        else:
            # For neural networks, use KernelExplainer
            def model_predict(X):
                return model.predict(X, verbose=0).flatten()
            
            # Use a subset for background (faster)
            background = shap.sample(X_train, min(100, len(X_train)))
            self.shap_explainer = shap.KernelExplainer(model_predict, background)
        
        # Initialize LIME explainer
        self.lime_explainer = LimeTabularExplainer(
            X_train.values,
            feature_names=feature_names,
            class_names=['No Churn', 'Churn'],
            mode='classification',
            random_state=42
        )
        
        logger.info(f"✅ Explainer initialized for {model_type} model")
    
    def explain_with_shap(self, X: pd.DataFrame, max_display: int = 10) -> Dict:
        """
        Generate SHAP explanations
        
        Args:
            X: Data to explain
            max_display: Maximum features to display
        
        Returns:
            explanation_dict: Dict with SHAP values and feature importance
        """
        logger.info("Generating SHAP explanations...")
        
        # Calculate SHAP values
        shap_values = self.shap_explainer(X)
        
        # Get feature importance (mean absolute SHAP values)
        if self.model_type == 'baseline':
            # For binary classification, use values for positive class
            if len(shap_values.shape) == 3:
                shap_vals = shap_values.values[:, :, 1]
            else:
                shap_vals = shap_values.values
        else:
            shap_vals = shap_values.values
        
        feature_importance = np.abs(shap_vals).mean(axis=0)
        
        # Create importance dict
        importance_dict = dict(zip(self.feature_names, feature_importance))
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        # Top features
        top_features = list(importance_dict.items())[:max_display]
        
        logger.info(f"Top {len(top_features)} most important features (SHAP):")
        for i, (feature, importance) in enumerate(top_features, 1):
            logger.info(f"  {i}. {feature}: {importance:.4f}")
        
        return {
            'shap_values': shap_vals.tolist() if isinstance(shap_vals, np.ndarray) else shap_vals,
            'feature_importance': importance_dict,
            'top_features': top_features,
            'method': 'SHAP'
        }
    
    def explain_single_prediction_shap(self, X_single: pd.DataFrame, customer_id: str = None) -> Dict:
        """
        Explain single prediction using SHAP
        
        Args:
            X_single: Single customer data (1 row)
            customer_id: Optional customer ID
        
        Returns:
            explanation: Dict with SHAP explanation for single prediction
        """
        logger.info(f"Explaining single prediction with SHAP (customer: {customer_id})...")
        
        # Calculate SHAP values
        shap_values = self.shap_explainer(X_single)
        
        # Get values for this prediction
        if self.model_type == 'baseline':
            if len(shap_values.shape) == 3:
                shap_vals = shap_values.values[0, :, 1]
            else:
                shap_vals = shap_values.values[0]
        else:
            shap_vals = shap_values.values[0]
        
        # Create feature contribution dict
        contributions = dict(zip(self.feature_names, shap_vals))
        contributions = dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True))
        
        # Top 5 contributing features
        top_contributions = list(contributions.items())[:5]
        
        logger.info("Top 5 contributing features:")
        for feature, contribution in top_contributions:
            direction = "increases" if contribution > 0 else "decreases"
            logger.info(f"  {feature}: {contribution:+.4f} ({direction} churn risk)")
        
        return {
            'customer_id': customer_id,
            'contributions': contributions,
            'top_contributions': top_contributions,
            'base_value': float(shap_values.base_values[0]) if hasattr(shap_values, 'base_values') else 0.0,
            'method': 'SHAP'
        }
    
    def explain_with_lime(self, X_single: pd.DataFrame, customer_id: str = None, num_features: int = 10) -> Dict:
        """
        Generate LIME explanation for single prediction
        
        Args:
            X_single: Single customer data (1 row)
            customer_id: Optional customer ID
            num_features: Number of features to include in explanation
        
        Returns:
            explanation_dict: Dict with LIME explanation
        """
        logger.info(f"Generating LIME explanation (customer: {customer_id})...")
        
        # Predict function for LIME
        if self.model_type == 'baseline':
            predict_fn = self.model.predict_proba
        else:
            def predict_fn(X):
                preds = self.model.predict(X, verbose=0).flatten()
                return np.column_stack([1 - preds, preds])
        
        # Generate explanation
        exp = self.lime_explainer.explain_instance(
            X_single.values[0],
            predict_fn,
            num_features=num_features
        )
        
        # Extract feature contributions
        lime_values = exp.as_list()
        contributions = {feature: value for feature, value in lime_values}
        
        logger.info(f"Top {min(5, len(lime_values))} contributing features (LIME):")
        for i, (feature, contribution) in enumerate(lime_values[:5], 1):
            direction = "increases" if contribution > 0 else "decreases"
            logger.info(f"  {i}. {feature}: {contribution:+.4f} ({direction} churn risk)")
        
        return {
            'customer_id': customer_id,
            'contributions': contributions,
            'top_contributions': lime_values[:5],
            'prediction_probability': exp.predict_proba[1],
            'method': 'LIME'
        }
    
    def generate_explanation_report(self, X_single: pd.DataFrame, customer_id: str = None) -> Dict:
        """
        Generate comprehensive explanation using both SHAP and LIME
        
        Args:
            X_single: Single customer data
            customer_id: Optional customer ID
        
        Returns:
            report: Combined explanation report
        """
        logger.info("=" * 80)
        logger.info("GENERATING COMPREHENSIVE EXPLANATION REPORT")
        logger.info("=" * 80)
        
        # Get prediction
        if self.model_type == 'baseline':
            prediction_proba = self.model.predict_proba(X_single.values)[0][1]
        else:
            prediction_proba = self.model.predict(X_single.values, verbose=0)[0][0]
        
        prediction = "Churn" if prediction_proba > 0.5 else "No Churn"
        
        logger.info(f"Customer ID: {customer_id}")
        logger.info(f"Prediction: {prediction} (probability: {prediction_proba:.3f})")
        
        # SHAP explanation
        shap_explanation = self.explain_single_prediction_shap(X_single, customer_id)
        
        # LIME explanation
        lime_explanation = self.explain_with_lime(X_single, customer_id)
        
        # Combined report
        report = {
            'customer_id': customer_id,
            'prediction': prediction,
            'prediction_probability': float(prediction_proba),
            'shap_explanation': shap_explanation,
            'lime_explanation': lime_explanation,
            'model_type': self.model_type
        }
        
        logger.info("=" * 80)
        
        return report
    
    def save_explanation(self, explanation: Dict, output_file: str):
        """
        Save explanation to JSON file
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(explanation, f, indent=2)
        
        logger.info(f"✅ Explanation saved to {output_file}")


def main():
    """
    Test explainability
    """
    import yaml
    import joblib
    from src.data.preprocessing import DataPreprocessor
    from src.features.engineering import FeatureEngineer
    
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
    
    # Load baseline model
    model_path = 'models/baseline/logistic_regression_v1.pkl'
    if Path(model_path).exists():
        model_data = joblib.load(model_path)
        model = model_data['model']
        
        # Create explainer
        explainer = ModelExplainer(model, X_train, preprocessor.feature_names, model_type='baseline')
        
        # Global feature importance
        global_explanation = explainer.explain_with_shap(X_test[:100])
        
        # Single prediction explanation
        single_explanation = explainer.generate_explanation_report(
            X_test.iloc[[0]], 
            customer_id="TEST_001"
        )
        
        # Save explanations
        explainer.save_explanation(global_explanation, 'artifacts/explanations/global_shap.json')
        explainer.save_explanation(single_explanation, 'artifacts/explanations/single_prediction.json')
        
        logger.info("\n✅ Explainability test completed!")
    else:
        logger.error(f"Model not found: {model_path}")
        logger.error("Please train the model first: python src/training/train.py --model baseline")


if __name__ == "__main__":
    main()
