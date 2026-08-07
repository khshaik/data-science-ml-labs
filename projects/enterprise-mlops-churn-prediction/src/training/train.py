"""
Training pipeline with MLflow tracking
Trains both baseline and candidate models

Section B: Model Training & Offline Evaluation (25%) - Training Pipeline
"""

import pandas as pd
import numpy as np
import yaml
import mlflow
import mlflow.sklearn
import mlflow.keras
import logging
from pathlib import Path
from datetime import datetime
import argparse

from src.features.engineering import FeatureEngineer
from src.data.preprocessing import DataPreprocessor
from src.models.baseline import BaselineModel
from src.models.candidate import CandidateModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    End-to-end training pipeline
    
    Flow: Load data → Feature engineering → Preprocessing → Train → Evaluate → Save
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set up MLflow
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
    
    def run(self, model_type: str = 'baseline'):
        """
        Run complete training pipeline
        
        Args:
            model_type: 'baseline' or 'candidate'
        """
        logger.info("=" * 80)
        logger.info(f"STARTING TRAINING PIPELINE - {model_type.upper()} MODEL")
        logger.info("=" * 80)
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Log parameters
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("data_path", self.config['data']['raw_path'])
            mlflow.log_param("train_split", self.config['data']['train_split'])
            mlflow.log_param("val_split", self.config['data']['val_split'])
            mlflow.log_param("test_split", self.config['data']['test_split'])
            
            # Step 1: Load data
            logger.info("\n[1/6] Loading data...")
            df = pd.read_csv(self.config['data']['raw_path'])
            logger.info(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
            mlflow.log_metric("dataset_size", len(df))
            
            # Step 2: Split raw data before fitting data-derived features.
            # This prevents the high-value percentile from seeing validation
            # or test rows.
            logger.info("\n[2/6] Splitting raw data and creating features...")
            preprocessor = DataPreprocessor(self.config)
            X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = preprocessor.split_data(df)

            engineer = FeatureEngineer()
            X_train_features = engineer.create_features(X_train_raw, mode='offline')
            X_val_features = engineer.create_features(X_val_raw, mode='online')
            X_test_features = engineer.create_features(X_test_raw, mode='online')
            logger.info(f"✅ Created {len(engineer.get_engineered_feature_names())} engineered features")

            engineer.save_threshold('artifacts/feature_threshold.json')
            mlflow.log_artifact('artifacts/feature_threshold.json')

            # Step 3: Fit encoders/scaler on training rows only.
            logger.info("\n[3/6] Preprocessing feature splits...")
            X_train, X_val, X_test = preprocessor.preprocess_feature_splits(
                X_train_features, X_val_features, X_test_features
            )
            
            # Save preprocessor for serving
            preprocessor.save_preprocessor('artifacts/preprocessor.pkl')
            mlflow.log_artifact('artifacts/preprocessor.pkl')
            
            # Log data splits
            mlflow.log_metric("train_size", len(X_train))
            mlflow.log_metric("val_size", len(X_val))
            mlflow.log_metric("test_size", len(X_test))
            mlflow.log_metric("churn_rate_train", y_train.mean())
            
            # Step 4: Train model
            logger.info(f"\n[4/6] Training {model_type} model...")
            
            if model_type == 'baseline':
                model = BaselineModel(self.config)
                
                # Log hyperparameters
                for key, value in self.config['models']['baseline']['hyperparameters'].items():
                    mlflow.log_param(f"baseline_{key}", value)
                
                # Train
                training_metrics = model.train(X_train, y_train)
                
                # Get feature importance
                importance = model.get_feature_importance(preprocessor.feature_names)
                
            else:  # candidate
                model = CandidateModel(self.config)
                
                # Log hyperparameters
                for key, value in self.config['models']['candidate']['hyperparameters'].items():
                    mlflow.log_param(f"candidate_{key}", value)
                
                # Train
                training_metrics = model.train(X_train, y_train, X_val, y_val)
            
            # Step 5: Evaluate
            logger.info("\n[5/6] Evaluating model...")
            
            val_metrics = model.evaluate(X_val, y_val, "validation")
            test_metrics = model.evaluate(X_test, y_test, "test")
            
            # Log metrics to MLflow
            for metric_name, metric_value in val_metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(f"val_{metric_name}", metric_value)
            
            for metric_name, metric_value in test_metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(f"test_{metric_name}", metric_value)
            
            # Step 6: Save model
            logger.info("\n[6/6] Saving model...")
            
            model_dir = Path(self.config['models'][model_type]['path'])
            model_dir.mkdir(parents=True, exist_ok=True)
            
            if model_type == 'baseline':
                model_path = model_dir / 'logistic_regression_v1.pkl'
                model.save_model(str(model_path))
                
                # Log model to MLflow
                mlflow.sklearn.log_model(model.model, "model")
                
            else:  # candidate
                model_path = model_dir / 'neural_network_v1.h5'
                model.save_model(str(model_path))
                
                # Log model to MLflow
                mlflow.keras.log_model(model.model, "model")
            
            # Log model artifact
            mlflow.log_artifact(str(model_path))
            
            # Save evaluation report
            self._save_evaluation_report(model_type, val_metrics, test_metrics)
            
            logger.info("\n" + "=" * 80)
            logger.info(f"✅ TRAINING PIPELINE COMPLETED - {model_type.upper()}")
            logger.info("=" * 80)
            logger.info(f"Test AUC: {test_metrics['auc']:.4f}")
            logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
            logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
            logger.info("=" * 80)
    
    def _save_evaluation_report(self, model_type: str, val_metrics: dict, test_metrics: dict):
        """
        Save evaluation report
        """
        import json
        
        report = {
            'model_type': model_type,
            'timestamp': datetime.now().isoformat(),
            'validation_metrics': val_metrics,
            'test_metrics': test_metrics
        }
        
        report_dir = Path('artifacts/eval')
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = report_dir / f'{model_type}_evaluation.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✅ Evaluation report saved to {report_path}")
        mlflow.log_artifact(str(report_path))


def main():
    """
    Main training script
    """
    parser = argparse.ArgumentParser(description='Train churn prediction model')
    parser.add_argument('--model', type=str, default='baseline', 
                       choices=['baseline', 'candidate'],
                       help='Model type to train')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Config file path')
    
    args = parser.parse_args()
    
    # Run training pipeline
    pipeline = TrainingPipeline(config_path=args.config)
    pipeline.run(model_type=args.model)


if __name__ == "__main__":
    main()
