"""
Candidate Model: Neural Network
More complex model for churn prediction, based on original notebook architecture

Section B: Model Training & Offline Evaluation (25%) - Candidate Model
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
import logging
from typing import Dict, Tuple
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class CandidateModel:
    """
    Neural Network model for churn prediction
    Architecture based on original notebook (2025em1100102_telcochurn_prediction_neuralnetworks.ipynb)
    
    Justification:
    - Captures non-linear patterns
    - Higher capacity than logistic regression
    - Dropout for regularization
    - Early stopping to prevent overfitting
    - Proven architecture from original notebook
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.history = None
        self.metrics = {}
    
    def build_model(self, input_dim: int):
        """
        Build Neural Network model
        Architecture from original notebook with improvements
        
        Args:
            input_dim: Number of input features
        """
        hyperparams = self.config['models']['candidate']['hyperparameters']
        
        # Build sequential model (from original notebook architecture)
        self.model = models.Sequential([
            # Input layer
            layers.Input(shape=(input_dim,)),
            
            # Hidden layer 1
            layers.Dense(hyperparams['hidden_layers'][0], activation='relu'),
            layers.Dropout(hyperparams['dropout_rate']),
            
            # Hidden layer 2
            layers.Dense(hyperparams['hidden_layers'][1], activation='relu'),
            layers.Dropout(hyperparams['dropout_rate']),
            
            # Hidden layer 3
            layers.Dense(hyperparams['hidden_layers'][2], activation='relu'),
            layers.Dropout(hyperparams['dropout_rate']),
            
            # Output layer
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hyperparams['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc'), 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        logger.info("✅ Candidate model (Neural Network) built")
        logger.info(f"   Architecture: {hyperparams['hidden_layers']}")
        logger.info(f"   Dropout: {hyperparams['dropout_rate']}")
        logger.info(f"   Learning rate: {hyperparams['learning_rate']}")
        
        # Print model summary
        self.model.summary(print_fn=logger.info)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        
        Returns:
            training_history: Dict with training history
        """
        logger.info("Training candidate model...")
        
        if self.model is None:
            self.build_model(X_train.shape[1])
        
        hyperparams = self.config['models']['candidate']['hyperparameters']
        
        # Callbacks (from original notebook approach)
        callback_list = [
            # Early stopping to prevent overfitting
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=hyperparams['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=hyperparams['epochs'],
            batch_size=hyperparams['batch_size'],
            callbacks=callback_list,
            verbose=1
        )
        
        # Get final training metrics
        training_metrics = {
            'final_epoch': len(self.history.history['loss']),
            'train_loss': float(self.history.history['loss'][-1]),
            'train_accuracy': float(self.history.history['accuracy'][-1]),
            'train_auc': float(self.history.history['auc'][-1]),
            'val_loss': float(self.history.history['val_loss'][-1]),
            'val_accuracy': float(self.history.history['val_accuracy'][-1]),
            'val_auc': float(self.history.history['val_auc'][-1])
        }
        
        logger.info("✅ Training completed")
        logger.info(f"   Final epoch: {training_metrics['final_epoch']}")
        logger.info(f"   Validation AUC: {training_metrics['val_auc']:.4f}")
        logger.info(f"   Validation Accuracy: {training_metrics['val_accuracy']:.4f}")
        
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
        y_proba = self.model.predict(X, verbose=0).flatten()
        y_pred = (y_proba > 0.5).astype(int)
        
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
    
    def save_model(self, filepath: str):
        """
        Save model to disk
        
        Args:
            filepath: Path to save model (.h5 format)
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save Keras model
        self.model.save(filepath)
        
        # Save metrics and history separately
        metadata = {
            'metrics': self.metrics,
            'training_history': {k: [float(v) for v in vals] 
                               for k, vals in self.history.history.items()} if self.history else {},
            'config': self.config
        }
        
        metadata_path = filepath.replace('.h5', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Model saved to {filepath}")
        logger.info(f"✅ Metadata saved to {metadata_path}")
    
    def load_model(self, filepath: str):
        """
        Load model from disk
        
        Args:
            filepath: Path to load model from
        """
        # Load Keras model
        self.model = keras.models.load_model(filepath)
        
        # Load metadata
        metadata_path = filepath.replace('.h5', '_metadata.json')
        if Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.metrics = metadata['metrics']
            self.config = metadata['config']
        
        logger.info(f"✅ Model loaded from {filepath}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features
        
        Returns:
            predictions: Binary predictions (0 or 1)
        """
        y_proba = self.model.predict(X, verbose=0).flatten()
        return (y_proba > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities
        
        Args:
            X: Features
        
        Returns:
            probabilities: Probability of churn (class 1)
        """
        return self.model.predict(X, verbose=0).flatten()


def main():
    """
    Test candidate model
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
    
    # Train candidate model
    candidate = CandidateModel(config)
    training_history = candidate.train(X_train, y_train, X_val, y_val)
    
    # Evaluate
    val_metrics = candidate.evaluate(X_val, y_val, "validation")
    test_metrics = candidate.evaluate(X_test, y_test, "test")
    
    # Save model
    candidate.save_model('models/candidate/neural_network_v1.h5')
    
    print("\n✅ Candidate model test completed successfully!")


if __name__ == "__main__":
    main()
