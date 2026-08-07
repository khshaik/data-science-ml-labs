"""
Batch prediction script
Process large volumes of customers for monthly scoring

Section C: Serving & Inference Pattern (25%) - Batch Pipeline Component
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import time
from datetime import datetime
import argparse
import yaml
import joblib
import json

from src.features.engineering import FeatureEngineer
from src.data.preprocessing import DataPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchPredictor:
    """
    Batch prediction for churn scoring
    
    Use Case: Monthly scoring of entire customer base for marketing campaigns
    Throughput: Process 7,000+ customers in minutes
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.preprocessor = None
        self.feature_engineer = None
        self.model_version = None
    
    def load_model(self):
        """
        Load model and preprocessor
        """
        logger.info("Loading model and preprocessor...")
        
        # Load feature engineer
        self.feature_engineer = FeatureEngineer()
        self.feature_engineer.load_threshold('artifacts/feature_threshold.json')
        
        # Load preprocessor
        self.preprocessor = DataPreprocessor(self.config)
        self.preprocessor.load_preprocessor('artifacts/preprocessor.pkl')
        
        champion_path = Path(self.config['models'].get('current_best_path', 'models/current_best.json'))
        if not champion_path.exists():
            raise FileNotFoundError("Champion manifest not found; run model evaluation first")
        with open(champion_path, 'r') as f:
            champion = json.load(f)
        selected_path = Path(champion['model_path'])

        if champion['model_type'] == 'candidate':
            import tensorflow as tf
            self.model = tf.keras.models.load_model(str(selected_path))
            self.model_version = champion['model_version']
            logger.info("✅ Loaded candidate model (Neural Network)")
        elif champion['model_type'] == 'baseline':
            model_data = joblib.load(selected_path)
            self.model = model_data['model']
            self.model_version = champion['model_version']
            logger.info("✅ Loaded baseline model (Logistic Regression)")
        else:
            raise ValueError(f"Unsupported champion model type: {champion['model_type']}")
    
    def predict_batch(self, input_file: str, output_file: str, chunk_size: int = 1000):
        """
        Predict churn for batch of customers
        
        Args:
            input_file: Input CSV file with customer data
            output_file: Output CSV file with predictions
            chunk_size: Number of rows to process at once
        """
        logger.info("=" * 80)
        logger.info("BATCH PREDICTION PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Input: {input_file}")
        logger.info(f"Output: {output_file}")
        logger.info(f"Chunk size: {chunk_size}")
        
        start_time = time.time()
        
        # Read input data
        logger.info("\n[1/4] Reading input data...")
        df = pd.read_csv(input_file)
        total_rows = len(df)
        logger.info(f"✅ Loaded {total_rows} customers")
        
        # Store original customerID if present
        has_customer_id = 'customerID' in df.columns
        if has_customer_id:
            customer_ids = df['customerID'].copy()
        
        # Feature engineering
        logger.info("\n[2/4] Creating features...")
        df = self.feature_engineer.create_features(df, mode='online')
        logger.info("✅ Features created")
        
        # Preprocessing
        logger.info("\n[3/4] Preprocessing data...")
        df = self.preprocessor.preprocess_for_serving(df)
        df = df[self.preprocessor.feature_names]
        logger.info("✅ Data preprocessed")
        
        # Predict in chunks
        logger.info(f"\n[4/4] Generating predictions (chunk size: {chunk_size})...")
        predictions = []
        probabilities = []
        
        num_chunks = (total_rows + chunk_size - 1) // chunk_size
        
        for i in range(0, total_rows, chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            chunk_num = i // chunk_size + 1
            
            if 'candidate' in self.model_version:
                # Neural Network
                chunk_proba = self.model.predict(chunk.values, verbose=0).flatten()
            else:
                # Logistic Regression
                chunk_proba = self.model.predict_proba(chunk)[:, 1]
            
            chunk_pred = (chunk_proba > 0.5).astype(int)
            
            predictions.extend(chunk_pred)
            probabilities.extend(chunk_proba)
            
            logger.info(f"  Processed chunk {chunk_num}/{num_chunks} ({len(chunk)} rows)")
        
        # Calculate statistics
        elapsed_time = time.time() - start_time
        throughput = total_rows / elapsed_time
        
        logger.info("✅ Predictions generated")
        logger.info(f"\n" + "=" * 80)
        logger.info("BATCH PREDICTION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total rows processed: {total_rows}")
        logger.info(f"Total time: {elapsed_time:.2f} seconds")
        logger.info(f"Throughput: {throughput:.2f} rows/sec")
        logger.info(f"Average latency: {(elapsed_time/total_rows)*1000:.2f} ms/row")
        
        # Churn statistics
        churn_count = sum(predictions)
        churn_rate = churn_count / total_rows
        logger.info(f"\nChurn predictions:")
        logger.info(f"  Predicted churn: {churn_count} ({churn_rate*100:.1f}%)")
        logger.info(f"  Predicted no churn: {total_rows - churn_count} ({(1-churn_rate)*100:.1f}%)")
        logger.info("=" * 80)
        
        # Create output DataFrame
        output_df = pd.DataFrame({
            'churn_probability': probabilities,
            'churn_prediction': ['Yes' if p == 1 else 'No' for p in predictions],
            'risk_level': ['High' if p >= 0.7 else 'Medium' if p >= 0.4 else 'Low' 
                          for p in probabilities],
            'model_version': self.model_version,
            'prediction_date': datetime.now().strftime('%Y-%m-%d')
        })
        
        # Add customerID if present
        if has_customer_id:
            output_df.insert(0, 'customerID', customer_ids.values)
        
        # Save predictions
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_file, index=False)
        logger.info(f"✅ Predictions saved to {output_file}")
        
        return {
            'total_rows': total_rows,
            'elapsed_time': elapsed_time,
            'throughput': throughput,
            'churn_count': churn_count,
            'churn_rate': churn_rate
        }


def main():
    """
    Main batch prediction script
    """
    parser = argparse.ArgumentParser(description='Batch churn prediction')
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--output', required=True, help='Output CSV file')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Chunk size for processing')
    parser.add_argument('--config', default='config/config.yaml', help='Config file')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = BatchPredictor(config_path=args.config)
    
    # Load model
    predictor.load_model()
    
    # Run batch prediction
    results = predictor.predict_batch(args.input, args.output, args.chunk_size)
    
    logger.info("\n✅ Batch prediction completed successfully!")


if __name__ == "__main__":
    main()
