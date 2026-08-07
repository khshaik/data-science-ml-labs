"""
FastAPI service for real-time churn prediction
Provides REST API endpoints for online inference

Section C: Serving & Inference Pattern (25%) - Online API Component
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Optional
import numpy as np
import pandas as pd
import logging
import time
from datetime import datetime
from pathlib import Path
import json
import yaml
import joblib

from src.features.engineering import FeatureEngineer
from src.data.preprocessing import DataPreprocessor

# Prometheus metrics
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="Real-time customer churn prediction service",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
prediction_counter = Counter('predictions_total', 'Total number of predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency in seconds')
error_counter = Counter('prediction_errors_total', 'Total number of prediction errors')


# Request/Response models
class CustomerData(BaseModel):
    """
    Customer data for churn prediction
    """
    gender: str = Field(..., example="Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., ge=0, example=12)
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="No")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., ge=0, example=65.50)
    TotalCharges: float = Field(..., ge=0, example=786.00)
    
    class Config:
        schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 65.50,
                "TotalCharges": 786.00
            }
        }


class PredictionResponse(BaseModel):
    """
    Churn prediction response
    """
    churn_probability: float = Field(..., description="Probability of churn (0-1)")
    churn_prediction: str = Field(..., description="Predicted class (Yes/No)")
    risk_level: str = Field(..., description="Risk level (Low/Medium/High)")
    model_version: str = Field(..., description="Model version used")
    latency_ms: float = Field(..., description="Prediction latency in milliseconds")
    timestamp: str = Field(..., description="Prediction timestamp")


# Global variables for model and preprocessor
model = None
preprocessor = None
feature_engineer = None
explainer = None
config = None
model_version = "v1.0.0"
X_train_sample = None  # For explainer


@app.on_event("startup")
async def load_model():
    """
    Load model and preprocessor on startup
    """
    global model, preprocessor, feature_engineer, explainer, config, model_version, X_train_sample
    
    logger.info("Loading model and preprocessor...")
    
    try:
        # Load config
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Load feature engineer
        feature_engineer = FeatureEngineer()
        feature_engineer.load_threshold('artifacts/feature_threshold.json')
        
        # Load preprocessor
        preprocessor = DataPreprocessor(config)
        preprocessor.load_preprocessor('artifacts/preprocessor.pkl')
        
        # Load the champion selected by the validation guardrails.
        champion_path = Path(config['models'].get('current_best_path', 'models/current_best.json'))
        if not champion_path.exists():
            raise FileNotFoundError("Champion manifest not found; run model evaluation first")
        with open(champion_path, 'r') as f:
            champion = json.load(f)
        selected_path = Path(champion['model_path'])

        if champion['model_type'] == 'candidate':
            import tensorflow as tf
            model = tf.keras.models.load_model(str(selected_path))
            model_version = champion['model_version']
            logger.info("✅ Loaded candidate model (Neural Network)")
        elif champion['model_type'] == 'baseline':
            model_data = joblib.load(selected_path)
            model = model_data['model']
            model_version = champion['model_version']
            logger.info("✅ Loaded baseline model (Logistic Regression)")
        else:
            raise ValueError(f"Unsupported champion model type: {champion['model_type']}")
        
        logger.info(f"✅ Model loaded successfully: {model_version}")
        
        # Load explainer (optional, for /explain endpoint)
        try:
            from src.models.explainer import ModelExplainer
            
            # Load a small sample of training data for explainer
            df_sample = pd.read_csv('data/raw/telco_customer_churn.csv').head(100)
            df_sample = feature_engineer.create_features(df_sample, mode='online')
            X_train_sample = preprocessor.preprocess_for_serving(df_sample)
            X_train_sample = X_train_sample[preprocessor.feature_names]
            
            # Create explainer
            model_type_str = 'candidate' if 'candidate' in model_version else 'baseline'
            explainer = ModelExplainer(model, X_train_sample, preprocessor.feature_names, model_type=model_type_str)
            logger.info("✅ Explainer loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️  Explainer not loaded: {e}")
            explainer = None
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


@app.get("/")
async def root():
    """
    Root endpoint
    """
    return {
        "service": "Churn Prediction API",
        "version": "1.0.0",
        "model_version": model_version,
        "status": "healthy",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "metrics": "/metrics"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": model_version,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerData):
    """
    Predict customer churn
    
    Args:
        customer: Customer data
    
    Returns:
        PredictionResponse with churn probability and prediction
    """
    start_time = time.time()
    
    try:
        # Convert to DataFrame
        customer_dict = customer.dict()
        df = pd.DataFrame([customer_dict])
        
        # Feature engineering (online mode)
        df = feature_engineer.create_features(df, mode='online')
        
        # Preprocessing
        df = preprocessor.preprocess_for_serving(df)
        
        # Ensure correct feature order
        df = df[preprocessor.feature_names]
        
        # Predict
        if 'candidate' in model_version:
            # Neural Network
            churn_proba = float(model.predict(df.values, verbose=0)[0][0])
        else:
            # Logistic Regression
            churn_proba = float(model.predict_proba(df)[0][1])
        
        # Determine prediction and risk level
        churn_prediction = "Yes" if churn_proba > 0.5 else "No"
        
        if churn_proba >= 0.7:
            risk_level = "High"
        elif churn_proba >= 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Update metrics
        prediction_counter.inc()
        prediction_latency.observe(time.time() - start_time)
        
        # Create response
        response = PredictionResponse(
            churn_probability=churn_proba,
            churn_prediction=churn_prediction,
            risk_level=risk_level,
            model_version=model_version,
            latency_ms=latency_ms,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Prediction: {churn_prediction} (prob={churn_proba:.3f}, latency={latency_ms:.1f}ms)")
        
        return response
        
    except Exception as e:
        error_counter.inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/explain")
async def explain_prediction(customer: CustomerData):
    """
    Explain prediction using SHAP and LIME
    
    Bonus Feature: Model Explainability (+1 mark)
    
    Args:
        customer: Customer data
    
    Returns:
        Explanation with feature contributions
    """
    if explainer is None:
        raise HTTPException(status_code=503, detail="Explainer not available")
    
    try:
        # Convert to DataFrame
        customer_dict = customer.dict()
        df = pd.DataFrame([customer_dict])
        
        # Feature engineering (online mode)
        df = feature_engineer.create_features(df, mode='online')
        
        # Preprocessing
        df = preprocessor.preprocess_for_serving(df)
        df = df[preprocessor.feature_names]
        
        # Generate explanation
        explanation = explainer.generate_explanation_report(
            df,
            customer_id=customer_dict.get('customerID', 'UNKNOWN')
        )
        
        logger.info(f"Explanation generated for customer")
        
        return explanation
        
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Run server
    uvicorn.run(
        app,
        host=config['serving']['api']['host'],
        port=config['serving']['api']['port'],
        log_level="info"
    )
