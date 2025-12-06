# This is the FastAPI application for the ML Forge Prediction API.
# It loads a pre-trained model and provides endpoints for health checks, predictions, and model explanations using SHAP.
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model
import pickle
import pandas as pd
from typing import List, Any
import os
import shap  # Imported for Explainable AI (SHAP) explanations
import numpy as np  # For handling arrays in SHAP

app = FastAPI(title="ML Forge - Prediction API")

# Global variables for the loaded model, feature names, and target encoder
model = None
feature_names = None
target_encoder = None

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Load the model, feature names, and target encoder if files exist
if os.path.exists("models/best_model.pkl"):
    model = pickle.load(open("models/best_model.pkl", "rb"))
    feature_names = pickle.load(open("models/feature_names.pkl", "rb"))
    if os.path.exists("models/target_encoder.pkl"):
        target_encoder = pickle.load(open("models/target_encoder.pkl", "rb"))

# Dynamically create a Pydantic model for individual input items based on feature_names
# This enforces the exact model signature (expected features) in the API schema/docs
if feature_names:
    InputItem = create_model(
        'InputItem',
        # Use Any for flexibility; change to float if all numeric
        **{feature: (Any, ...) for feature in feature_names}
    )
else:
    # Fallback if features not loaded
    class InputItem(BaseModel):
        pass

# Pydantic model for input data validation (list of InputItem for batch predictions)
class InputData(BaseModel):
    data: List[InputItem]

# Root endpoint for basic status check
@app.get("/")
def home():
    return {"status": "API Running ✅", "model_loaded": model is not None}

# Health check endpoint: Returns model info like type and loaded status
# (Added as per requirements for model information)
@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": type(model).__name__,  # e.g., 'RandomForestClassifier'
        # Note: Score is not available unless saved; could extend to load from a file if needed
    }

# Prediction endpoint: Accepts input data, validates, predicts, and decodes if necessary
@app.post("/predict")
def predict(payload: InputData):
    if model is None:
        raise HTTPException(500, "No model saved yet!")

    # Convert payload to DataFrame
    df = pd.DataFrame([item.dict() for item in payload.data])

    # Input validation: Check if all required features are present (already enforced by Pydantic, but double-check)
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing_features}")

    # Reindex to match model's feature order, fill missing with 0 (though Pydantic should prevent missing)
    df = df.reindex(columns=feature_names, fill_value=0)

    # Make predictions
    pred = model.predict(df)

    # Decode predictions if target encoder exists (for categorical targets)
    if target_encoder:
        pred = target_encoder.inverse_transform(pred.astype(int))

    return {"predictions": pred.tolist()}

# Explanation endpoint: Provides SHAP values for input data to explain predictions
# (Added as per requirements for Explainable AI using SHAP)
@app.post("/explain")
def explain(payload: InputData):
    if model is None:
        raise HTTPException(500, "No model saved yet!")

    # Convert payload to DataFrame
    df = pd.DataFrame([item.dict() for item in payload.data])

    # Input validation: Same as predict, check for missing features
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing_features}")

    # Reindex to match model's feature order
    df = df.reindex(columns=feature_names, fill_value=0)

    # Create SHAP explainer (using TreeExplainer if tree-based, else KernelExplainer)
    # Note: This assumes the model is compatible with SHAP; for unsupported models, it may fall back to slower methods
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(df)

        # Convert SHAP values to a JSON-serializable format (list of lists)
        # For multi-class, shap_values.values is [samples, features, classes]; flatten appropriately
        if isinstance(shap_values.values, np.ndarray) and len(shap_values.values.shape) == 3:
            # Multi-class: Return list of shap values per class
            shap_list = shap_values.values.tolist()
        else:
            # Binary/Regression: Simple list
            shap_list = shap_values.values.tolist()

        return {
            "shap_values": shap_list,
            "feature_names": feature_names
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP explanation error: {str(e)}")
