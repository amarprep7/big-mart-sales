from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from .config import MODEL_PATH
from .data_preprocessing import preprocess_data

app = FastAPI()
 
# Load model and features at startup
model = None
features = None

class PredictionRequest(BaseModel):
    data: list  # List of dicts, each dict is a row

@app.on_event("startup")
def load_model():
    global model, features
    model = joblib.load(MODEL_PATH)
    features_path = MODEL_PATH.replace('.joblib', '_features.txt')
    with open(features_path) as f:
        features = [line.strip() for line in f]

@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None or features is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    # Convert input to DataFrame
    df = pd.DataFrame(request.data)
    # Preprocess
    df = preprocess_data(df)
    # Select only the features used in training
    try:
        df = df[features]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing features: {e}")
    # Predict
    preds = model.predict(df)
    return {"predictions": preds.tolist()}
