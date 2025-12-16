import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

from src.data_processing import preprocess_data


# -----------------------------
# App
# -----------------------------
app = FastAPI(title="Fraud Detection API")

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"

# -----------------------------
# Load models once
# -----------------------------
lr = joblib.load(MODEL_DIR / "logistic_regression.pkl")
gb = joblib.load(MODEL_DIR / "gradient_boosting.pkl")

# -----------------------------
# Feature list (MUST match training)
# -----------------------------
FEATURES = [
    "CountryCode",
    "Amount",
    "Value",
    "PricingStrategy"
]

# -----------------------------
# Input schema
# -----------------------------
class Transaction(BaseModel):
    CountryCode: int
    Amount: float
    Value: float
    PricingStrategy: int

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def health():
    return {"status": "API is running"}

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(transaction: Transaction):
    # Convert input to DataFrame
    df = pd.DataFrame([transaction.dict()])

    # Preprocess
    df_processed = preprocess_data(df)

    # Select features
    X = df_processed[FEATURES]

    # Predict probabilities
    lr_prob = lr.predict_proba(X)[0, 1]
    gb_prob = gb.predict_proba(X)[0, 1]

    return {
        "logistic_regression_probability": lr_prob,
        "gradient_boosting_probability": gb_prob
    }
