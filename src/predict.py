import pandas as pd
import joblib
from pathlib import Path


from data_processing import preprocess_data

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "data.csv"
MODEL_DIR = BASE_DIR / "models"

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
# Load data
# -----------------------------
df_raw = pd.read_csv(DATA_PATH)

# Preprocess raw data
df_processed = preprocess_data(df_raw)

# IMPORTANT: make a clean feature matrix
X = df_processed[FEATURES].copy()

# -----------------------------
# Load models
# -----------------------------
lr = joblib.load(MODEL_DIR / "logistic_regression.pkl")
gb = joblib.load(MODEL_DIR / "gradient_boosting.pkl")

# -----------------------------
# Predictions (NO mutation of X)
# -----------------------------
lr_probs = lr.predict_proba(X)[:, 1]
gb_probs = gb.predict_proba(X)[:, 1]

# -----------------------------
# Attach results safely
# -----------------------------
results = df_raw.copy()
results["lr_prob"] = lr_probs
results["gb_prob"] = gb_probs

print(results[["lr_prob", "gb_prob"]].head())


# Create output directory if it doesn't exist
output_dir = BASE_DIR / "data" / "predictions"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "fraud_predictions.csv"
results[["lr_prob", "gb_prob"]].to_csv(output_path, index=False)

print(f"Predictions saved to: {output_path}")

output_dir.mkdir(parents=True, exist_ok=True)

# Save predictions
output_path = output_dir / "fraud_predictions.csv"
results[["lr_prob", "gb_prob"]].to_csv(output_path, index=False)


print(f"Predictions saved to {output_path}")
