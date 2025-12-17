# tests/test_models.py

import os
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Paths to saved models
GB_MODEL_PATH = "models/gradient_boosting.pkl"
LR_MODEL_PATH = "models/logistic_regression.pkl"

# Check that models exist
for path in [GB_MODEL_PATH, LR_MODEL_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")

# Load models
gb_model = joblib.load(GB_MODEL_PATH)
lr_model = joblib.load(LR_MODEL_PATH)

print("✅ Models loaded successfully.")

# Prepare sample input with correct feature names and types
sample = pd.DataFrame({
    "CountryCode": [251],
    "Amount": [1500],
    "Value": [3],
    "PricingStrategy": [1],
    "RFM_Cluster": [2]  # Ensure this matches training column name
})

# Reorder columns to match training
sample = sample[gb_model.feature_names_in_]

# Predictions
gb_pred = gb_model.predict(sample)
gb_prob = gb_model.predict_proba(sample)

lr_pred = lr_model.predict(sample)
lr_prob = lr_model.predict_proba(sample)

# Display results
print("\nGradient Boosting Predictions:")
print("Class:", gb_pred)
print("Probability:", gb_prob)

print("\nLogistic Regression Predictions:")
print("Class:", lr_pred)
print("Probability:", lr_prob)

print("\n✅ Test completed successfully.")
