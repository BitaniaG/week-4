import requests
import pandas as pd

# -----------------------------
# API URL
# -----------------------------
API_URL = "http://127.0.0.1:8000/predict"

# -----------------------------
# Batch of transactions
# -----------------------------
batch_data = [
    {"CountryCode": 251, "Amount": 1500, "Value": 3, "PricingStrategy": 1},
    {"CountryCode": 44, "Amount": 200, "Value": 1, "PricingStrategy": 2},
    {"CountryCode": 1, "Amount": 500, "Value": 2, "PricingStrategy": 1},
    {"CountryCode": 2, "Amount": -100, "Value": 1, "PricingStrategy": 1}  # Example invalid input
]

# -----------------------------
# Collect predictions with error handling
# -----------------------------
results = []

for i, txn in enumerate(batch_data, start=1):
    try:
        response = requests.post(API_URL, json=txn)
        response.raise_for_status()  # Raises for 4xx/5xx HTTP errors
        pred = response.json()
        # Check if API returned an error message
        if "detail" in pred:
            print(f"Transaction {i} API error:", pred["detail"])
            results.append({**txn, "error": pred["detail"]})
        else:
            results.append({**txn, **pred})
    except requests.exceptions.RequestException as e:
        print(f"Transaction {i} HTTP request error:", e)
        results.append({**txn, "error": str(e)})
    except Exception as e:
        print(f"Transaction {i} unexpected error:", e)
        results.append({**txn, "error": str(e)})

# -----------------------------
# Convert to DataFrame and save
# -----------------------------
df_results = pd.DataFrame(results)
output_path = "batch_predictions.csv"
df_results.to_csv(output_path, index=False)

print(f"\nAll predictions saved to {output_path}")
print(df_results)
