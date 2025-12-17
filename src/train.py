# src/train.py

import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.cluster import KMeans

# -------------------------------
# Load data
# -------------------------------
data_path = "data/processed/processed_data.csv"  # update if needed
df = pd.read_csv(data_path)

# -------------------------------
# Ensure required features exist
# -------------------------------
required_features = ["CountryCode", "Amount", "Value", "PricingStrategy"]
for col in required_features:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# -------------------------------
# Create RFM_Cluster if missing
# -------------------------------
if "RFM_Cluster" not in df.columns:
    print("RFM_Cluster not found. Creating proxy cluster using KMeans...")
    rfm_features = df[["Amount", "Value"]]  # adjust based on your RFM calculation
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["RFM_Cluster"] = kmeans.fit_predict(rfm_features)

# -------------------------------
# Ensure target exists
# -------------------------------
if "Target" not in df.columns:
    df["Target"] = df["RFM_Cluster"]

# -------------------------------
# Prepare features and target
# -------------------------------
features = ["CountryCode", "Amount", "Value", "PricingStrategy", "RFM_Cluster"]
X = df[features]
y = df["Target"]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# MLflow experiment
# -------------------------------
mlflow.set_experiment("Credit_Risk_Model")

with mlflow.start_run():

    # ---------------------------
    # Hyperparameter Tuning
    # ---------------------------
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1]
    }

    grid = GridSearchCV(
        GradientBoostingClassifier(),
        param_grid,
        cv=5,
        scoring="roc_auc_ovr",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    # Best model and params
    best_model = grid.best_estimator_
    best_params = grid.best_params_
    mlflow.log_params(best_params)
    mlflow.log_metric("best_score", grid.best_score_)

    # ---------------------------
    # Model Evaluation (multiclass)
    # ---------------------------
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')


    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    # ---------------------------
    # Save model
    # ---------------------------
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/gradient_boosting.pkl")
    mlflow.sklearn.log_model(best_model, "gradient_boosting_model")

    print("✅ Gradient Boosting model trained and logged successfully.")
    print("Best Parameters:", best_params)
    print("Evaluation Metrics:")
    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
