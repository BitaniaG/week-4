# src/data_processing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from feature_engine.encoding import WoEEncoder

def preprocess_data(df):
    # Numeric columns
    num_features = ['Amount', 'Value']
    for col in num_features:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)

    # Categorical columns
    cat_features = ['PricingStrategy']
    for col in cat_features:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Standardize numeric columns
    scaler = StandardScaler()
    df[num_features] = scaler.fit_transform(df[num_features])

    # WoE encoding (if is_high_risk exists)
    for col in cat_features:
        df[col] = df[col].astype(str)

    if 'is_high_risk' in df.columns:
        X = df.drop(columns='is_high_risk')
        y = df['is_high_risk']
        encoder = WoEEncoder(variables=cat_features)
        encoder.fit(X, y)
        X_woe = encoder.transform(X)
        df[X_woe.columns] = X_woe

    return df
