"""
preprocessing.py
-----------------
Data cleaning & preprocessing pipeline:
- Missing value handling
- Duplicate removal
- Outlier detection & treatment
- Feature encoding
- Feature scaling / normalization
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

RAW_PATH = "/home/claude/energy_project/data/energy_dataset.csv"
OUT_DIR = "/home/claude/energy_project/data"
MODEL_DIR = "/home/claude/energy_project/models"


def load_data():
    return pd.read_csv(RAW_PATH)


def handle_missing_values(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    return df


def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)
    print(f"Removed {before - after} duplicate rows")
    return df


def handle_outliers(df, target_col="Daily_Electricity_Consumption_kWh"):
    # IQR-based capping (winsorizing) rather than dropping, to preserve data volume
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((df[target_col] < lower) | (df[target_col] > upper)).sum()
    print(f"Detected {n_outliers} outliers in {target_col}; capping to [{lower:.2f}, {upper:.2f}]")
    df[target_col] = df[target_col].clip(lower=lower, upper=upper)
    return df


def encode_features(df):
    categorical_cols = ["House_Type", "Day_of_Week", "Season"]
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + "_Encoded"] = le.fit_transform(df[col])
        encoders[col] = le
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))
    return df, encoders


def scale_features(df, feature_cols):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[feature_cols])
    scaled_df = pd.DataFrame(scaled, columns=[c + "_Scaled" for c in feature_cols])
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    return pd.concat([df.reset_index(drop=True), scaled_df], axis=1), scaler


def run_pipeline():
    df = load_data()
    print("Initial shape:", df.shape)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = handle_outliers(df)
    df, encoders = encode_features(df)

    numeric_feature_cols = [
        "Family_Members", "Number_of_Rooms", "Daily_Appliance_Usage_Count",
        "AC_Usage_Hours", "Fan_Usage_Hours", "Refrigerator_Usage_Hours",
        "Washing_Machine_Usage_Hours", "Water_Motor_Usage_Hours",
        "Lighting_Hours", "Outdoor_Temperature_C"
    ]
    df, scaler = scale_features(df, numeric_feature_cols)

    df.to_csv(os.path.join(OUT_DIR, "energy_dataset_cleaned.csv"), index=False)
    print("Final cleaned shape:", df.shape)
    return df


if __name__ == "__main__":
    run_pipeline()
