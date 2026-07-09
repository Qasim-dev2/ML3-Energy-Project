"""
feature_engineering.py
-----------------------
Feature selection, transformation, correlation-based pruning, and
feature-importance evaluation (via a quick Random Forest).
Outputs the final model-ready feature matrix.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os

DATA_PATH = "/home/claude/energy_project/data/energy_dataset_cleaned.csv"
OUT_PATH = "/home/claude/energy_project/data/energy_dataset_model_ready.csv"
REPORT_DIR = "/home/claude/energy_project/reports"
TARGET = "Daily_Electricity_Consumption_kWh"

df = pd.read_csv(DATA_PATH)

# Derived / engineered features
df["Total_Appliance_Hours"] = (
    df["AC_Usage_Hours"] + df["Fan_Usage_Hours"] + df["Refrigerator_Usage_Hours"]
    + df["Washing_Machine_Usage_Hours"] + df["Water_Motor_Usage_Hours"] + df["Lighting_Hours"]
)
df["Rooms_per_Person"] = df["Number_of_Rooms"] / df["Family_Members"].replace(0, 1)
df["High_Temp_Flag"] = (df["Outdoor_Temperature_C"] > 30).astype(int)
df["Weekend_Or_Holiday"] = df["Is_Holiday"]

# Candidate feature set for modeling (encoded categoricals + engineered + raw numeric)
FEATURE_COLS = [
    "Family_Members", "Number_of_Rooms", "Daily_Appliance_Usage_Count",
    "AC_Usage_Hours", "Fan_Usage_Hours", "Refrigerator_Usage_Hours",
    "Washing_Machine_Usage_Hours", "Water_Motor_Usage_Hours", "Lighting_Hours",
    "Outdoor_Temperature_C", "Is_Holiday",
    "House_Type_Encoded", "Day_of_Week_Encoded", "Season_Encoded",
    "Total_Appliance_Hours", "Rooms_per_Person", "High_Temp_Flag"
]

X = df[FEATURE_COLS]
y = df[TARGET]

# Correlation-based check (drop anything with near-zero correlation to target)
corrs = X.assign(**{TARGET: y}).corr()[TARGET].drop(TARGET).abs().sort_values(ascending=False)
print("Feature correlation with target:\n", corrs)

# Quick RF-based importance for feature selection sanity check
rf_quick = RandomForestRegressor(n_estimators=200, random_state=42)
rf_quick.fit(X, y)
importances = pd.Series(rf_quick.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
print("\nFeature importances:\n", importances)

os.makedirs(REPORT_DIR, exist_ok=True)
importances.to_csv(os.path.join(REPORT_DIR, "feature_importance.csv"), header=["importance"])
corrs.to_csv(os.path.join(REPORT_DIR, "feature_target_correlation.csv"), header=["abs_correlation"])

# Keep all features that have at least minor signal (importance/corr threshold)
selected_features = importances[importances > 0.005].index.tolist()
print(f"\nSelected {len(selected_features)} / {len(FEATURE_COLS)} features for modeling.")

final_df = df[selected_features + [TARGET]].copy()
final_df.to_csv(OUT_PATH, index=False)

with open(os.path.join(REPORT_DIR, "selected_features.txt"), "w") as f:
    f.write("\n".join(selected_features))

print("Model-ready dataset saved:", OUT_PATH, final_df.shape)
