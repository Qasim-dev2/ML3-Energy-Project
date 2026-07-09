"""
explainability.py
------------------
Bonus feature: Explainable AI using SHAP to interpret the best model's
predictions and produce a feature-importance / summary plot.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import joblib
import os

DATA_PATH = "/home/claude/energy_project/data/energy_dataset_model_ready.csv"
MODEL_DIR = "/home/claude/energy_project/models"
REPORT_DIR = "/home/claude/energy_project/reports"
TARGET = "Daily_Electricity_Consumption_kWh"

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=[TARGET])

model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
with open(os.path.join(MODEL_DIR, "best_model_name.txt")) as f:
    best_model_name = f.read().strip()

print(f"Generating SHAP explanations for: {best_model_name}")

sample = X.sample(min(150, len(X)), random_state=42)

try:
    explainer = shap.Explainer(model.predict, sample)
    shap_values = explainer(sample)
except Exception as e:
    print("Falling back to KernelExplainer:", e)
    explainer = shap.KernelExplainer(model.predict, shap.sample(sample, 50))
    shap_values = explainer(sample)

plt.figure()
shap.summary_plot(shap_values, sample, show=False)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "shap_summary_plot.png"), dpi=110, bbox_inches="tight")
plt.close()

print("SHAP summary plot saved to reports/shap_summary_plot.png")
