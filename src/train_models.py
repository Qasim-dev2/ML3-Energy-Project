"""
train_models.py
----------------
Trains and compares 3 regression models:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
Selects the best model by R2/RMSE, saves it, and produces evaluation
plots (actual vs predicted) + a metrics comparison table.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

DATA_PATH = "/home/claude/energy_project/data/energy_dataset_model_ready.csv"
MODEL_DIR = "/home/claude/energy_project/models"
REPORT_DIR = "/home/claude/energy_project/reports"
TARGET = "Daily_Electricity_Consumption_kWh"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=[TARGET])
y = df[TARGET]
FEATURE_COLS = X.columns.tolist()
with open(os.path.join(MODEL_DIR, "feature_columns.txt"), "w") as f:
    f.write("\n".join(FEATURE_COLS))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42, max_depth=8),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=300, random_state=42, max_depth=12),
}

results = []
predictions = {}
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    results.append({"Model": name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2_Score": r2})
    predictions[name] = preds
    trained_models[name] = model
    print(f"{name}: MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.4f}")

results_df = pd.DataFrame(results).sort_values("R2_Score", ascending=False)
results_df.to_csv(os.path.join(REPORT_DIR, "model_performance.csv"), index=False)
print("\nModel comparison:\n", results_df)

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]
print(f"\nBest model selected: {best_model_name}")

# ---- Optional bonus: hyperparameter tuning on the best model (Random Forest) ----
if best_model_name == "Random Forest Regressor":
    param_grid = {
        "n_estimators": [200, 300, 400],
        "max_depth": [8, 12, 16, None],
        "min_samples_split": [2, 5],
    }
    grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3,
                         scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)
    tuned_model = grid.best_estimator_
    tuned_preds = tuned_model.predict(X_test)
    tuned_r2 = r2_score(y_test, tuned_preds)
    print(f"Tuned RF best params: {grid.best_params_}, tuned R2={tuned_r2:.4f}")
    if tuned_r2 >= results_df.iloc[0]["R2_Score"]:
        best_model = tuned_model
        predictions[best_model_name] = tuned_preds
        with open(os.path.join(REPORT_DIR, "hyperparameter_tuning.txt"), "w") as f:
            f.write(f"Best params: {grid.best_params_}\nTuned R2: {tuned_r2:.4f}\n")

joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
with open(os.path.join(MODEL_DIR, "best_model_name.txt"), "w") as f:
    f.write(best_model_name)

# ---- Actual vs Predicted plots for all models ----
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, preds) in zip(axes, predictions.items()):
    ax.scatter(y_test, preds, alpha=0.5, color="#2e7d32")
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    ax.plot(lims, lims, "r--")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(name)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "actual_vs_predicted.png"), dpi=110)
plt.close()

# Model comparison bar chart
plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["R2_Score"], color="#43a047")
plt.ylabel("R2 Score")
plt.title("Model Comparison (R2 Score)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "model_comparison_r2.png"), dpi=110)
plt.close()

print("\nTraining complete. Best model saved to models/best_model.pkl")
