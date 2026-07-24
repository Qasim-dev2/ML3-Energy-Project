"""
config.py
---------
Centralized configuration for the Smart Electricity Prediction project.
All paths, constants, and hyperparameters are defined here so that
individual scripts can import from a single source of truth.
"""

import os

# ── Root paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR  = os.path.join(PROJECT_ROOT, "reports")

# ── Dataset files ─────────────────────────────────────────────────────────────
RAW_DATA_PATH        = os.path.join(DATA_DIR, "energy_dataset.csv")
CLEANED_DATA_PATH    = os.path.join(DATA_DIR, "energy_dataset_cleaned.csv")
MODEL_READY_PATH     = os.path.join(DATA_DIR, "energy_dataset_model_ready.csv")

# ── Model artifacts ───────────────────────────────────────────────────────────
BEST_MODEL_PATH      = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PATH          = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODERS_PATH        = os.path.join(MODEL_DIR, "encoders.pkl")
FEATURE_COLS_PATH    = os.path.join(MODEL_DIR, "feature_columns.txt")
BEST_MODEL_NAME_PATH = os.path.join(MODEL_DIR, "best_model_name.txt")

# ── Target column ─────────────────────────────────────────────────────────────
TARGET = "Daily_Electricity_Consumption_kWh"

# ── Billing ───────────────────────────────────────────────────────────────────
RATE_PER_KWH   = 35.0    # PKR per kWh (flat estimate)
DAYS_IN_MONTH  = 30

# ── Dataset generation ────────────────────────────────────────────────────────
DATASET_ROWS   = 658
RANDOM_SEED    = 42

# ── Model training ────────────────────────────────────────────────────────────
TEST_SIZE      = 0.2
CV_FOLDS       = 5

# ── App ───────────────────────────────────────────────────────────────────────
APP_HOST       = "0.0.0.0"
APP_PORT       = 7860
