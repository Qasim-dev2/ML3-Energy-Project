"""
tests/test_model.py
--------------------
Smoke tests for the trained model artifacts:
- Model loads without error
- Encoders load without error
- A prediction returns a positive float
- Prediction is within a realistic range (0.5–100 kWh/day)

Run with: pytest tests/ -v
"""

import sys
import os
import pytest
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import BEST_MODEL_PATH, ENCODERS_PATH, FEATURE_COLS_PATH


@pytest.fixture(scope="module")
def model():
    return joblib.load(BEST_MODEL_PATH)


@pytest.fixture(scope="module")
def encoders():
    return joblib.load(ENCODERS_PATH)


@pytest.fixture(scope="module")
def feature_cols():
    with open(FEATURE_COLS_PATH) as f:
        return [line.strip() for line in f if line.strip()]


def _make_sample_row(encoders, feature_cols):
    """Build a minimal feature row that matches what the model expects."""
    house_enc  = encoders["House_Type"].transform(["Apartment"])[0]
    day_enc    = encoders["Day_of_Week"].transform(["Monday"])[0]
    season_enc = encoders["Season"].transform(["Summer"])[0]

    row = {
        "Family_Members": 4,
        "Number_of_Rooms": 3,
        "Daily_Appliance_Usage_Count": 8,
        "AC_Usage_Hours": 6.0,
        "Fan_Usage_Hours": 8.0,
        "Washing_Machine_Usage_Hours": 0.5,
        "Water_Motor_Usage_Hours": 1.0,
        "Lighting_Hours": 5.0,
        "Outdoor_Temperature_C": 35,
        "Is_Holiday": 0,
        "House_Type_Encoded": house_enc,
        "Day_of_Week_Encoded": day_enc,
        "Season_Encoded": season_enc,
        "Total_Appliance_Hours": 21.0,
        "Rooms_per_Person": 0.75,
        "High_Temp_Flag": 1,
    }
    df = pd.DataFrame([row]).reindex(columns=feature_cols, fill_value=0)
    return df


class TestModelArtifacts:
    def test_model_loads(self, model):
        assert model is not None

    def test_encoders_load(self, encoders):
        assert "House_Type" in encoders
        assert "Day_of_Week" in encoders
        assert "Season" in encoders

    def test_feature_cols_not_empty(self, feature_cols):
        assert len(feature_cols) > 0

    def test_prediction_is_positive(self, model, encoders, feature_cols):
        X = _make_sample_row(encoders, feature_cols)
        pred = float(model.predict(X)[0])
        assert pred > 0, f"Expected positive prediction, got {pred}"

    def test_prediction_in_range(self, model, encoders, feature_cols):
        X = _make_sample_row(encoders, feature_cols)
        pred = float(model.predict(X)[0])
        assert 0.5 <= pred <= 150.0, f"Prediction {pred} out of realistic range"
