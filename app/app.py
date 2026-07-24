"""
app.py
------
Gradio interface for the Smart Electricity Consumption Prediction &
Energy Optimization System.

Run with: python3 app.py
"""

import sys
import os
import csv
import logging
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import joblib
import gradio as gr
from recommendations import estimate_bill, efficiency_score, peak_usage_hours, generate_recommendations
from config import APP_HOST, APP_PORT, MODEL_DIR as CFG_MODEL_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Prediction history log path
LOG_DIR  = os.path.join(os.path.dirname(__file__), "..", "logs")
LOG_FILE = os.path.join(LOG_DIR, "predictions.csv")
os.makedirs(LOG_DIR, exist_ok=True)


def _append_prediction_log(inputs_dict: dict, daily_kwh: float, bill: float, eff: float):
    """Append one prediction row to the CSV history log."""
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "house_type", "family_members", "ac_hours",
            "season", "outdoor_temp", "daily_kwh", "monthly_bill_pkr", "efficiency_score"
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp":        datetime.now().isoformat(timespec="seconds"),
            "house_type":       inputs_dict.get("house_type", ""),
            "family_members":   inputs_dict.get("family_members", ""),
            "ac_hours":         inputs_dict.get("ac_hours", ""),
            "season":           inputs_dict.get("season", ""),
            "outdoor_temp":     inputs_dict.get("outdoor_temp", ""),
            "daily_kwh":        round(daily_kwh, 3),
            "monthly_bill_pkr": round(bill, 2),
            "efficiency_score": eff,
        })
    log.info("Prediction logged → %s", LOG_FILE)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
with open(os.path.join(MODEL_DIR, "feature_columns.txt")) as f:
    FEATURE_COLS = [line.strip() for line in f if line.strip()]
with open(os.path.join(MODEL_DIR, "best_model_name.txt")) as f:
    BEST_MODEL_NAME = f.read().strip()

HOUSE_TYPES = list(encoders["House_Type"].classes_)
DAYS = list(encoders["Day_of_Week"].classes_)
SEASONS = list(encoders["Season"].classes_)


def build_feature_row(house_type, family_members, rooms, appliance_count,
                       ac_hours, fan_hours, fridge_hours, washing_hours,
                       motor_hours, lighting_hours, outdoor_temp, day, season,
                       is_holiday):
    house_enc = encoders["House_Type"].transform([house_type])[0]
    day_enc = encoders["Day_of_Week"].transform([day])[0]
    season_enc = encoders["Season"].transform([season])[0]

    total_appliance_hours = (ac_hours + fan_hours + fridge_hours + washing_hours
                              + motor_hours + lighting_hours)
    rooms_per_person = rooms / max(family_members, 1)
    high_temp_flag = 1 if outdoor_temp > 30 else 0

    row = {
        "Family_Members": family_members,
        "Number_of_Rooms": rooms,
        "Daily_Appliance_Usage_Count": appliance_count,
        "AC_Usage_Hours": ac_hours,
        "Fan_Usage_Hours": fan_hours,
        "Washing_Machine_Usage_Hours": washing_hours,
        "Water_Motor_Usage_Hours": motor_hours,
        "Lighting_Hours": lighting_hours,
        "Outdoor_Temperature_C": outdoor_temp,
        "Is_Holiday": int(is_holiday),
        "House_Type_Encoded": house_enc,
        "Day_of_Week_Encoded": day_enc,
        "Season_Encoded": season_enc,
        "Total_Appliance_Hours": total_appliance_hours,
        "Rooms_per_Person": rooms_per_person,
        "High_Temp_Flag": high_temp_flag,
    }
    df_row = pd.DataFrame([row])
    # ensure exact column order/subset expected by the model
    df_row = df_row.reindex(columns=FEATURE_COLS, fill_value=0)
    return df_row


def predict(house_type, family_members, rooms, appliance_count,
            ac_hours, fan_hours, fridge_hours, washing_hours,
            motor_hours, lighting_hours, outdoor_temp, day, season, is_holiday):

    X_row = build_feature_row(house_type, family_members, rooms, appliance_count,
                               ac_hours, fan_hours, fridge_hours, washing_hours,
                               motor_hours, lighting_hours, outdoor_temp, day,
                               season, is_holiday)

    daily_kwh = float(model.predict(X_row)[0])
    daily_kwh = max(daily_kwh, 0.5)
    monthly_kwh, bill = estimate_bill(daily_kwh)
    eff_score = efficiency_score(daily_kwh)
    peak = peak_usage_hours(ac_hours, fan_hours, season)

    inputs = {
        "ac_hours": ac_hours, "fan_hours": fan_hours, "season": season,
        "washing_hours": washing_hours, "lighting_hours": lighting_hours,
    }
    recs = generate_recommendations(inputs, daily_kwh, monthly_kwh, bill, eff_score)

    inputs_log = {
        "house_type": house_type, "family_members": family_members,
        "ac_hours": ac_hours, "season": season, "outdoor_temp": outdoor_temp,
    }
    _append_prediction_log(inputs_log, daily_kwh, bill, eff_score)

    result_md = f"""
### Prediction Results (Model: {BEST_MODEL_NAME})
| Metric | Value |
|---|---|
| Predicted Daily Consumption | **{daily_kwh:.2f} kWh** |
| Estimated Monthly Usage | **{monthly_kwh:.2f} kWh** |
| Estimated Monthly Bill | **PKR {bill:,.2f}** |
| Peak Usage Hours | **{peak}** |
| Energy Efficiency Score | **{eff_score}/100** |
"""
    recs_md = "### Personalized Energy Optimization Recommendations\n" + \
              "\n".join(f"- {r}" for r in recs)

    return result_md, recs_md


with gr.Blocks(title="Smart Electricity Consumption Prediction System") as demo:
    gr.Markdown("# ⚡ Smart Electricity Consumption Prediction & Energy Optimization System")
    gr.Markdown("Enter your household's appliance usage details to get a consumption "
                "prediction, bill estimate, and personalized energy-saving tips.")

    with gr.Row():
        with gr.Column():
            house_type = gr.Dropdown(HOUSE_TYPES, label="House Type", value=HOUSE_TYPES[0])
            family_members = gr.Slider(1, 10, value=4, step=1, label="Family Members")
            rooms = gr.Slider(1, 12, value=4, step=1, label="Number of Rooms")
            appliance_count = gr.Slider(1, 20, value=8, step=1, label="Daily Appliance Usage Count")
            ac_hours = gr.Slider(0, 16, value=4, step=0.5, label="AC Usage Hours")
            fan_hours = gr.Slider(0, 20, value=6, step=0.5, label="Fan Usage Hours")
        with gr.Column():
            fridge_hours = gr.Slider(0, 24, value=24, step=1, label="Refrigerator Usage Hours")
            washing_hours = gr.Slider(0, 3, value=0.5, step=0.1, label="Washing Machine Usage Hours")
            motor_hours = gr.Slider(0, 4, value=1.0, step=0.1, label="Water Motor Usage Hours")
            lighting_hours = gr.Slider(0, 14, value=6, step=0.5, label="Lighting Hours")
            outdoor_temp = gr.Slider(-5, 50, value=25, step=1, label="Outdoor Temperature (°C)")
            day = gr.Dropdown(DAYS, label="Day of Week", value=DAYS[0])
            season = gr.Dropdown(SEASONS, label="Season", value=SEASONS[0])
            is_holiday = gr.Checkbox(label="Working Day is a Holiday/Weekend?")

    predict_btn = gr.Button("Predict Consumption & Get Recommendations", variant="primary")
    result_out = gr.Markdown()
    rec_out = gr.Markdown()

    predict_btn.click(
        predict,
        inputs=[house_type, family_members, rooms, appliance_count, ac_hours,
                fan_hours, fridge_hours, washing_hours, motor_hours, lighting_hours,
                outdoor_temp, day, season, is_holiday],
        outputs=[result_out, rec_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
