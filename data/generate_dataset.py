"""
generate_dataset.py
--------------------
Original synthetic dataset generator for the Smart Electricity Consumption
Prediction & Energy Optimization System (Task ML-3, Teyzix Core Internship).

Why synthetic (and NOT a public dataset):
Task rules explicitly forbid using Kaggle/UCI/GitHub datasets. Since we don't
have access to real smart-meter data for 500+ households, we simulate a
realistic household energy-usage dataset using domain knowledge of how
appliances actually consume power (rated wattage x hours used), plus
believable behavioral/seasonal variation and random household-level noise.

Method: each household's daily electricity consumption is built bottom-up
from appliance-level energy draw (in kWh) using approximate real-world
appliance power ratings, then behavioral multipliers (season, weekday/
holiday, family size) and 5% random measurement noise are layered on top.
This is documented fully in dataset_documentation.md.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 650  # > 500 required, gives buffer after dropping duplicates/na rows

house_types = ["Apartment", "Independent House", "Villa"]
seasons = ["Summer", "Winter", "Spring/Autumn"]
days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Approx real-world appliance power ratings (kW) used to simulate physically
# plausible consumption instead of pure random numbers.
AC_KW = 1.5
FAN_KW = 0.075
FRIDGE_KW = 0.15   # runs ~24h cycling, rated per hour equivalent
WASHING_KW = 0.5
MOTOR_KW = 0.75
LIGHT_KW = 0.06

rows = []
for i in range(N):
    house_type = np.random.choice(house_types, p=[0.5, 0.35, 0.15])
    family_members = np.random.randint(1, 8)
    if house_type == "Apartment":
        rooms = np.random.randint(2, 5)
    elif house_type == "Independent House":
        rooms = np.random.randint(3, 7)
    else:
        rooms = np.random.randint(5, 10)

    season = np.random.choice(seasons, p=[0.4, 0.3, 0.3])
    day = np.random.choice(days_of_week)
    is_holiday = 1 if (day in ["Saturday", "Sunday"] or np.random.rand() < 0.05) else 0

    # Outdoor temperature depends on season (deg C)
    if season == "Summer":
        outdoor_temp = np.random.normal(38, 4)
        ac_hours = np.clip(np.random.normal(7 + is_holiday * 1.5, 2), 0, 16)
        fan_hours = np.clip(np.random.normal(10, 2), 0, 20)
    elif season == "Winter":
        outdoor_temp = np.random.normal(12, 4)
        ac_hours = np.clip(np.random.normal(0.5, 1), 0, 4)
        fan_hours = np.clip(np.random.normal(2, 1.5), 0, 8)
    else:
        outdoor_temp = np.random.normal(24, 4)
        ac_hours = np.clip(np.random.normal(2, 1.5), 0, 8)
        fan_hours = np.clip(np.random.normal(6, 2), 0, 14)

    appliance_count = np.random.randint(4, 15)
    fridge_usage_hours = 24  # always on
    washing_machine_hours = np.clip(np.random.poisson(0.6) * np.random.uniform(0.5, 1.2), 0, 3)
    water_motor_hours = np.clip(np.random.normal(1 + 0.1 * rooms, 0.5), 0, 4)
    lighting_hours = np.clip(np.random.normal(5 + 0.3 * rooms, 1.5), 1, 14)

    daily_appliance_usage = appliance_count  # count of distinct appliances used that day

    # ---- physically-grounded energy calculation (kWh) ----
    e_ac = ac_hours * AC_KW * np.random.uniform(0.9, 1.1)
    e_fan = fan_hours * FAN_KW * np.random.uniform(0.9, 1.1)
    e_fridge = fridge_usage_hours * FRIDGE_KW
    e_wash = washing_machine_hours * WASHING_KW
    e_motor = water_motor_hours * MOTOR_KW
    e_light = lighting_hours * LIGHT_KW * rooms * 0.5

    family_factor = 1 + 0.04 * (family_members - 3)   # bigger families -> more misc load
    weekday_factor = 1.08 if is_holiday else 1.0        # more usage on holidays/weekends

    base_consumption = (e_ac + e_fan + e_fridge + e_wash + e_motor + e_light)
    daily_consumption = base_consumption * family_factor * weekday_factor
    daily_consumption *= np.random.normal(1.0, 0.05)    # 5% measurement noise
    daily_consumption = max(daily_consumption, 1.0)

    rows.append({
        "House_Type": house_type,
        "Family_Members": family_members,
        "Number_of_Rooms": rooms,
        "Daily_Appliance_Usage_Count": daily_appliance_usage,
        "AC_Usage_Hours": round(ac_hours, 2),
        "Fan_Usage_Hours": round(fan_hours, 2),
        "Refrigerator_Usage_Hours": fridge_usage_hours,
        "Washing_Machine_Usage_Hours": round(washing_machine_hours, 2),
        "Water_Motor_Usage_Hours": round(water_motor_hours, 2),
        "Lighting_Hours": round(lighting_hours, 2),
        "Outdoor_Temperature_C": round(outdoor_temp, 1),
        "Day_of_Week": day,
        "Season": season,
        "Is_Holiday": is_holiday,
        "Daily_Electricity_Consumption_kWh": round(daily_consumption, 2),
    })

df = pd.DataFrame(rows)

# ---- Intentionally inject realistic data-quality issues for the
# preprocessing stage to handle (missing values, duplicates, outliers) ----
missing_cols = ["Outdoor_Temperature_C", "Washing_Machine_Usage_Hours", "Lighting_Hours"]
for col in missing_cols:
    idx = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
    df.loc[idx, col] = np.nan

# add a handful of duplicate rows
dupes = df.sample(8, random_state=1)
df = pd.concat([df, dupes], ignore_index=True)

# inject a few extreme outliers
outlier_idx = np.random.choice(df.index, size=6, replace=False)
df.loc[outlier_idx, "Daily_Electricity_Consumption_kWh"] *= np.random.uniform(3, 5)

df = df.sample(frac=1, random_state=7).reset_index(drop=True)  # shuffle

df.to_csv("/home/claude/energy_project/data/energy_dataset.csv", index=False)
print("Dataset generated:", df.shape)
print(df.head())
