"""
recommendations.py
--------------------
Rule-based energy optimization recommendation engine + bill/efficiency
score calculation. Used by the Gradio app and CLI.
"""

import os
import sys
import pandas as pd
import numpy as np

# Allow importing config from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import MODEL_READY_PATH, TARGET, RATE_PER_KWH, DAYS_IN_MONTH

# Reference distribution (used for efficiency percentile scoring)
_df = pd.read_csv(MODEL_READY_PATH)
_consumption_dist = _df[TARGET].values


def estimate_bill(daily_kwh, days_in_month=DAYS_IN_MONTH, rate=RATE_PER_KWH):
    """Return (monthly_kwh, estimated_bill_PKR) for a given daily consumption."""
    monthly_kwh = daily_kwh * days_in_month
    bill = monthly_kwh * rate
    return monthly_kwh, bill


def efficiency_score(daily_kwh):
    # Percentile rank vs reference dataset distribution, inverted so LOWER
    # consumption -> HIGHER efficiency score (0-100)
    percentile = (np.sum(_consumption_dist <= daily_kwh) / len(_consumption_dist)) * 100
    score = max(0, min(100, 100 - percentile))
    return round(score, 1)


def peak_usage_hours(ac_hours, fan_hours, season):
    # Simple heuristic: in hot seasons, peak load is midday-evening;
    # in cooler seasons, peak shifts to evening lighting/appliance use.
    if season == "Summer" and ac_hours > 4:
        return "1:00 PM - 6:00 PM (AC-driven peak)"
    elif season == "Winter":
        return "6:00 PM - 9:00 PM (evening lighting/heating peak)"
    else:
        return "5:00 PM - 8:00 PM (evening household peak)"


def generate_recommendations(inputs, daily_kwh, monthly_kwh, bill, eff_score):
    recs = []

    if inputs["ac_hours"] > 6:
        recs.append("Your AC usage is high — shifting non-essential AC hours to "
                     "after 9 PM (cooler ambient temps) can reduce load by ~10-15%.")
    if inputs["season"] == "Summer" and inputs["ac_hours"] > 4:
        recs.append("Peak alert: heavy AC usage during 1 PM-6 PM coincides with "
                     "typical grid peak hours — consider raising thermostat by 1-2°C.")
    if inputs["washing_hours"] > 0:
        recs.append("Run washing machine and water motor during off-peak hours "
                     "(before 8 AM or after 10 PM) for lower effective cost if "
                     "time-of-use tariffs apply.")
    if inputs["lighting_hours"] > 7:
        recs.append("Lighting hours are on the higher side — switching to LED "
                     "bulbs (if not already) can cut lighting load by up to 80%.")
    if eff_score < 40:
        recs.append("Your household's efficiency score is below average. Focus "
                     "first on AC and fan run-time, since these dominate your "
                     "total consumption.")
    else:
        recs.append("Your household is operating at a reasonably efficient level "
                     "compared to similar homes — maintain current usage patterns.")

    # Estimated potential savings if AC/fan hours reduced by 20%
    reduced_kwh = daily_kwh * 0.90  # heuristic 10% reduction achievable
    _, reduced_bill = estimate_bill(reduced_kwh)
    potential_monthly_savings = round(bill - reduced_bill, 2)
    recs.append(f"Estimated potential monthly savings if usage is optimized "
                f"(~10% reduction): PKR {potential_monthly_savings:,.2f}")

    return recs
