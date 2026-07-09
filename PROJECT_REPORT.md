# Project Report: Smart Electricity Consumption Prediction & Energy Optimization System

**Task ID:** ML-3 | Teyzix Core Internship (June Batch)
**Domain:** Machine Learning — Advanced (Industry-Based)

---

## 1. Problem Statement
Rising electricity costs and growing energy demand make it valuable for
households to understand and anticipate their consumption. This project
builds a system that predicts daily electricity consumption from
household/appliance usage patterns, forecasts monthly bills, and provides
personalized recommendations to reduce usage.

## 2. Dataset Collection Method
As required, **no public dataset was used**. An original dataset was
generated via `data/generate_dataset.py` using a **physically-grounded
synthetic simulation**: each household's consumption is built bottom-up
from real-world appliance power ratings (AC ≈ 1.5 kW, fan ≈ 75 W, fridge
≈ 150 W, washing machine ≈ 500 W, water motor ≈ 750 W, lighting ≈ 60 W per
room) multiplied by simulated usage hours, with seasonal, household-size,
and weekday/holiday behavioral factors layered on top, plus 5% random
noise. Full methodology is in `data/dataset_documentation.md`.

- **658 records**, 14 input features + 1 target (`Daily_Electricity_Consumption_kWh`)
- Deliberately includes ~2% missing values, 8 duplicate rows, and 6 extreme
  outliers to make preprocessing meaningful.

## 3. Feature Description
See the full feature table in `data/dataset_documentation.md`. Key
features: House Type, Family Members, Number of Rooms, AC/Fan/Fridge/
Washing Machine/Water Motor/Lighting usage hours, Outdoor Temperature,
Day of Week, Season, and Holiday flag.

## 4. Data Preprocessing
- **Missing values**: median imputation on numeric columns.
- **Duplicates**: exact-row duplicates dropped (8 removed).
- **Outliers**: IQR-based detection and capping (winsorizing) on the
  target variable (6 outliers capped).
- **Encoding**: Label encoding for House_Type, Day_of_Week, Season.
- **Scaling**: StandardScaler applied to all numeric features (scaled
  columns kept alongside raw ones for flexibility).

## 5. Exploratory Data Analysis — Key Business Insights
- AC usage hours has the strongest correlation with consumption (r ≈ 0.91) — the single biggest lever for savings.
- Outdoor temperature is the second strongest driver (r ≈ 0.71).
- Summer shows the highest average daily consumption across seasons (≈18.9 kWh/day).
- Villas show the highest average consumption across house types (≈14.4 kWh/day), driven by more rooms/appliances.
- Holidays/weekends show noticeably higher consumption (≈14.4 kWh) than working days (≈12.2 kWh).

All supporting plots (distributions, correlation heatmap, boxplots,
scatter plots, category comparisons) are saved in `reports/eda_plots/`.

## 6. Feature Engineering
Derived features added: `Total_Appliance_Hours`, `Rooms_per_Person`,
`High_Temp_Flag`. Feature selection combined **correlation-with-target**
analysis and **Random Forest feature importance**, retaining 12 of 17
candidate features for final modeling (see `reports/feature_importance.csv`).

## 7. Model Selection & Comparison
Three regression algorithms were trained and compared on an 80/20 train-test split:

| Model | MAE | MSE | RMSE | R² Score |
|---|---|---|---|---|
| **Linear Regression** | 1.052 | 1.664 | 1.290 | **0.957** |
| Random Forest Regressor | 1.149 | 2.544 | 1.595 | 0.935 |
| Decision Tree Regressor | 1.294 | 3.266 | 1.807 | 0.916 |

**Linear Regression was selected as the best model.** This is expected
given the underlying data-generation process is largely additive
(appliance-hours × fixed power rating), so a linear relationship captures
the signal very well. Random Forest still underwent GridSearchCV
hyperparameter tuning as a bonus exercise, and both tree-based models
remain useful as they would likely generalize better to real-world data
with more non-linear interactions and noise than this synthetic set has.

## 8. Model Evaluation
Actual-vs-predicted scatter plots for all 3 models are saved in
`reports/actual_vs_predicted.png`, and a model-comparison bar chart in
`reports/model_comparison_r2.png`. Predictions track the diagonal closely
for the winning model, confirming the R² score.

## 9. Explainable AI (Bonus)
SHAP was used to interpret the best model's predictions
(`src/explainability.py`, output in `reports/shap_summary_plot.png`),
confirming AC usage hours as the dominant feature driving predictions,
consistent with the correlation/importance analysis.

## 10. Prediction System
A **Gradio** web interface (`app/app.py`) lets a user input household and
appliance usage details and returns:
- Predicted Daily Electricity Consumption (kWh)
- Estimated Monthly Electricity Usage (kWh)
- Peak Usage Hours (season/appliance-aware heuristic)
- Estimated Monthly Electricity Bill (PKR, configurable rate/kWh)
- Energy Efficiency Score (0–100, percentile-based vs. the reference dataset)

See `screenshots/gradio_app_screenshot.png` for a working screenshot.

## 11. Energy Optimization Recommendations
A rule-based recommendation engine (`src/recommendations.py`) generates
personalized tips: best times to run high-power appliances, estimated
potential monthly savings from a 10% usage reduction, peak-consumption
alerts, and appliance-specific efficiency suggestions.

## 12. Reports Generated
All in `reports/`, plus a combined `Energy_Project_Reports.xlsx`:
1. Dataset Summary
2. Model Performance Report
3. Energy Consumption Report (by season/house type)
4. Monthly Usage Forecast (10 sample households)
5. Recommendation Summary

## 13. Challenges Faced
- **Designing a synthetic dataset that is both original and realistic**
  required simulating real appliance power draw rather than pure random
  number generation, so that the resulting relationships would be
  meaningful for ML rather than noise.
- **Balancing signal vs. injected noise/outliers** so that preprocessing
  and feature engineering steps were genuinely necessary rather than
  cosmetic.
- **SHAP compute time** on a 150-sample background set required tuning
  sample size to keep runtime reasonable while still producing a
  meaningful summary plot.

## 14. Future Improvements
- Integrate a live weather API for real-time outdoor temperature instead
  of simulated values.
- Extend hyperparameter tuning across all three models, not just the
  winner.
- Deploy via Docker for a fully reproducible production environment.
- Collect real smart-meter or utility-billing data to validate and
  recalibrate the synthetic model against ground truth.
- Add time-series modeling (e.g., predicting a full week/month trajectory
  rather than a single day) for more actionable forecasting.

## 15. Technology Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, SHAP, Gradio,
Joblib, OpenPyXL.
