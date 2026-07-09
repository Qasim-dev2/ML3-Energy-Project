import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md("""# Smart Electricity Consumption Prediction & Energy Optimization System
### Task ID: ML-3 — Teyzix Core Internship (June Batch)

This notebook walks through the complete machine learning lifecycle:
dataset generation, preprocessing, EDA, feature engineering, model
training & comparison, evaluation, prediction, and energy optimization
recommendations.

**Note on dataset**: per task rules, no public dataset (Kaggle/UCI/GitHub)
was used. An original synthetic dataset was generated using a physically-
grounded simulation of household appliance energy draw — see
`data/dataset_documentation.md` for full methodology.
""")

code("""import os
os.chdir("/home/claude/energy_project")  # ensure paths resolve from project root
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
%matplotlib inline
""")

md("## 1. Dataset Creation (Original, Synthetic)\nSee `data/generate_dataset.py` for the full generator. Re-running it here for reproducibility.")

code("""import sys, os
sys.path.append("src")
exec(open("data/generate_dataset.py").read())
""")

md("## 2. Load Raw Dataset")
code("""df_raw = pd.read_csv("data/energy_dataset.csv")
print(df_raw.shape)
df_raw.head()
""")

md("## 3. Data Preprocessing\n- Missing value handling\n- Duplicate removal\n- Outlier detection (IQR capping)\n- Feature encoding\n- Feature scaling")

code("""exec(open("src/preprocessing.py").read())
""")

code("""df_clean = pd.read_csv("data/energy_dataset_cleaned.csv")
print("Cleaned shape:", df_clean.shape)
df_clean.head()
""")

md("## 4. Exploratory Data Analysis\nStatistical summary, distributions, correlation heatmap, boxplots, scatter plots. Plots are saved under `reports/eda_plots/`.")

code("""summary = df_clean.describe().T
summary
""")

code("""corr = df_clean[["Family_Members","Number_of_Rooms","AC_Usage_Hours","Fan_Usage_Hours",
                  "Outdoor_Temperature_C","Daily_Electricity_Consumption_kWh"]].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="Greens")
plt.title("Correlation Heatmap (key features)")
plt.show()
""")

code("""plt.figure(figsize=(7,5))
sns.scatterplot(x="AC_Usage_Hours", y="Daily_Electricity_Consumption_kWh", data=df_clean, alpha=0.5)
plt.title("AC Usage Hours vs Daily Consumption")
plt.show()
""")

md("""### Business Insights
- AC usage hours has the strongest correlation with daily consumption (r ≈ 0.91) — the single biggest lever for savings.
- Outdoor temperature is the second strongest driver (r ≈ 0.71), confirming a physically-grounded relationship.
- Summer shows the highest average consumption across seasons.
- Villas show the highest average consumption across house types (more rooms/appliances).
- Holidays/weekends show noticeably higher consumption than working days.
""")

md("## 5. Feature Engineering\nDerived features (Total Appliance Hours, Rooms per Person, High Temp Flag), correlation-based and Random-Forest-importance-based feature selection.")

code("""exec(open("src/feature_engineering.py").read())
""")

code("""df_model = pd.read_csv("data/energy_dataset_model_ready.csv")
print(df_model.shape)
df_model.head()
""")

md("## 6. Model Development & Comparison\nTraining Linear Regression, Decision Tree Regressor, and Random Forest Regressor; comparing with MAE, MSE, RMSE, R².")

code("""exec(open("src/train_models.py").read())
""")

code("""perf = pd.read_csv("reports/model_performance.csv")
perf
""")

md("## 7. Model Evaluation — Actual vs Predicted")
code("""from PIL import Image
img = Image.open("reports/actual_vs_predicted.png")
plt.figure(figsize=(14,5))
plt.imshow(img)
plt.axis("off")
plt.show()
""")

md("## 8. Explainable AI (Bonus) — SHAP")
code("""exec(open("src/explainability.py").read())
""")

code("""img2 = Image.open("reports/shap_summary_plot.png")
plt.figure(figsize=(8,6))
plt.imshow(img2)
plt.axis("off")
plt.show()
""")

md("""## 9. Prediction System & Energy Optimization Recommendations
A Gradio-based interface (`app/app.py`) lets users enter household usage
details and receive: predicted daily consumption, monthly usage estimate,
peak usage hours, estimated monthly bill, energy efficiency score, and
personalized optimization recommendations.

Run: `python3 app/app.py`
""")

code("""sys.path.append("src")
from recommendations import estimate_bill, efficiency_score, peak_usage_hours, generate_recommendations
import joblib

model = joblib.load("models/best_model.pkl")
sample = df_model.drop(columns=["Daily_Electricity_Consumption_kWh"]).iloc[[0]]
pred = model.predict(sample)[0]
monthly_kwh, bill = estimate_bill(pred)
eff = efficiency_score(pred)
print(f"Predicted Daily kWh: {pred:.2f}")
print(f"Monthly kWh: {monthly_kwh:.2f}, Estimated Bill: PKR {bill:,.2f}")
print(f"Efficiency Score: {eff}/100")
""")

md("## 10. Reports\nAll reports (dataset summary, model performance, consumption report, monthly forecast, recommendation summary) are generated in `reports/` and combined into `reports/Energy_Project_Reports.xlsx`.")

code("""exec(open("src/generate_reports.py").read())
""")

md("""## 11. Conclusion
Linear Regression achieved the best performance (R² ≈ 0.96) on this
dataset, which makes sense given the underlying data-generation process is
largely additive (appliance-hours × power-rating). Random Forest and
Decision Tree still perform strongly and offer better robustness on
non-linear interactions, which would likely matter more with real-world
(noisier, less additive) smart meter data.

### Challenges Faced
- Creating a synthetic dataset that is both original and *realistic*
  required simulating actual appliance power draw rather than pure random
  sampling.
- Balancing signal vs. noise so preprocessing and feature engineering
  steps remain meaningful.

### Future Improvements
- Integrate a live weather API for real-time outdoor temperature.
- Add hyperparameter optimization across all models (not just the winner).
- Deploy via Docker for reproducible production use.
- Collect real smart-meter data to validate the synthetic model.
""")

nb['cells'] = cells
with open("/home/claude/energy_project/notebooks/ML3_Energy_Project.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook created.")
