# ⚡ Smart Electricity Consumption Prediction & Energy Optimization System

**Task ID:** ML-3 | Teyzix Core Internship (June Batch)
**Domain:** Machine Learning (Advanced, Industry-Based)

A production-style ML system that predicts household daily electricity
consumption, forecasts monthly usage/bill, and generates personalized
energy-saving recommendations — built on an **original, self-generated
synthetic dataset** (no public datasets used, per task rules).

---

## 📁 Project Structure

```
energy_project/
├── data/
│   ├── generate_dataset.py            # Original synthetic dataset generator
│   ├── dataset_documentation.md       # Full data collection methodology
│   ├── energy_dataset.csv             # Raw generated dataset (658 rows)
│   ├── energy_dataset_cleaned.csv     # After preprocessing
│   └── energy_dataset_model_ready.csv # After feature engineering
├── src/
│   ├── preprocessing.py               # Missing values, duplicates, outliers, encoding, scaling
│   ├── eda.py                         # Statistical summary + all EDA plots
│   ├── feature_engineering.py         # Derived features + feature selection
│   ├── train_models.py                # Trains & compares 3 models, hyperparameter tuning
│   ├── explainability.py              # SHAP explainability (bonus)
│   ├── recommendations.py             # Bill/efficiency/recommendation engine
│   └── generate_reports.py            # Produces all deliverable reports
├── models/
│   ├── best_model.pkl                 # Trained best model (Linear Regression)
│   ├── scaler.pkl / encoders.pkl      # Preprocessing artifacts
│   └── feature_columns.txt
├── app/
│   └── app.py                         # Gradio prediction interface
├── notebooks/
│   └── ML3_Energy_Project.ipynb       # Full executed pipeline walkthrough
├── reports/                           # All generated reports (CSV/Excel) + plots
├── screenshots/
│   └── gradio_app_screenshot.png
├── PROJECT_REPORT.md
├── requirements.txt
└── README.md
```

## 🚀 How to Run

```bash
pip install -r requirements.txt

# 1. Generate the dataset
python3 data/generate_dataset.py

# 2. Run the full pipeline
python3 src/preprocessing.py
python3 src/eda.py
python3 src/feature_engineering.py
python3 src/train_models.py
python3 src/explainability.py
python3 src/generate_reports.py

# 3. Launch the prediction app
python3 app/app.py
# Open http://localhost:7860
```

Or simply open `notebooks/ML3_Energy_Project.ipynb` — it runs the entire
pipeline end-to-end (already executed with outputs saved).

## 🧠 Models Compared
| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Linear Regression (best)** | 1.05 | 1.29 | **0.957** |
| Random Forest Regressor | 1.15 | 1.60 | 0.935 |
| Decision Tree Regressor | 1.29 | 1.81 | 0.916 |

## ✨ Features
- Original synthetic dataset (650+ rows, 14 features) built from realistic appliance power-draw simulation
- Full preprocessing: missing values, duplicates, outlier capping, encoding, scaling
- Comprehensive EDA with correlation heatmaps, distributions, boxplots, scatter plots
- Feature engineering with correlation + Random-Forest-importance-based selection
- 3-model comparison with MAE/MSE/RMSE/R², hyperparameter tuning on the top model
- **Bonus:** SHAP explainability for model interpretation
- Gradio prediction interface: daily/monthly consumption, bill estimate, peak hours, efficiency score
- Personalized energy optimization recommendations
- Auto-generated reports (CSV + combined Excel workbook)

## 📊 Key Business Insight
AC usage hours is the dominant driver of electricity consumption (r ≈ 0.91),
followed by outdoor temperature (r ≈ 0.71) — confirming that cooling load
is the single biggest lever for household energy savings.
