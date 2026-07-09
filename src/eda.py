"""
eda.py
------
Exploratory Data Analysis: statistical summary, distributions, correlation
heatmap, boxplots, scatter plots, and feature relationships.
Saves all plots as PNG into /home/claude/energy_project/reports/eda_plots/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH = "/home/claude/energy_project/data/energy_dataset_cleaned.csv"
PLOT_DIR = "/home/claude/energy_project/reports/eda_plots"
TARGET = "Daily_Electricity_Consumption_kWh"

os.makedirs(PLOT_DIR, exist_ok=True)
sns.set_style("whitegrid")

df = pd.read_csv(DATA_PATH)

numeric_cols = [
    "Family_Members", "Number_of_Rooms", "Daily_Appliance_Usage_Count",
    "AC_Usage_Hours", "Fan_Usage_Hours", "Refrigerator_Usage_Hours",
    "Washing_Machine_Usage_Hours", "Water_Motor_Usage_Hours",
    "Lighting_Hours", "Outdoor_Temperature_C", TARGET
]

# 1. Statistical summary
summary = df[numeric_cols].describe().T
summary.to_csv(os.path.join(PLOT_DIR, "..", "dataset_statistical_summary.csv"))
print(summary)

# 2. Distribution histograms
fig, axes = plt.subplots(3, 4, figsize=(20, 12))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    sns.histplot(df[col], kde=True, ax=axes[i], color="#2e7d32")
    axes[i].set_title(col)
for j in range(len(numeric_cols), len(axes)):
    axes[j].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "distributions.png"), dpi=110)
plt.close()

# 3. Correlation heatmap
plt.figure(figsize=(11, 9))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="Greens", square=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "correlation_heatmap.png"), dpi=110)
plt.close()

# 4. Boxplots (outlier check per feature)
fig, axes = plt.subplots(3, 4, figsize=(20, 12))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    sns.boxplot(y=df[col], ax=axes[i], color="#66bb6a")
    axes[i].set_title(col)
for j in range(len(numeric_cols), len(axes)):
    axes[j].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "boxplots.png"), dpi=110)
plt.close()

# 5. Scatter plots vs target
key_features = ["AC_Usage_Hours", "Fan_Usage_Hours", "Outdoor_Temperature_C",
                 "Number_of_Rooms", "Family_Members", "Lighting_Hours"]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for i, col in enumerate(key_features):
    sns.scatterplot(x=df[col], y=df[TARGET], ax=axes[i], alpha=0.5, color="#1b5e20")
    axes[i].set_title(f"{col} vs {TARGET}")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "scatter_relationships.png"), dpi=110)
plt.close()

# 6. Consumption by Season / House Type (business insight plots)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(x="Season", y=TARGET, data=df, ax=axes[0], palette="Greens")
axes[0].set_title("Consumption by Season")
sns.boxplot(x="House_Type", y=TARGET, data=df, ax=axes[1], palette="Greens")
axes[1].set_title("Consumption by House Type")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "consumption_by_category.png"), dpi=110)
plt.close()

print("EDA complete. Plots saved to:", PLOT_DIR)

# Business insights (printed + saved)
insights = []
top_corr = corr[TARGET].drop(TARGET).sort_values(ascending=False)
insights.append(f"Top positive correlation with consumption: {top_corr.index[0]} (r={top_corr.iloc[0]:.2f})")
insights.append(f"Second strongest driver: {top_corr.index[1]} (r={top_corr.iloc[1]:.2f})")
season_avg = df.groupby("Season")[TARGET].mean().sort_values(ascending=False)
insights.append(f"Highest average consumption season: {season_avg.index[0]} ({season_avg.iloc[0]:.2f} kWh/day)")
house_avg = df.groupby("House_Type")[TARGET].mean().sort_values(ascending=False)
insights.append(f"Highest average consumption house type: {house_avg.index[0]} ({house_avg.iloc[0]:.2f} kWh/day)")
holiday_avg = df.groupby("Is_Holiday")[TARGET].mean()
insights.append(f"Holiday/weekend avg consumption ({holiday_avg.get(1,0):.2f} kWh) vs working day ({holiday_avg.get(0,0):.2f} kWh)")

with open(os.path.join(PLOT_DIR, "..", "business_insights.txt"), "w") as f:
    f.write("\n".join(insights))

for i in insights:
    print("-", i)
