# Dataset Documentation — Smart Electricity Consumption Dataset

## 1. Collection Method: Realistic Synthetic Data Generation

Per task rules, no public dataset (Kaggle/UCI/GitHub) was used. Instead, an
**original synthetic dataset** was generated programmatically
(`generate_dataset.py`) using a **bottom-up physical simulation approach**
rather than pure random sampling, so relationships in the data mirror how
real households actually consume electricity.

### Why this approach is realistic
Each household's daily consumption is built from actual appliance power
draw (kW) × hours used, using standard real-world appliance wattage
approximations:

| Appliance | Approx. Power Rating |
|---|---|
| Air Conditioner | 1.5 kW |
| Ceiling Fan | 75 W |
| Refrigerator (24h equivalent) | 150 W |
| Washing Machine | 500 W |
| Water Motor/Pump | 750 W |
| Lighting (per room factor) | 60 W |

On top of the physical base load, the generator layers:
- **Seasonal effects**: Summer → high AC/fan hours & high outdoor temp;
  Winter → minimal AC, low fan hours; Spring/Autumn → moderate.
- **Household-size effects**: larger families draw slightly more
  miscellaneous load (`family_factor`).
- **Weekday/holiday effects**: weekends/holidays show ~8% higher usage
  (people home more).
- **Random measurement noise**: ±5% Gaussian noise to simulate real smart
  meter variance.

### Intentional data-quality issues (for the preprocessing stage)
To make preprocessing meaningful (not just "the CSV is already clean"),
the generator deliberately injects:
- ~2% missing values in 3 numeric columns (Outdoor Temperature, Washing
  Machine Hours, Lighting Hours)
- 8 duplicate rows
- 6 extreme outlier rows (consumption × 3–5) simulating meter glitches
  or unusually heavy-usage days

## 2. Final Dataset Specs
- **Rows**: 658 (after injected duplicates; net >500 required even after
  cleaning)
- **Features**: 14 input features + 1 target = 15 columns

## 3. Feature Description

| Feature | Type | Description |
|---|---|---|
| House_Type | Categorical | Apartment / Independent House / Villa |
| Family_Members | Numeric | Number of people in household (1–7) |
| Number_of_Rooms | Numeric | Total rooms in the house |
| Daily_Appliance_Usage_Count | Numeric | Count of distinct appliances used that day |
| AC_Usage_Hours | Numeric | Hours air conditioner ran |
| Fan_Usage_Hours | Numeric | Hours fans ran |
| Refrigerator_Usage_Hours | Numeric | Hours fridge ran (always 24) |
| Washing_Machine_Usage_Hours | Numeric | Hours washing machine ran |
| Water_Motor_Usage_Hours | Numeric | Hours water pump/motor ran |
| Lighting_Hours | Numeric | Total lighting hours (scaled by rooms) |
| Outdoor_Temperature_C | Numeric | Ambient outdoor temperature |
| Day_of_Week | Categorical | Monday–Sunday |
| Season | Categorical | Summer / Winter / Spring-Autumn |
| Is_Holiday | Binary | 1 = weekend/holiday, 0 = working day |
| **Daily_Electricity_Consumption_kWh** | Numeric (Target) | Total daily electricity consumption |

## 4. Suitability for ML
The dataset is balanced across house types, seasons, and family sizes, has
a continuous physically-grounded target variable suitable for regression,
contains realistic data-quality noise (missing values/duplicates/outliers)
for genuine preprocessing work, and encodes non-trivial feature
interactions (e.g., AC hours × season × temperature) for feature
engineering and model comparison to be meaningful.
