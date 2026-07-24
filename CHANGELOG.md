# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added
- Prediction history CSV logging in `app/logs/predictions.csv`
- Centralized `config.py` for all paths and constants
- `src/utils.py` shared utility module
- GitHub Actions CI workflow (`.github/workflows/ci.yml`)
- MIT License
- Contributing guidelines
- Development requirements (`requirements-dev.txt`)
- Pytest unit tests for recommendations engine
- `.gitattributes` for cross-platform line endings

### Changed
- `src/recommendations.py` — now imports from `config.py`, removed hardcoded paths
- `src/preprocessing.py` — replaced hardcoded Linux paths with config; added structured logging

### Fixed
- Hardcoded `/home/claude/...` path in `src/recommendations.py` (broke on Windows)
- Hardcoded paths in `src/preprocessing.py`

---

## [1.0.0] — 2025-06-30

### Added
- Initial project submission for Teyzix ML Internship (Task ML-3)
- Original synthetic dataset (658 rows, 14 features)
- Full preprocessing pipeline: missing values, duplicates, outlier capping, encoding, scaling
- EDA with correlation heatmaps, distributions, boxplots, scatter plots
- Feature engineering with correlation + RF importance-based selection
- Three-model comparison (Linear Regression, Random Forest, Decision Tree)
- Hyperparameter tuning with GridSearchCV
- SHAP explainability (bonus)
- Gradio prediction interface with bill estimate and recommendations
- Auto-generated reports (CSV + combined Excel workbook)
- Jupyter notebook walkthrough (`notebooks/ML3_Energy_Project.ipynb`)
