# ── Smart Electricity Prediction — Developer Makefile ────────────────────────
# Usage: make <target>
# Requires GNU Make (available via Git Bash / WSL on Windows)

PYTHON  = python
SRC_DIR = src
APP     = app/app.py

.PHONY: help install install-dev pipeline app test lint clean

help:          ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:       ## Install production dependencies
	pip install -r requirements.txt

install-dev:   ## Install all dependencies (prod + dev)
	pip install -r requirements-dev.txt

pipeline:      ## Run the full ML pipeline end-to-end
	$(PYTHON) data/generate_dataset.py
	$(PYTHON) $(SRC_DIR)/preprocessing.py
	$(PYTHON) $(SRC_DIR)/eda.py
	$(PYTHON) $(SRC_DIR)/feature_engineering.py
	$(PYTHON) $(SRC_DIR)/train_models.py
	$(PYTHON) $(SRC_DIR)/explainability.py
	$(PYTHON) $(SRC_DIR)/generate_reports.py

app:           ## Launch the Gradio prediction app (http://localhost:7860)
	$(PYTHON) $(APP)

test:          ## Run all pytest unit tests
	pytest tests/ -v --tb=short

lint:          ## Run flake8 linter on all source files
	flake8 $(SRC_DIR) app config.py --max-line-length=100

clean:         ## Remove Python cache files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
