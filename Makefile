PYTHON := .venv/bin/python
RAW_INPUT ?= data/raw/cyberbullying_tweets.csv
PROCESSED_INPUT ?= data/processed/experiment_ready.csv

.PHONY: install audit eda prepare splits verify-splits baseline test lint

install:
	python3.12 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -r requirements-dev.txt

audit:
	$(PYTHON) scripts/run_data_audit.py --input "$(RAW_INPUT)"

eda:
	$(PYTHON) scripts/run_eda.py --input "$(RAW_INPUT)"

prepare:
	$(PYTHON) scripts/prepare_dataset.py --input "$(RAW_INPUT)" --output "$(PROCESSED_INPUT)"

splits:
	$(PYTHON) scripts/build_splits.py --input "$(PROCESSED_INPUT)"

verify-splits:
	$(PYTHON) scripts/verify_frozen_splits.py --input "$(PROCESSED_INPUT)"

baseline:
	$(PYTHON) scripts/run_tfidf_baseline.py --overwrite

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .
