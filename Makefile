PYTHON := .venv/bin/python
RAW_INPUT ?= data/raw/cyberbullying_tweets.csv
PROCESSED_INPUT ?= data/processed/experiment_ready.csv

.PHONY: install audit eda prepare splits verify-splits baseline representations classifiers contextual imbalance leakage-gap candidates test lint

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

representations:
	$(PYTHON) scripts/run_sparse_benchmark.py --config sparse_representation.yaml --overwrite

classifiers:
	$(PYTHON) scripts/run_sparse_benchmark.py --config classifier_benchmark.yaml --overwrite

contextual:
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(PYTHON) scripts/run_contextual_baseline.py --overwrite

imbalance:
	$(PYTHON) scripts/run_sparse_benchmark.py --config imbalance_analysis.yaml --overwrite

leakage-gap:
	$(PYTHON) scripts/run_leakage_gap.py --overwrite

candidates:
	$(PYTHON) scripts/run_candidate_selection.py --overwrite

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .
