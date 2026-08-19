PYTHON := .venv/bin/python

.PHONY: install experiment-1 experiment-2 experiment-3 experiment-4 experiment-5 all demo

install:
	python3.12 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

experiment-1:
	$(PYTHON) -m experiments.experiment_1_data_preparation.run

experiment-2:
	$(PYTHON) -m experiments.experiment_2_text_comparison.run

experiment-3:
	$(PYTHON) -m experiments.experiment_3_model_comparison.run

experiment-4:
	$(PYTHON) -m experiments.experiment_4_model_review.run

experiment-5:
	$(PYTHON) -m experiments.experiment_5_final_evaluation.run

all: experiment-1 experiment-2 experiment-3 experiment-4 experiment-5

demo:
	$(PYTHON) -m uvicorn demo.app:app --host 127.0.0.1 --port 8000
