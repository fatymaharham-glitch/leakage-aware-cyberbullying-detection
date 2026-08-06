# Multi-Class Cyberbullying Detection on Twitter: A Comparative Study of Machine Learning Algorithms

## Research overview

This repository compares machine-learning algorithms for classifying individual
Twitter posts into cyberbullying-related categories. It is organised around
dataset provenance, validation, exploratory analysis, duplicate control, and
leakage-aware evaluation.

### Expected classes

- `age`
- `ethnicity`
- `gender`
- `religion`
- `other_cyberbullying`
- `not_cyberbullying`

## Current scope

The repository includes responsible dataset audit, documentation, text preparation,
duplicate control, reproducible splits, preprocessing ablation, sparse representation
comparison, and bounded classifier benchmarking. Raw data and model artefacts are not
committed. Aggregate experiment evidence is under `reports/model_results/`.

## Important interpretation boundary

This system is intended to detect cyberbullying-associated content in individual tweets. A prediction does **not** prove that a complete cyberbullying event occurred, establish intent, identify a perpetrator, or determine real-world harm.

## Repository structure

```text
configs/    Reproducible paths and preprocessing assumptions
data/       Local dataset staging area; data files are intentionally ignored by Git
docs/       Dataset provenance, methodology, and IPR-progress notes
reports/    Generated figures and tables
scripts/    Safe command-line entry points for data and experiment workflows
src/        Reusable validation, preprocessing, deduplication, and splitting code
tests/      Minimal checks for critical data assumptions
```

## Environment setup

Use Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Verify the setup:

```bash
python -m compileall src scripts
pytest
ruff check .
```

## Dataset acquisition

No dataset is downloaded or committed by this project. First, identify and document an approved dataset source in [docs/dataset_source.md](docs/dataset_source.md). Then configure Kaggle credentials outside this repository (for example, in `~/.kaggle/kaggle.json`) and run:

```bash
bash scripts/download_dataset.sh --dataset OWNER/DATASET-SLUG
```

The script writes only to `data/raw/`; it requires an explicit Kaggle dataset identifier. Never add `kaggle.json`, raw tweets, derived data, or credentials to Git.

After locating the approved source file, run the data preparation phase with
project-relative paths:

```bash
python scripts/run_data_audit.py --input data/raw/cyberbullying_tweets.csv
python scripts/run_eda.py --input data/raw/cyberbullying_tweets.csv
python scripts/prepare_dataset.py \
  --input data/raw/cyberbullying_tweets.csv \
  --output data/processed/experiment_ready.csv
python scripts/build_splits.py --input data/processed/experiment_ready.csv
```

## Reproducibility rules

1. Keep source URL, retrieval date, checksum, licence, and dataset version in `docs/dataset_card.md`.
2. Keep raw data unchanged. Write derived datasets to `data/interim/`, `data/processed/`, and `data/splits/` only.
3. Use the configured random seed (`42`) and commit configuration changes with an explanation.
4. Audit invalid labels, empty text, duplicates, conflicting labels, and near-duplicates before splitting.
5. Exclude conflicting-label exact text groups; retain near duplicates only within one leakage-aware fold.
6. Treat leakage-aware fold 0 as frozen final test data: no preprocessing, tuning, or feature decisions may use it.
7. Save experiment configurations, aggregate metrics, and manifests under `reports/model_results/`; do not fabricate findings or metrics.

## Ethical warning

The dataset may contain offensive, hateful, abusive, or distressing language. Handle it only for the approved research purpose, minimise exposure, protect any personal data, respect the source licence and platform terms, and avoid publishing identifiable text unless ethics approval and licensing allow it.

## Useful commands

```bash
make install
make audit
make eda
make prepare
make splits
make verify-splits
make baseline
make representations
make classifiers
make test
make lint
```
