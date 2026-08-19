# Leakage-aware cyberbullying detection

Complete research/demo prototype for suggesting one of six tweet-level categories: `age`, `ethnicity`, `gender`, `religion`, `other_cyberbullying`, or `not_cyberbullying`.

Important boundary: a prediction does not prove intent, repetition, power imbalance, harm, or a complete cyberbullying event. It must not trigger automatic punishment.

## Outcome

Combined word/character TF-IDF with balanced Logistic Regression was selected on grouped development folds. Linear SVM remains the different-family backup. Protected fold 0 was accessed only after selection; later executions are labelled reproductions in Experiment 5.

The repository intentionally has five experiment records. Each folder contains exactly:

- `README.md` — problem, choices, solution/result, limitations, command.
- `run.py` — genuine calculation against protected inputs.
- `results.json` — headline and detailed fold/class/runtime evidence with provenance.
- `plot.png` — one multi-panel presentation figure.

## Experiments

| Experiment | Coverage |
|---|---|
| 1. Data preparation | Schema/label validation, exact duplicates, conflicts, near-duplicate groups, EDA summaries, hashes, protected folds |
| 2. Text comparison | Exact P0/P1/P2 settings; word, character, and combined TF-IDF; fold/class/runtime evidence |
| 3. Model comparison | Five families, bounded linear variants, imbalance treatments, pinned MiniLM, grouped/random leakage check, candidate selection |
| 4. Model review | Both finalists under robustness changes, Davidson transfer, confidence/referral, paired bootstrap, masked explanations, probes, aggregate errors |
| 5. Final evaluation | Frozen Logistic Regression on protected fold 0, per-class metrics, confusion matrix, access history, model export |

## Setup and commands

Python 3.12 is required.

```bash
make install
make experiment-1
make experiment-2
make experiment-3
make experiment-4
make experiment-5
make all
make demo
```

Open `http://127.0.0.1:8000` after `make demo`. The API preserves `GET /`, `GET /health`, `GET /project`, `GET /plots/{experiment_id}`, and `POST /predict`. Prediction requests accept 1–500 characters and one of five registry model IDs.

## Main dataset: manual local download

The main source is Kaggle's **Cyberbullying Classification**, owner/slug `andrewmvd/cyberbullying-classification`:

<https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification>

Download manually and place the unchanged CSV at:

```text
data/cyberbullying_tweets.csv
```

Expected SHA-256: `2e88deb41537cab09e9490f659c6c5d9e0ce9ebe9396f1329b7c5685cb3796b3`.

The prepared local input is `data/experiment_ready.csv`, expected SHA-256 `f203129d7c5f761471ab084cd6acac1b16f2039182e1d654a25290b7681af907`. Both remain ignored because retrieval date, release/version, licence, platform restrictions, and permission to redistribute tweet text are unresolved. Do not commit or redistribute them until verified.

## Tracked data

- `data/leakage_aware_folds.csv` — exact 43,826-row protected grouped assignment; fold 0 has 8,766 rows; no near-duplicate group crosses folds.
- `data/external_validation.csv` — 24,783-row Davidson et al. hate/offensive-language Twitter dataset.
- `data/external_validation_LICENSE.txt` — MIT licence copied from the Davidson source repository.

Davidson labels 0/1 map to harmful and 2 to not harmful. Project predictions map `not_cyberbullying` to not harmful and the remaining five classes to harmful. This is only a binary domain-transfer check because annotation tasks differ.

## Reproducibility and evidence rules

- Every learned transform fits training rows only.
- Fold 0 is absent from preprocessing, model selection, thresholds, and tuning.
- Input hashes, exact row membership, class order, folds `{0,1,2,3,4}`, and zero group overlap are fail-closed checks.
- MiniLM is pinned to revision `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`; its ignored cache records row order, input, dimensionality, and normalization.
- Result JSON declares fresh reproduction or verified migration. No raw tweets or row-level predictions are committed.
- Experiment 5 compares reproductions with prior verified classical metrics at `1e-6` tolerance.

## Repository map

```text
data/           tracked external data/licence and protected folds; ignored primary text
demo/           FastAPI app, dashboard, five model pipelines, checksum registry
experiments/    shared logic plus five four-file experiment folders
Makefile        install, five experiments, all, demo
requirements.txt pinned direct dependencies
PRESENTATION_GUIDE.md beginner speaking guide and honest limitations
```

Earlier duplicate research structures were consolidated into these experiment records. There is intentionally no retained pytest suite.

## Known limitations

- Main dataset licence/version/redistribution permission is unknown.
- Raw labels may contain annotation ambiguity even after conflict controls.
- Individual tweets omit conversational and social context.
- Obfuscated spelling, especially leetspeak and masking, reduces performance.
- External false positives show domain shift.
- Identity probes are diagnostics, not a complete fairness audit.
- Human qualitative error coding remains optional and was not performed.

See `PRESENTATION_GUIDE.md` for the beginner presentation narrative.
