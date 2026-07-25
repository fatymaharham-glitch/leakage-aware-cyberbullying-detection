# Frozen Split Validation

Status: PASS

This aggregate report contains no tweet text, usernames, URLs, or row IDs.

## Verified checks

- Experiment-ready rows: 43826
- Random split rows: 43826
- Grouped-fold rows: 43826
- Frozen fold-0 test rows: 8766
- Input SHA-256 matches manifest: True
- Random split covers every experiment-ready row once.
- Grouped folds cover every experiment-ready row once.
- No near-duplicate group crosses grouped fold boundaries.
- Saved manifest exactly matches current input, seed, folds, and fold-0 membership.

## Operational rule

Treat grouped fold 0 as frozen final-test membership. Do not use it for preprocessing, feature, threshold, or hyperparameter decisions.
