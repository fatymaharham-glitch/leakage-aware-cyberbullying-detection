# Benchmark selection record

Configuration: `sparse_representation.yaml`. Evaluation used grouped folds [1, 2, 3, 4]; frozen fold 0 was excluded.

## Ranking

| Rank | Representation | Configuration | Mean macro-F1 | SD |
|---:|---|---|---:|---:|
| 1 | combined | logistic_regression | 0.8690 | 0.0033 |
| 2 | character | logistic_regression | 0.8671 | 0.0013 |
| 3 | combined | linear_svm | 0.8659 | 0.0036 |

Highest mean macro-F1: **combined + logistic_regression** (0.8690 ± 0.0033). Weakest class by mean F1 for that candidate: **not_cyberbullying** (0.6792).

## Week 6 representation decision

Use **combined** TF-IDF for classifier benchmarking. No auxiliary hand-crafted features were added: incremental value is unproven, and heuristic identity/social features would widen the sensitivity surface.

## Limits

This is grouped-validation screening, not a frozen final-test result. Overlapping fold variability means small score differences are not yet statistically established; later statistical, robustness, calibration, and leakage-gap phases must inform final selection.
