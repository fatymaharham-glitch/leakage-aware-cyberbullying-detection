# Benchmark selection record

Configuration: `classifier_benchmark.yaml`. Evaluation used grouped folds [1, 2, 3, 4]; frozen fold 0 was excluded.

## Ranking

| Rank | Representation | Configuration | Mean macro-F1 | SD |
|---:|---|---|---:|---:|
| 1 | combined | logistic_regression_c1_balanced | 0.8703 | 0.0029 |
| 2 | combined | logistic_regression_c2 | 0.8696 | 0.0034 |
| 3 | combined | logistic_regression_c1 | 0.8690 | 0.0033 |

Highest mean macro-F1: **combined + logistic_regression_c1_balanced** (0.8703 ± 0.0029). Weakest class by mean F1 for that candidate: **not_cyberbullying** (0.6832).

## Week 6 shortlist

Retain **logistic_regression_c1_balanced** as primary and **linear_svm_c0_5** as the faster linear comparator. Keep XGBoost only as a non-linear benchmark; its grouped macro-F1 is lower and its runtime is materially higher.

Balanced class weighting changed Logistic Regression mean macro-F1 by +0.0012. Treat this as an ablation result, not proof of a population-level improvement.

## Limits

This is grouped-validation screening, not a frozen final-test result. Overlapping fold variability means small score differences are not yet statistically established; later statistical, robustness, calibration, and leakage-gap phases must inform final selection.
