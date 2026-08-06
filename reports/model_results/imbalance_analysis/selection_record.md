# Benchmark selection record

Configuration: `imbalance_analysis.yaml`. Evaluation used grouped folds [1, 2, 3, 4]; frozen fold 0 was excluded.

## Ranking

| Rank | Representation | Configuration | Mean macro-F1 | SD |
|---:|---|---|---:|---:|
| 1 | combined | logistic_regression_balanced | 0.8703 | 0.0029 |
| 2 | combined | logistic_regression_unweighted | 0.8690 | 0.0033 |
| 3 | combined | logistic_regression_random_oversampling | 0.8690 | 0.0035 |

Highest mean macro-F1: **combined + logistic_regression_balanced** (0.8703 ± 0.0029). Weakest class by mean F1 for that candidate: **not_cyberbullying** (0.6832).

## Class-imbalance decision

Retain **logistic_regression_balanced** from the bounded weighting and random-oversampling comparison. Random oversampling was applied only after each training-fold split. Confidence-threshold selection is deferred to P11 calibration to avoid tuning and judging thresholds on the same validation observations.
For the retained treatment, validation log loss was 0.3481 and multiclass Brier score was 0.1736 (lower is better). Lowest log loss was 0.3464 (logistic_regression_random_oversampling); lowest Brier was 0.1734 (logistic_regression_random_oversampling). These small mixed differences do not justify changing the macro-F1 decision. Full calibration and confidence-threshold selection remain P11 work.

## Limits

This is grouped-validation screening, not a frozen final-test result. Overlapping fold variability means small score differences are not yet statistically established; later statistical, robustness, calibration, and leakage-gap phases must inform final selection.
