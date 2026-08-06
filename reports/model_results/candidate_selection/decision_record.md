# P8 candidate freeze

Selected using grouped folds 1-4 only. Frozen fold 0 was not evaluated.

1. **Balanced Logistic Regression + combined TF-IDF** — primary; grouped macro-F1 0.8703, best overall and strongest bounded imbalance treatment.
2. **Linear SVM (C=0.5) + combined TF-IDF** — secondary; grouped macro-F1 0.8690, competitive different-family comparator with lower fit time.

Random validation changed the primary score by +0.0012 versus grouped validation; grouped evidence controls selection. The contextual baseline was not retained because its macro-F1 was materially lower and its embedding dimensions are less directly interpretable. These are P8 candidates, not final-test winners; P9-P15 can still reject either before P16.
