# P7 leakage credibility-gap decision

The same combined TF-IDF + balanced Logistic Regression candidate was evaluated with grouped and random-stratified folds [1, 2, 3, 4]. Frozen fold 0 remained unchanged and was never evaluated.

Random minus grouped macro-F1: **+0.0012**. Grouped folds had **0** near-duplicate groups crossing validation boundaries; random folds had **165**. The grouped score remains the selection score because it better represents performance on genuinely different text.
