# Experiment 3 — model comparison

## Problem

Model selection must compare distinct families, bounded variants, imbalance treatments, contextual features, and leakage risk without consulting protected fold 0.

## Choices tried

Naive Bayes; bounded Logistic Regression and Linear SVM variants; Random Forest; XGBoost; unweighted, balanced, and oversampled training; pinned MiniLM embeddings; grouped versus random stratified folds.

## Solution and result

Balanced Logistic Regression is primary; Linear SVM C=0.5 is the different-family backup. Selection uses macro-F1 on grouped folds 1–4. Detailed fold/class/runtime/confusion evidence and model-selection reasoning live in `results.json`.

## Limitations

Tree searches are deliberately bounded. MiniLM is frozen, English-oriented, and truncated. The random-split result is diagnostic, not the reported estimate.

## Run

```bash
make experiment-3
```

The pinned MiniLM revision downloads only when the ignored, hash-checked embedding cache is absent.
