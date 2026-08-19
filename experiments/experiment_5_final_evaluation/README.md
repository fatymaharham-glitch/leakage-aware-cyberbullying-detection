# Experiment 5 — final evaluation

## Problem

The final score is credible only if model/feature selection was frozen before the protected fold was accessed.

## Choices tried

No new model choice occurs here. The preselected balanced Logistic Regression with P1 combined TF-IDF trains on folds 1–4 and evaluates exact fold 0 membership once; later runs are marked reproductions.

## Solution and result

`results.json` contains headline and per-class metrics, class order, confusion counts, confidence distribution, checksums, and append-only access history. The same fitted pipeline is exported for the demo.

## Limitations

Repeated runs do not create new independent test evidence. A single tweet still lacks the context required to establish a complete cyberbullying event.

## Run

```bash
make experiment-5
```
