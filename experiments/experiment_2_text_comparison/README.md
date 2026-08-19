# Experiment 2 — text comparison

## Problem

Social-media spelling, mentions, URLs, hashtags, emoji, and obfuscation make representation choices consequential.

## Choices tried

Exact P0 minimal, P1 social-media-aware, and P2 aggressive preprocessing with Logistic Regression and Linear SVM; then word, character, and combined TF-IDF with the same two classifiers.

## Solution and result

P1 plus combined word/character TF-IDF was selected using grouped folds 1–4. `results.json` contains every fold, per-class metric, confusion matrix, feature count, and runtime—not only headline means.

## Limitations

Bounded English-oriented sparse features cannot capture all multilingual meaning or conversational context.

## Run

```bash
make experiment-2
```
