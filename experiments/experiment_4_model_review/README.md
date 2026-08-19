# Experiment 4 — model review

## Problem

Similar validation scores do not show behavior under obfuscation, domain shift, uncertainty, identity terms, or recurring errors.

## Choices tried

Both frozen finalists under five text conditions; Davidson external transfer; seven confidence thresholds; paired bootstrap; masked coefficients; neutral identity probes; aggregate confusion pairs.

## Solution and result

Retain balanced Logistic Regression as the explainable probabilistic primary, refer confidence below 0.45 to a human, and disclose the external false-positive and obfuscation weaknesses. `results.json` contains computed evidence for both finalists.

## Limitations

External labels only support a binary transfer check. Identity probes are not a fairness audit. Raw errors are not redistributed; optional manual qualitative coding was deliberately omitted.

## Run

```bash
make experiment-4
```
