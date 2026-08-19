# Experiment 1 — data preparation

## Problem

Invalid labels, conflicting duplicates, and similar tweets split across training and validation can inflate results.

## Choices tried

Validated missing/empty values and six labels; audited exact duplicates and label conflicts; inspected character-five-gram near-duplicate groups; verified five protected grouped folds.

## Solution and result

Raw text stays unchanged and local. Preparation excludes invalid/conflicting rows, retains one canonical exact duplicate, and assigns each near-duplicate group to one fold. `results.json` records fresh counts, complete fold membership, hashes, and zero group overlap.

## Limitations

Main Kaggle redistribution permission is unresolved. This compact run validates protected near-duplicate assignments rather than rebuilding the expensive MinHash grouping.

## Run

```bash
make experiment-1
```
