# Dataset Card - Primary Twitter Cyberbullying Dataset

Last reconciled: 2026-07-25

## Intended research use

This dataset supports research into classification of cyberbullying-associated
content and likely target category in individual tweets. It is not sufficient to
prove a complete cyberbullying event, user intent, repetition, power imbalance,
or real-world harm.

## Schema and labels

| Item | Verified value | Evidence |
| --- | --- | --- |
| Raw records | 47,692 | `data/interim/data_audit.json` |
| Raw columns | `tweet_text`, `cyberbullying_type` | audit JSON |
| Expected labels | `age`, `ethnicity`, `gender`, `religion`, `other_cyberbullying`, `not_cyberbullying` | `configs/preprocessing.yaml` |
| Missing text rows | 0 | audit JSON |
| Empty text rows | 0 | audit JSON |
| Invalid label rows | 0 | audit JSON |
| Experiment-ready records | 43,826 | `data/processed/experiment_ready.csv` row count |

## Source and access

The recorded source is Kaggle's `Cyberbullying Classification` dataset,
owner/slug `andrewmvd/cyberbullying-classification`:

`https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification`

The local raw file checksum is recorded in `docs/dataset_source.md`. The
retrieval date, dataset version, licence, and permission to redistribute raw
tweet text are still not recorded. Follow Kaggle and applicable platform terms;
do not redistribute raw text until those terms are verified.

## Data preparation and quality controls

- The raw layer remains under `data/raw/`; validated, processed, and split layers
  are stored separately and ignored by Git.
- The audit records UTF-8 parsing, schema, labels, missing/empty/non-string text,
  exact duplicates, and conflicting labels.
- Conflicting-label text groups are excluded from experiment-ready data. Later
  copies of the same normalised text and label are excluded; near duplicates are
  retained but assigned a group ID for grouped evaluation.
- Near-duplicate diagnostics use character five-gram MinHash/LSH at thresholds
  0.80, 0.85, and 0.90. The currently configured primary threshold is 0.85.
- Aggregate EDA outputs include class distributions, text/word length, URLs,
  user mentions, hashtags, emoji presence, uppercase ratio, punctuation, and
  repeated characters. They do not include raw tweet text.

## Known limitations

- Retrieval date, version, licence, and redistribution permission are not
  recorded. See `docs/dataset_source.md`.
- Labels may contain annotation ambiguity. The audit identified conflicting-label
  text groups; exclusion reduces one known inconsistency but does not guarantee
  label correctness.
- The dataset represents individual tweet text, not full conversational or social
  context.
- Identity-term and target-category labels must not be interpreted as demographic
  fairness ground truth.

## Ethical and access controls

- Treat raw tweets as potentially harmful or distressing.
- Do not publish unnecessary raw examples, usernames, URLs, or personal data.
- Do not attempt user identification, personal-data enrichment, or reconstruction
  of deleted content.
- Do not redistribute raw data unless the original source and licence explicitly
  permit it.
- The planned prototype supports human moderation and is not autonomous
  punishment.

## Required updates before final release

- Add verified source, licence, retrieval, version, and access information.
- Record final split and processing checksums after any legitimate data decision.
- Add model-specific results and limitations only after reproducible experiments.
