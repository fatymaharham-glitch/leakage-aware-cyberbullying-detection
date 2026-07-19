# Data quality report

This report contains aggregate metadata only. It intentionally excludes raw tweet text.

## Inventory

- File path: `data/raw/cyberbullying_tweets.csv`
- SHA-256: `2e88deb41537cab09e9490f659c6c5d9e0ce9ebe9396f1329b7c5685cb3796b3`
- Shape: 47692 rows × 2 columns
- Columns: `tweet_text`, `cyberbullying_type`
- UTF-8 parsing errors: none observed

## Data types

| Column | Data type |
| --- | --- |
| `tweet_text` | `str` |
| `cyberbullying_type` | `str` |

## Missing values

| Column | Missing rows |
| --- | --- |
| `tweet_text` | 0 |
| `cyberbullying_type` | 0 |

- Missing text rows: 0
- Empty or whitespace-only text rows: 0
- Non-string text rows: 0
- Invalid label rows: 0
- Invalid or unexpected labels: None observed
- Expected labels not observed: None observed

## Exact duplicate and conflict audit

| Check | Count |
| --- | --- |
| full duplicate rows involved | 72 |
| full duplicate rows beyond first | 36 |
| same text same label groups | 210 |
| same text same label rows beyond canonical | 588 |
| conflicting label groups | 1635 |
| conflicting label rows | 3525 |

Conflicting-label text groups are excluded from `data/processed/experiment_ready.csv`.
Near duplicates are retained and controlled only through group-aware folds.
