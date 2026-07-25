# Dataset Source Record

Last reconciled: 2026-07-25

## Purpose

This record preserves verified provenance information for the local dataset. It
is not evidence of permission, a licence, or approval for a particular use.

## Verified local record

| Field | Recorded value | Evidence |
| --- | --- | --- |
| Local filename | `data/raw/cyberbullying_tweets.csv` | Local repository workspace |
| File role | Primary raw tweet dataset | `data/README.md`; `configs/paths.yaml` |
| Provider | Kaggle | Dataset source recorded 2026-07-25 |
| Dataset title | `Cyberbullying Classification` | Kaggle dataset page |
| Kaggle owner / slug | `andrewmvd/cyberbullying-classification` | Dataset source recorded 2026-07-25 |
| Canonical source URL | `https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification` | Dataset source recorded 2026-07-25 |
| SHA-256 | `2e88deb41537cab09e9490f659c6c5d9e0ce9ebe9396f1329b7c5685cb3796b3` | `data/interim/data_audit.json` |
| Expected columns | `tweet_text`, `cyberbullying_type` | `configs/preprocessing.yaml`; audit JSON |
| Local handling | Raw data are ignored by Git and derived outputs are stored separately. | `.gitignore`; `README.md`; `data/README.md` |

## Information not recorded

The following require original acquisition evidence. Do not guess or infer
values from the local filename.

| Required field | Status | Required action |
| --- | --- | --- |
| Retrieval date | Not recorded | Record the date from acquisition evidence. |
| Dataset version or release | Not recorded | Record the provider's version/release if available. |
| Licence and platform terms | Not recorded | Record the applicable licence and Twitter/platform restrictions. |
| Permission to redistribute raw text | Not recorded | Assume no redistribution until permission is clear. |
| Ethics approval/handling requirement | Not recorded | Add the verified institutional requirement. |

## Provenance and release limitation

The provider, dataset URL, and Kaggle dataset slug are now recorded. Licence,
release/version, retrieval date, and redistribution conditions remain open
project risks. Until that information is verified, do not redistribute
raw tweet text, source archives, usernames, URLs, or derived text examples.
Share only code, hashes, split indices where permitted, and aggregate statistics.
