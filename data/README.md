# Data directory

Data files are deliberately excluded from version control. Keep credentials, raw tweets, source archives, and derived files local.

- `raw/`: immutable files obtained from the documented source.
- `interim/`: temporary audit or working files.
- `processed/`: validated and prepared dataset files.
- `splits/`: train, validation, and test partitions.

Before placing data here, document its source, licence, retrieval date, and checksum in `docs/dataset_card.md`. Do not place `kaggle.json` in this repository.
