"""Reproduce data-quality and protected-fold validation evidence."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.shared import LABELS, RAW_DATA, metadata, normalize, preprocess_p1, strict_project_data, write_json

import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parent


def duplicate_audit(raw: pd.DataFrame) -> dict[str, int]:
    normalized = raw["tweet_text"].map(preprocess_p1)
    label = raw["cyberbullying_type"].fillna("<MISSING_LABEL>").astype(str)
    text_label = pd.Series(list(zip(normalized, label, strict=True)), index=raw.index)
    label_counts = pd.DataFrame({"text": normalized, "label": label}).groupby("text")["label"].nunique()
    conflicting = normalized.isin(label_counts[label_counts > 1].index)
    group_sizes = text_label.value_counts()
    duplicate_groups = group_sizes[group_sizes > 1]
    return {
        "full_duplicate_rows_involved": int(raw.duplicated(keep=False).sum()),
        "full_duplicate_rows_beyond_first": int(raw.duplicated(keep="first").sum()),
        "same_text_same_label_groups": int(len(duplicate_groups)),
        "same_text_same_label_rows_involved": int(duplicate_groups.sum()),
        "same_text_same_label_rows_beyond_canonical": int((duplicate_groups - 1).sum()),
        "conflicting_label_groups": int((label_counts > 1).sum()),
        "conflicting_label_rows": int(conflicting.sum()),
    }


def main() -> None:
    frame = strict_project_data()
    if not RAW_DATA.is_file():
        raise FileNotFoundError("Download the main Kaggle CSV to data/cyberbullying_tweets.csv before Experiment 1.")
    raw = pd.read_csv(RAW_DATA)
    missing = {column: int(raw[column].isna().sum()) for column in raw.columns}
    invalid_labels = sorted(set(raw["cyberbullying_type"].dropna()) - set(LABELS))
    empty_text = int(raw["tweet_text"].map(normalize).eq("").sum())
    duplicates = duplicate_audit(raw)
    class_counts = frame["cyberbullying_type"].value_counts().reindex(LABELS)
    fold_counts = frame.groupby(["fold", "cyberbullying_type"]).size().unstack(fill_value=0).reindex(columns=LABELS)
    group_sizes = frame.groupby("near_duplicate_group_id").size()
    group_overlap = frame.groupby("near_duplicate_group_id")["fold"].nunique()
    results = {
        "experiment": 1,
        "title": "Data preparation and leakage safeguards",
        "question": "Is the dataset valid, and can similar tweets be kept together?",
        "choices_tried": [
            "missing, empty-text, and six-label validation",
            "exact duplicate and conflicting-label removal",
            "character-five-gram near-duplicate groups at Jaccard 0.85",
            "five stratified grouped folds with fold 0 protected",
        ],
        "solution": "Preserve raw text, exclude invalid/conflicting duplicate records, keep one canonical exact duplicate, and keep every near-duplicate group in one fold.",
        "results": {
            "raw": {"rows": int(len(raw)), "missing_values": missing, "empty_text_rows": empty_text, "invalid_labels": invalid_labels},
            "duplicate_audit": duplicates,
            "prepared": {
                "rows": int(len(frame)),
                "removed_from_raw": int(len(raw) - len(frame)),
                "classes": {label: int(class_counts[label]) for label in LABELS},
                "near_duplicate_groups": int(group_sizes.size),
                "multi_row_near_duplicate_groups": int((group_sizes > 1).sum()),
                "rows_in_multi_row_groups": int(group_sizes[group_sizes > 1].sum()),
            },
            "folds": {
                "row_counts": {str(fold): int(count) for fold, count in frame["fold"].value_counts().sort_index().items()},
                "class_counts": {str(fold): {label: int(value) for label, value in row.items()} for fold, row in fold_counts.iterrows()},
                "near_duplicate_groups_crossing_folds": int((group_overlap > 1).sum()),
                "protected_final_fold": 0,
            },
        },
        "limitations": [
            "Main Kaggle dataset redistribution licence/version remains unconfirmed, so raw and prepared text stay local and ignored.",
            "This run validates the existing deterministic near-duplicate assignments; rebuilding MinHash groups is intentionally outside the compact runner.",
        ],
        "metadata": metadata(),
    }
    write_json(OUTPUT / "results.json", results)

    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    class_counts.plot.bar(ax=axes[0, 0], color="#2368a2", title="Prepared class counts")
    pd.Series(duplicates).plot.bar(ax=axes[0, 1], color="#a23b42", title="Duplicate and conflict audit")
    frame["fold"].value_counts().sort_index().plot.bar(ax=axes[1, 0], color="#18745a", title="Protected fold sizes")
    group_sizes.value_counts().sort_index().head(8).plot.bar(ax=axes[1, 1], color="#9a5b0b", logy=True, title="Near-duplicate group sizes")
    for axis in axes.flat:
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Experiment 1 — data preparation evidence", fontsize=16)
    figure.tight_layout()
    figure.savefig(OUTPUT / "plot.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
