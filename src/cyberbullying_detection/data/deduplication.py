"""Exact-duplicate and label-conflict controls for experiment-ready data."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cyberbullying_detection.preprocessing.text import duplicate_matching_normalize
from cyberbullying_detection.utils.hashing import text_hash

NORMALIZED_TEXT_HASH_COLUMN = "normalized_text_hash"


class ConflictingLabelError(ValueError):
    """Retained for callers that require conflicts to halt their own workflow."""


@dataclass(frozen=True)
class DuplicateAudit:
    """Aggregate exact-duplicate findings that do not expose tweet content."""

    full_duplicate_rows_involved: int
    full_duplicate_rows_beyond_first: int
    same_text_same_label_groups: int
    same_text_same_label_rows_involved: int
    same_text_same_label_rows_beyond_canonical: int
    conflicting_label_groups: int
    conflicting_label_rows: int


def _label_key(value: object) -> str:
    """Create a deterministic label key without treating missing values as equal text."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "<MISSING_LABEL>"
    return str(value)


def create_row_id(tweet_text: object, original_label: object) -> str:
    """Create the stable row identity from duplicate-normalised text and raw label."""
    payload = f"{duplicate_matching_normalize(tweet_text)}\x1f{_label_key(original_label)}"
    return text_hash(payload)


def annotate_exact_duplicates(
    frame: pd.DataFrame, *, text_column: str, label_column: str
) -> tuple[pd.DataFrame, DuplicateAudit]:
    """Add deterministic duplicate/conflict flags without dropping any rows.

    Full duplicates are evaluated on the input frame before audit columns are added.
    Text-label duplicates use duplicate-matching normalisation, while conflicting labels
    mark every row sharing that normalised text for later exclusion.
    """
    result = frame.copy()
    result[NORMALIZED_TEXT_HASH_COLUMN] = result[text_column].map(
        lambda value: text_hash(duplicate_matching_normalize(value))
    )
    label_keys = result[label_column].map(_label_key)
    text_label_keys = pd.Series(
        list(zip(result[NORMALIZED_TEXT_HASH_COLUMN], label_keys, strict=True)),
        index=result.index,
    )

    result["is_full_duplicate_row"] = frame.duplicated(keep=False)
    result["is_full_duplicate_row_beyond_first"] = frame.duplicated(keep="first")
    result["is_same_text_same_label_duplicate"] = text_label_keys.duplicated(keep=False)
    result["is_same_text_same_label_beyond_canonical"] = text_label_keys.duplicated(
        keep="first"
    )

    label_counts = result.groupby(NORMALIZED_TEXT_HASH_COLUMN, sort=False)[
        label_column
    ].agg(lambda values: len({_label_key(value) for value in values}))
    conflicting_hashes = set(label_counts[label_counts > 1].index)
    result["is_conflicting_label_text"] = result[NORMALIZED_TEXT_HASH_COLUMN].isin(
        conflicting_hashes
    )

    group_sizes = text_label_keys.value_counts(sort=False)
    duplicate_group_sizes = group_sizes[group_sizes > 1]
    audit = DuplicateAudit(
        full_duplicate_rows_involved=int(result["is_full_duplicate_row"].sum()),
        full_duplicate_rows_beyond_first=int(
            result["is_full_duplicate_row_beyond_first"].sum()
        ),
        same_text_same_label_groups=int(len(duplicate_group_sizes)),
        same_text_same_label_rows_involved=int(duplicate_group_sizes.sum()),
        same_text_same_label_rows_beyond_canonical=int(
            (duplicate_group_sizes - 1).sum()
        ),
        conflicting_label_groups=len(conflicting_hashes),
        conflicting_label_rows=int(result["is_conflicting_label_text"].sum()),
    )
    return result, audit


def exact_duplicate_summary_table(audit: DuplicateAudit) -> pd.DataFrame:
    """Return a sorted aggregate table suitable for a non-sensitive report."""
    records = [
        ("conflicting_label_groups", audit.conflicting_label_groups),
        ("conflicting_label_rows", audit.conflicting_label_rows),
        ("full_duplicate_rows_beyond_first", audit.full_duplicate_rows_beyond_first),
        ("full_duplicate_rows_involved", audit.full_duplicate_rows_involved),
        ("same_text_same_label_groups", audit.same_text_same_label_groups),
        (
            "same_text_same_label_rows_beyond_canonical",
            audit.same_text_same_label_rows_beyond_canonical,
        ),
        (
            "same_text_same_label_rows_involved",
            audit.same_text_same_label_rows_involved,
        ),
    ]
    return pd.DataFrame(records, columns=["metric", "count"]).sort_values(
        "metric", ignore_index=True
    )


def conflicting_label_summary_table(
    frame: pd.DataFrame, *, label_column: str
) -> pd.DataFrame:
    """Summarise label combinations in conflicts without including any tweet text."""
    conflicting = frame.loc[frame["is_conflicting_label_text"]].copy()
    if conflicting.empty:
        return pd.DataFrame(
            columns=["label_combination", "conflicting_text_groups", "rows_excluded"]
        )

    grouped = conflicting.groupby(NORMALIZED_TEXT_HASH_COLUMN, sort=True)[label_column].agg(
        lambda labels: " | ".join(sorted({_label_key(label) for label in labels}))
    )
    group_table = grouped.rename("label_combination").reset_index()
    row_counts = conflicting.groupby(NORMALIZED_TEXT_HASH_COLUMN, sort=True).size()
    group_table["rows_excluded"] = group_table[NORMALIZED_TEXT_HASH_COLUMN].map(
        row_counts
    )
    summary = (
        group_table.groupby("label_combination", as_index=False, sort=True)
        .agg(
            conflicting_text_groups=(NORMALIZED_TEXT_HASH_COLUMN, "size"),
            rows_excluded=("rows_excluded", "sum"),
        )
        .sort_values("label_combination", ignore_index=True)
    )
    return summary


def experiment_ready_mask(
    frame: pd.DataFrame,
    *,
    valid_label_mask: pd.Series,
    valid_text_mask: pd.Series,
) -> pd.Series:
    """Select one canonical record per valid non-conflicting text-label pair."""
    return (
        valid_label_mask
        & valid_text_mask
        & ~frame["is_conflicting_label_text"]
        & ~frame["is_same_text_same_label_beyond_canonical"]
    )
