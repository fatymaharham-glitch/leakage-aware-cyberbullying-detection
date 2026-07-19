"""Frozen diagnostic and leakage-aware split assignment utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


class SplitError(ValueError):
    """Raised when source rows cannot support a deterministic safe split."""


@dataclass(frozen=True)
class SplitAssignments:
    """Metadata-only assignments and aggregate class distributions."""

    random_split: pd.DataFrame
    leakage_aware_folds: pd.DataFrame
    class_distribution: pd.DataFrame
    final_test_row_ids: list[str]


def _assert_unique_row_ids(frame: pd.DataFrame, row_id_column: str) -> None:
    """Ensure each experiment-ready record has exactly one stable identity."""
    if frame[row_id_column].isna().any():
        raise SplitError("row_id contains missing values.")
    if frame[row_id_column].duplicated().any():
        raise SplitError("row_id values must be unique in the experiment-ready dataset.")


def _class_distribution(
    assignments: pd.DataFrame,
    *,
    scheme: str,
    partition_column: str,
    label_column: str,
    expected_labels: list[str],
) -> pd.DataFrame:
    """Record count and percentage per label for each named partition."""
    records: list[dict[str, str | float | int]] = []
    for partition, subset in assignments.groupby(partition_column, sort=True):
        total = len(subset)
        counts = subset[label_column].value_counts().reindex(expected_labels, fill_value=0)
        for label, count in counts.items():
            records.append(
                {
                    "scheme": scheme,
                    "partition": str(partition),
                    "cyberbullying_type": label,
                    "row_count": int(count),
                    "class_percentage": float(count / total * 100) if total else 0.0,
                }
            )
    return pd.DataFrame(records).sort_values(
        ["scheme", "partition", "cyberbullying_type"], ignore_index=True
    )


def create_frozen_split_assignments(
    frame: pd.DataFrame,
    *,
    row_id_column: str,
    source_row_column: str,
    label_column: str,
    group_column: str,
    expected_labels: list[str],
    random_seed: int,
) -> SplitAssignments:
    """Create deterministic diagnostic and group-aware assignments without text data."""
    required_columns = {row_id_column, source_row_column, label_column, group_column}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise SplitError(f"Split input is missing columns: {', '.join(missing_columns)}")
    if frame.empty:
        raise SplitError("Cannot create frozen splits from an empty dataset.")
    _assert_unique_row_ids(frame, row_id_column)

    positions = list(range(len(frame)))
    _, test_positions = train_test_split(
        positions,
        test_size=0.20,
        random_state=random_seed,
        shuffle=True,
        stratify=frame[label_column],
    )
    random_partition = pd.Series("train", index=frame.index)
    random_partition.iloc[test_positions] = "test"
    random_split = frame[
        [row_id_column, source_row_column, label_column, group_column]
    ].copy()
    random_split["split"] = random_partition.to_numpy()
    random_split = random_split.sort_values(source_row_column, ignore_index=True)

    random_train_ids = set(random_split.loc[random_split["split"] == "train", row_id_column])
    random_test_ids = set(random_split.loc[random_split["split"] == "test", row_id_column])
    if random_train_ids & random_test_ids:
        raise SplitError("row_id overlap detected in the diagnostic random split.")
    if random_train_ids | random_test_ids != set(frame[row_id_column]):
        raise SplitError("Diagnostic random split does not cover every row_id exactly once.")

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_seed)
    fold_values = pd.Series(-1, index=frame.index, dtype="int64")
    for fold, (_, test_indices) in enumerate(
        splitter.split(frame, frame[label_column], groups=frame[group_column])
    ):
        fold_values.iloc[test_indices] = fold
    if (fold_values < 0).any():
        raise SplitError("At least one row was not assigned a leakage-aware fold.")

    leakage_aware_folds = frame[
        [row_id_column, source_row_column, label_column, group_column]
    ].copy()
    leakage_aware_folds["fold"] = fold_values.to_numpy()
    leakage_aware_folds = leakage_aware_folds.sort_values(source_row_column, ignore_index=True)
    group_fold_counts = leakage_aware_folds.groupby(group_column, sort=False)["fold"].nunique()
    if (group_fold_counts > 1).any():
        raise SplitError("A near-duplicate group crosses leakage-aware fold boundaries.")

    final_test_row_ids = sorted(
        leakage_aware_folds.loc[leakage_aware_folds["fold"] == 0, row_id_column]
    )
    if not final_test_row_ids:
        raise SplitError("Frozen final test fold 0 is empty.")
    fold_partition = leakage_aware_folds.copy()
    fold_partition["partition"] = fold_partition["fold"].map(
        lambda fold: f"fold_{fold}_test"
    )
    distributions = pd.concat(
        [
            _class_distribution(
                random_split,
                scheme="random_80_20",
                partition_column="split",
                label_column=label_column,
                expected_labels=expected_labels,
            ),
            _class_distribution(
                fold_partition,
                scheme="leakage_aware_5_fold",
                partition_column="partition",
                label_column=label_column,
                expected_labels=expected_labels,
            ),
        ],
        ignore_index=True,
    )
    return SplitAssignments(
        random_split=random_split,
        leakage_aware_folds=leakage_aware_folds,
        class_distribution=distributions,
        final_test_row_ids=final_test_row_ids,
    )
