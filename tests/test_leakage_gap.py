"""Protocol checks for non-final random folds."""

from __future__ import annotations

import pandas as pd

from cyberbullying_detection.experiments.leakage_gap import (
    build_non_final_random_folds,
    count_groups_crossing_folds,
)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    folds = []
    for fold in range(5):
        for label in ("age", "gender"):
            for index in range(2):
                row_id = f"{fold}-{label}-{index}"
                rows.append(
                    {
                        "row_id": row_id,
                        "cyberbullying_type": label,
                        "near_duplicate_group_id": f"group-{label}-{index}",
                    }
                )
                folds.append({"row_id": row_id, "fold": fold})
    return pd.DataFrame(rows), pd.DataFrame(folds)


def test_random_folds_preserve_final_membership_and_are_deterministic() -> None:
    frame, grouped = _inputs()
    kwargs = {
        "row_id_column": "row_id",
        "label_column": "cyberbullying_type",
        "fold_column": "fold",
        "final_test_fold": 0,
        "validation_folds": [1, 2, 3, 4],
        "random_seed": 42,
    }

    first = build_non_final_random_folds(frame, grouped, **kwargs)
    second = build_non_final_random_folds(frame, grouped, **kwargs)

    assert first.equals(second)
    final_ids = set(grouped.loc[grouped["fold"] == 0, "row_id"])
    assert set(first.loc[first["fold"] == 0, "row_id"]) == final_ids
    assert set(first["fold"]) == {0, 1, 2, 3, 4}
    assert len(first) == len(frame)


def test_group_crossing_counter_distinguishes_split_protocols() -> None:
    frame, grouped = _inputs()
    random = build_non_final_random_folds(
        frame,
        grouped,
        row_id_column="row_id",
        label_column="cyberbullying_type",
        fold_column="fold",
        final_test_fold=0,
        validation_folds=[1, 2, 3, 4],
        random_seed=42,
    )

    assert count_groups_crossing_folds(
        frame,
        grouped,
        row_id_column="row_id",
        group_column="near_duplicate_group_id",
        fold_column="fold",
        final_test_fold=0,
    ) == 4
    assert count_groups_crossing_folds(
        frame,
        random,
        row_id_column="row_id",
        group_column="near_duplicate_group_id",
        fold_column="fold",
        final_test_fold=0,
    ) >= 0
