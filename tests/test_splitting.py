"""Tests for deterministic and group-safe split assignments."""

from __future__ import annotations

import pandas as pd

from cyberbullying_detection.data.splitting import create_frozen_split_assignments


def test_grouped_assignments_are_deterministic_and_keep_groups_together() -> None:
    records = []
    for position in range(10):
        records.append(
            {
                "row_id": f"row-{position}",
                "source_row_number": position + 1,
                "cyberbullying_type": "age" if position < 5 else "gender",
                "near_duplicate_group_id": f"group-{position}",
            }
        )
    frame = pd.DataFrame(records)

    first = create_frozen_split_assignments(
        frame,
        row_id_column="row_id",
        source_row_column="source_row_number",
        label_column="cyberbullying_type",
        group_column="near_duplicate_group_id",
        expected_labels=["age", "gender"],
        random_seed=42,
    )
    second = create_frozen_split_assignments(
        frame,
        row_id_column="row_id",
        source_row_column="source_row_number",
        label_column="cyberbullying_type",
        group_column="near_duplicate_group_id",
        expected_labels=["age", "gender"],
        random_seed=42,
    )

    assert first.random_split.equals(second.random_split)
    assert first.leakage_aware_folds.equals(second.leakage_aware_folds)
    assert first.random_split["row_id"].is_unique
    assert set(first.random_split["row_id"]) == set(frame["row_id"])
    assert first.leakage_aware_folds.groupby("near_duplicate_group_id")["fold"].nunique().eq(1).all()
    assert first.final_test_row_ids == second.final_test_row_ids
