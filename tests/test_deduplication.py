"""Tests for duplicate and conflicting-label controls."""

from __future__ import annotations

import pandas as pd

from cyberbullying_detection.data.deduplication import annotate_exact_duplicates, experiment_ready_mask


def test_conflicts_are_excluded_and_same_label_copies_are_canonicalised() -> None:
    frame = pd.DataFrame(
        {
            "tweet_text": ["same message", "same message", "shared text", "shared text", "unique"],
            "cyberbullying_type": ["age", "age", "gender", "religion", "not_cyberbullying"],
        }
    )

    annotated, audit = annotate_exact_duplicates(
        frame, text_column="tweet_text", label_column="cyberbullying_type"
    )
    mask = experiment_ready_mask(
        annotated,
        valid_label_mask=pd.Series(True, index=annotated.index),
        valid_text_mask=pd.Series(True, index=annotated.index),
    )

    assert audit.same_text_same_label_rows_beyond_canonical == 1
    assert audit.conflicting_label_groups == 1
    assert audit.conflicting_label_rows == 2
    assert mask.tolist() == [True, False, False, False, True]
