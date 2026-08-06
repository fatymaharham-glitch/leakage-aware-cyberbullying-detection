"""Tests for transparent P8 candidate selection rules."""

from __future__ import annotations

import pandas as pd
import pytest

from cyberbullying_detection.experiments.candidate_selection import (
    CandidateSelectionError,
    select_candidates,
)


def test_selects_best_then_fast_competitive_different_family() -> None:
    matrix = pd.DataFrame(
        [
            {"candidate_id": "lr", "family": "lr", "macro_f1_mean": 0.870, "runtime_seconds_mean": 30},
            {"candidate_id": "svm", "family": "svm", "macro_f1_mean": 0.869, "runtime_seconds_mean": 10},
            {"candidate_id": "slow_svm", "family": "svm", "macro_f1_mean": 0.8695, "runtime_seconds_mean": 20},
            {"candidate_id": "contextual", "family": "dense", "macro_f1_mean": 0.82, "runtime_seconds_mean": 3},
        ]
    )

    assert select_candidates(matrix) == ["lr", "svm"]


def test_rejects_selection_without_competitive_different_family() -> None:
    matrix = pd.DataFrame(
        [
            {"candidate_id": "lr", "family": "lr", "macro_f1_mean": 0.87, "runtime_seconds_mean": 30},
            {"candidate_id": "dense", "family": "dense", "macro_f1_mean": 0.80, "runtime_seconds_mean": 3},
        ]
    )

    with pytest.raises(CandidateSelectionError, match="No competitive candidate"):
        select_candidates(matrix)
