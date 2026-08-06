"""Multi-criterion P8 candidate selection helpers."""

from __future__ import annotations

import pandas as pd


class CandidateSelectionError(ValueError):
    """Raised when P8 evidence cannot support a safe two-candidate freeze."""


def select_candidates(matrix: pd.DataFrame, *, macro_f1_tolerance: float = 0.005) -> list[str]:
    """Choose best grouped macro-F1, then a fast competitive different-family comparator."""
    required = {"candidate_id", "family", "macro_f1_mean", "runtime_seconds_mean"}
    missing = required - set(matrix.columns)
    if missing:
        raise CandidateSelectionError(
            f"Selection matrix missing columns: {', '.join(sorted(missing))}"
        )
    if matrix.empty or matrix["candidate_id"].duplicated().any():
        raise CandidateSelectionError("Selection matrix must contain unique candidates.")
    ranked = matrix.sort_values(
        ["macro_f1_mean", "runtime_seconds_mean"], ascending=[False, True]
    )
    primary = ranked.iloc[0]
    eligible = ranked.loc[
        (ranked["family"] != primary["family"])
        & (ranked["macro_f1_mean"] >= float(primary["macro_f1_mean"]) - macro_f1_tolerance)
    ]
    if eligible.empty:
        raise CandidateSelectionError("No competitive candidate from a different model family.")
    secondary = eligible.sort_values(
        ["runtime_seconds_mean", "macro_f1_mean"], ascending=[True, False]
    ).iloc[0]
    return [str(primary["candidate_id"]), str(secondary["candidate_id"])]
