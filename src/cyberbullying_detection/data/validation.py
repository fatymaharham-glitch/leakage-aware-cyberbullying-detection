"""Validation checks for the configured tweet text and class labels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import pandas as pd


class DataValidationError(ValueError):
    """Raised when a dataset cannot safely proceed through the workflow."""


@dataclass(frozen=True)
class DatasetValidationReport:
    """Structured validation findings suitable for an audit report."""

    missing_columns: list[str]
    unexpected_labels: list[str]
    missing_labels: list[str]
    empty_text_rows: list[object]
    non_string_text_rows: list[object]

    @property
    def has_errors(self) -> bool:
        """Return whether findings prevent safe preprocessing or splitting."""
        return bool(
            self.missing_columns
            or self.unexpected_labels
            or self.empty_text_rows
            or self.non_string_text_rows
        )

    def to_dict(self) -> dict[str, object]:
        """Convert findings to JSON-compatible primitives."""
        return asdict(self)


def _is_missing_value(value: object) -> bool:
    """Return whether a scalar dataset value is missing without coercing collections."""
    if value is None:
        return True
    if not pd.api.types.is_scalar(value):
        return False
    return bool(pd.isna(value))


def find_empty_text_rows(texts: pd.Series) -> list[object]:
    """Return indexes where tweet text is missing or whitespace-only."""
    empty_rows: list[object] = []
    for index, value in texts.items():
        if isinstance(value, str):
            if not value.strip():
                empty_rows.append(index)
            continue
        if _is_missing_value(value):
            empty_rows.append(index)
    return empty_rows


def find_non_string_text_rows(texts: pd.Series) -> list[object]:
    """Return indexes with present values that are not text strings."""
    invalid_rows: list[object] = []
    for index, value in texts.items():
        if isinstance(value, str) or _is_missing_value(value):
            continue
        invalid_rows.append(index)
    return invalid_rows


def validate_expected_labels(
    labels: Iterable[object], expected_labels: Iterable[str]
) -> list[str]:
    """Return observed labels that are not in the configured label set."""
    expected = set(expected_labels)
    unexpected: set[str] = set()
    for label in labels:
        if _is_missing_value(label):
            unexpected.add("<missing>")
        elif not isinstance(label, str) or label not in expected:
            unexpected.add(str(label))
    return sorted(unexpected)


def inspect_dataframe(
    frame: pd.DataFrame,
    *,
    text_column: str,
    label_column: str,
    expected_labels: Iterable[str],
) -> DatasetValidationReport:
    """Inspect a dataset without raising, preserving all audit findings."""
    expected = sorted(set(expected_labels))
    missing_columns = [
        column for column in (text_column, label_column) if column not in frame.columns
    ]
    if missing_columns:
        return DatasetValidationReport(missing_columns, [], expected, [], [])

    return DatasetValidationReport(
        missing_columns=[],
        unexpected_labels=validate_expected_labels(frame[label_column], expected),
        missing_labels=sorted(set(expected) - set(frame[label_column].dropna())),
        empty_text_rows=find_empty_text_rows(frame[text_column]),
        non_string_text_rows=find_non_string_text_rows(frame[text_column]),
    )


def validate_dataframe(
    frame: pd.DataFrame,
    *,
    text_column: str,
    label_column: str,
    expected_labels: Iterable[str],
    require_all_labels: bool = False,
) -> DatasetValidationReport:
    """Validate a frame and raise a readable error when it is unsafe to use."""
    report = inspect_dataframe(
        frame,
        text_column=text_column,
        label_column=label_column,
        expected_labels=expected_labels,
    )
    errors: list[str] = []
    if report.missing_columns:
        errors.append(f"missing columns: {', '.join(report.missing_columns)}")
    if report.unexpected_labels:
        errors.append(f"unexpected labels: {', '.join(report.unexpected_labels)}")
    if report.empty_text_rows:
        errors.append(f"empty tweet text at {len(report.empty_text_rows)} row(s)")
    if report.non_string_text_rows:
        errors.append(
            f"non-string tweet text at {len(report.non_string_text_rows)} row(s)"
        )
    if require_all_labels and report.missing_labels:
        errors.append(f"missing expected labels: {', '.join(report.missing_labels)}")
    if errors:
        raise DataValidationError("Dataset validation failed: " + "; ".join(errors))
    return report
