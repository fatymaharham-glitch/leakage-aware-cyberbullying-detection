"""Tests for schema and label validation using harmless synthetic records."""

from __future__ import annotations

import pandas as pd
import pytest

from cyberbullying_detection.data.validation import DataValidationError, inspect_dataframe, validate_dataframe


EXPECTED = ["age", "ethnicity", "gender", "religion", "other_cyberbullying", "not_cyberbullying"]


def test_validate_dataframe_accepts_complete_expected_schema() -> None:
    frame = pd.DataFrame({"tweet_text": ["calm message"], "cyberbullying_type": ["age"]})

    report = validate_dataframe(
        frame,
        text_column="tweet_text",
        label_column="cyberbullying_type",
        expected_labels=EXPECTED,
    )

    assert report.has_errors is False


def test_inspect_dataframe_reports_missing_columns() -> None:
    report = inspect_dataframe(
        pd.DataFrame({"tweet_text": ["calm message"]}),
        text_column="tweet_text",
        label_column="cyberbullying_type",
        expected_labels=EXPECTED,
    )

    assert report.missing_columns == ["cyberbullying_type"]
    assert report.missing_labels == sorted(EXPECTED)


@pytest.mark.parametrize(
    ("text", "label", "expected_message"),
    [
        ("   ", "age", "empty tweet text"),
        (42, "age", "non-string tweet text"),
        ("calm message", "unknown", "unexpected labels"),
    ],
)
def test_validate_dataframe_rejects_invalid_rows(
    text: object, label: str, expected_message: str
) -> None:
    frame = pd.DataFrame({"tweet_text": [text], "cyberbullying_type": [label]})

    with pytest.raises(DataValidationError, match=expected_message):
        validate_dataframe(
            frame,
            text_column="tweet_text",
            label_column="cyberbullying_type",
            expected_labels=EXPECTED,
        )
