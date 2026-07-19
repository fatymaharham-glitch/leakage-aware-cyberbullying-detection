"""Dataset inventory, validation flags, and aggregate-only quality reporting."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable

import pandas as pd

from cyberbullying_detection.data.deduplication import (
    DuplicateAudit,
    annotate_exact_duplicates,
    create_row_id,
    experiment_ready_mask,
)
from cyberbullying_detection.data.validation import (
    DataValidationError,
    DatasetValidationReport,
    inspect_dataframe,
)


@dataclass(frozen=True)
class ValidatedDataset:
    """Validated records plus non-sensitive inventory and duplicate findings."""

    frame: pd.DataFrame
    validation: DatasetValidationReport
    duplicate_audit: DuplicateAudit
    inventory: dict[str, object]


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 checksum without loading raw text into memory."""
    digest = sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_missing(value: object) -> bool:
    """Check scalar missingness without coercing arbitrary collections."""
    if value is None:
        return True
    if not pd.api.types.is_scalar(value):
        return False
    return bool(pd.isna(value))


def _valid_label_mask(labels: pd.Series, expected_labels: Iterable[str]) -> pd.Series:
    """Return rows with exactly one configured string label."""
    allowed = set(expected_labels)
    return labels.map(lambda value: isinstance(value, str) and value in allowed)


def _text_flags(texts: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return missing, whitespace-only, and non-string text flags."""
    is_missing = texts.map(_is_missing)
    is_empty = texts.map(lambda value: isinstance(value, str) and not value.strip())
    is_non_string = ~(texts.map(lambda value: isinstance(value, str)) | is_missing)
    return is_missing, is_empty, is_non_string


def build_validated_dataset(
    frame: pd.DataFrame,
    *,
    input_path: Path,
    input_path_display: str,
    text_column: str,
    label_column: str,
    expected_labels: Iterable[str],
) -> ValidatedDataset:
    """Create source identities and audit flags while retaining every raw record.

    The returned frame is suitable for ignored local staging only. All reports derived
    from it use aggregate counts and never include tweet text.
    """
    expected = list(expected_labels)
    validation = inspect_dataframe(
        frame,
        text_column=text_column,
        label_column=label_column,
        expected_labels=expected,
    )
    if validation.missing_columns:
        missing = ", ".join(validation.missing_columns)
        raise DataValidationError(
            f"Dataset validation failed: missing required columns: {missing}"
        )

    annotated, duplicate_audit = annotate_exact_duplicates(
        frame, text_column=text_column, label_column=label_column
    )
    result = annotated.copy()
    result.insert(0, "source_row_number", range(1, len(result) + 1))
    result.insert(
        1,
        "row_id",
        [
            create_row_id(tweet_text, label)
            for tweet_text, label in zip(
                result[text_column], result[label_column], strict=True
            )
        ],
    )
    is_missing_text, is_empty_text, is_non_string_text = _text_flags(result[text_column])
    result["is_missing_text"] = is_missing_text
    result["is_empty_text"] = is_empty_text
    result["is_non_string_text"] = is_non_string_text
    result["is_valid_label"] = _valid_label_mask(result[label_column], expected)
    result["is_experiment_eligible"] = experiment_ready_mask(
        result,
        valid_label_mask=result["is_valid_label"],
        valid_text_mask=~(
            result["is_missing_text"]
            | result["is_empty_text"]
            | result["is_non_string_text"]
        ),
    )

    checksum = sha256_file(input_path)
    inventory: dict[str, object] = {
        "file_path": input_path_display,
        "sha256": checksum,
        "shape": {"rows": int(frame.shape[0]), "columns": int(frame.shape[1])},
        "column_names": list(frame.columns),
        "data_types": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "missing_values": {
            column: int(count) for column, count in frame.isna().sum().items()
        },
        "empty_or_whitespace_text_rows": int(is_empty_text.sum()),
        "missing_text_rows": int(is_missing_text.sum()),
        "non_string_text_rows": int(is_non_string_text.sum()),
        "invalid_label_rows": int((~result["is_valid_label"]).sum()),
        "invalid_or_unexpected_labels": validation.unexpected_labels,
        "missing_expected_labels": validation.missing_labels,
        "encoding": "utf-8",
        "encoding_or_parsing_errors": [],
        "exact_duplicate_audit": {
            "full_duplicate_rows_involved": duplicate_audit.full_duplicate_rows_involved,
            "full_duplicate_rows_beyond_first": duplicate_audit.full_duplicate_rows_beyond_first,
            "same_text_same_label_groups": duplicate_audit.same_text_same_label_groups,
            "same_text_same_label_rows_beyond_canonical": (
                duplicate_audit.same_text_same_label_rows_beyond_canonical
            ),
            "conflicting_label_groups": duplicate_audit.conflicting_label_groups,
            "conflicting_label_rows": duplicate_audit.conflicting_label_rows,
        },
    }
    return ValidatedDataset(result, validation, duplicate_audit, inventory)


def data_quality_report_markdown(inventory: dict[str, object]) -> str:
    """Render a compact, aggregate-only Markdown data-quality report."""
    shape = inventory["shape"]
    missing_values = inventory["missing_values"]
    duplicate_audit = inventory["exact_duplicate_audit"]
    missing_rows = "\n".join(
        f"| `{column}` | {count} |" for column, count in missing_values.items()
    )
    duplicate_rows = "\n".join(
        f"| {metric.replace('_', ' ')} | {count} |"
        for metric, count in duplicate_audit.items()
    )
    invalid_labels = inventory["invalid_or_unexpected_labels"] or ["None observed"]
    missing_labels = inventory["missing_expected_labels"] or ["None observed"]
    return f"""# Data quality report

This report contains aggregate metadata only. It intentionally excludes raw tweet text.

## Inventory

- File path: `{inventory['file_path']}`
- SHA-256: `{inventory['sha256']}`
- Shape: {shape['rows']} rows × {shape['columns']} columns
- Columns: {', '.join(f"`{column}`" for column in inventory['column_names'])}
- UTF-8 parsing errors: none observed

## Data types

| Column | Data type |
| --- | --- |
{chr(10).join(f"| `{column}` | `{dtype}` |" for column, dtype in inventory['data_types'].items())}

## Missing values

| Column | Missing rows |
| --- | --- |
{missing_rows}

- Missing text rows: {inventory['missing_text_rows']}
- Empty or whitespace-only text rows: {inventory['empty_or_whitespace_text_rows']}
- Non-string text rows: {inventory['non_string_text_rows']}
- Invalid label rows: {inventory['invalid_label_rows']}
- Invalid or unexpected labels: {', '.join(invalid_labels)}
- Expected labels not observed: {', '.join(missing_labels)}

## Exact duplicate and conflict audit

| Check | Count |
| --- | --- |
{duplicate_rows}

Conflicting-label text groups are excluded from `data/processed/experiment_ready.csv`.
Near duplicates are retained and controlled only through group-aware folds.
"""
