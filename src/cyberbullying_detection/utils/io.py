"""Small, explicit helpers for local tabular inputs and outputs."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd


class DataFileError(ValueError):
    """Raised when a requested dataset file cannot be safely read or written."""


def read_tabular_data(path: Path) -> pd.DataFrame:
    """Read a CSV or TSV dataset file with a clear failure message."""
    if not path.is_file():
        raise DataFileError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path, encoding="utf-8", low_memory=False)
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t", encoding="utf-8", low_memory=False)
    except UnicodeError as error:
        raise DataFileError(f"UTF-8 decoding failed for {path}: {error}") from error
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise DataFileError(f"CSV/TSV parsing failed for {path}: {error}") from error
    raise DataFileError("Input must be a .csv or .tsv file.")


def write_csv(frame: pd.DataFrame, path: Path, *, overwrite: bool = False) -> None:
    """Write a CSV without replacing an existing file unless requested."""
    if path.exists() and not overwrite:
        raise DataFileError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        try:
            frame.to_csv(temporary_file, index=False)
            temporary_path.replace(path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise


def write_json(payload: dict[str, Any], path: Path, *, overwrite: bool = False) -> None:
    """Write a JSON report without replacing an existing file unless requested."""
    if path.exists() and not overwrite:
        raise DataFileError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        try:
            json.dump(payload, temporary_file, indent=2, sort_keys=True)
            temporary_file.write("\n")
            temporary_path.replace(path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise


def write_text(content: str, path: Path, *, overwrite: bool = False) -> None:
    """Write UTF-8 text atomically without replacing unrelated files by default."""
    if path.exists() and not overwrite:
        raise DataFileError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        try:
            temporary_file.write(content)
            temporary_path.replace(path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
