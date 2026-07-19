"""Generate aggregate EDA tables and figures from the raw local dataset."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def configure_source_path() -> None:
    """Make the project source package importable when executed as a script."""
    source_directory = str(PROJECT_ROOT / "src")
    if source_directory not in sys.path:
        sys.path.insert(0, source_directory)


def parse_args() -> argparse.Namespace:
    """Parse the required project-relative raw dataset path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="UTF-8 CSV or TSV relative to root.")
    return parser.parse_args()


def main() -> int:
    """Write deterministic aggregate EDA outputs without displaying tweet content."""
    configure_source_path()
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault(
        "MPLCONFIGDIR", str(PROJECT_ROOT / "data" / "interim" / ".matplotlib")
    )
    from cyberbullying_detection.config import (
        ConfigurationError,
        configured_path,
        load_yaml_config,
        resolve_project_path,
    )
    from cyberbullying_detection.data.eda import build_eda_tables, write_eda_outputs
    from cyberbullying_detection.data.validation import DataValidationError, inspect_dataframe
    from cyberbullying_detection.utils.io import DataFileError, read_tabular_data
    from cyberbullying_detection.utils.logging import get_logger

    logger = get_logger("eda")
    args = parse_args()
    try:
        preprocessing = load_yaml_config("preprocessing.yaml")
        paths = load_yaml_config("paths.yaml")
        input_path = resolve_project_path(args.input)
        raw_frame = read_tabular_data(input_path)
        validation = inspect_dataframe(
            raw_frame,
            text_column=preprocessing["expected_text_column"],
            label_column=preprocessing["expected_label_column"],
            expected_labels=preprocessing["expected_labels"],
        )
        if validation.missing_columns:
            raise DataValidationError(
                "Dataset validation failed: missing required columns: "
                + ", ".join(validation.missing_columns)
            )
        tables = build_eda_tables(
            raw_frame,
            text_column=preprocessing["expected_text_column"],
            label_column=preprocessing["expected_label_column"],
            expected_labels=preprocessing["expected_labels"],
        )
        write_eda_outputs(
            tables,
            figures_dir=configured_path(paths, "report_figures_dir"),
            tables_dir=configured_path(paths, "report_tables_dir"),
        )
    except (ConfigurationError, DataFileError, DataValidationError, KeyError) as error:
        logger.error("EDA not completed: %s", error)
        return 2

    logger.info("Aggregate EDA tables written to %s", configured_path(paths, "report_tables_dir"))
    logger.info("EDA figures written to %s", configured_path(paths, "report_figures_dir"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
