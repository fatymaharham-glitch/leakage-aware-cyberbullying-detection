"""Inventory and validate a raw dataset without altering its CSV file."""

from __future__ import annotations

import argparse
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
    """Write validated staging data and aggregate-only audit reports."""
    configure_source_path()
    from cyberbullying_detection.config import (
        ConfigurationError,
        configured_path,
        load_yaml_config,
        resolve_project_path,
    )
    from cyberbullying_detection.data.audit import (
        build_validated_dataset,
        data_quality_report_markdown,
    )
    from cyberbullying_detection.data.validation import DataValidationError
    from cyberbullying_detection.utils.io import (
        DataFileError,
        read_tabular_data,
        write_csv,
        write_json,
        write_text,
    )
    from cyberbullying_detection.utils.logging import get_logger

    logger = get_logger("data_audit")
    args = parse_args()
    try:
        preprocessing = load_yaml_config("preprocessing.yaml")
        paths = load_yaml_config("paths.yaml")
        input_path = resolve_project_path(args.input)
        interim_dir = configured_path(paths, "interim_data_dir")
        reports_dir = configured_path(paths, "report_tables_dir").parent
        documentation_dir = configured_path(paths, "documentation_dir")
        raw_frame = read_tabular_data(input_path)
        validated = build_validated_dataset(
            raw_frame,
            input_path=input_path,
            input_path_display=args.input,
            text_column=preprocessing["expected_text_column"],
            label_column=preprocessing["expected_label_column"],
            expected_labels=preprocessing["expected_labels"],
        )
        write_csv(validated.frame, interim_dir / "validated.csv", overwrite=True)
        write_json(validated.inventory, interim_dir / "data_audit.json", overwrite=True)
        write_text(
            data_quality_report_markdown(validated.inventory),
            reports_dir / "data_quality_report.md",
            overwrite=True,
        )
        write_text(
            f"{validated.inventory['sha256']}  {args.input}\n",
            documentation_dir / "dataset_checksum.txt",
            overwrite=True,
        )
    except (ConfigurationError, DataFileError, DataValidationError, KeyError) as error:
        logger.error("Audit not completed: %s", error)
        return 2

    logger.info("Validated staging data written to %s", interim_dir / "validated.csv")
    logger.info("Aggregate quality report written to %s", reports_dir / "data_quality_report.md")
    if validated.validation.has_errors:
        logger.warning(
            "Audit found rows that preparation will exclude; inspect %s",
            reports_dir / "data_quality_report.md",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
