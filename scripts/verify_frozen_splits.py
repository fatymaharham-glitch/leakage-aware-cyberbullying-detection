"""Verify saved split assignments and frozen final-test membership without mutation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def configure_source_path() -> None:
    """Make the project source package importable when executed as a script."""
    source_directory = str(PROJECT_ROOT / "src")
    if source_directory not in sys.path:
        sys.path.insert(0, source_directory)


def parse_args() -> argparse.Namespace:
    """Parse project-relative input and aggregate-report paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/processed/experiment_ready.csv",
        help="Experiment-ready CSV relative to the repository root.",
    )
    parser.add_argument(
        "--output",
        default="reports/frozen_split_validation.md",
        help="Aggregate Markdown report path relative to the repository root.",
    )
    return parser.parse_args()


def main() -> int:
    """Fail closed when saved assignments are inconsistent with frozen membership."""
    configure_source_path()
    from cyberbullying_detection.config import (
        ConfigurationError,
        configured_path,
        load_yaml_config,
        resolve_project_path,
    )
    from cyberbullying_detection.data.audit import sha256_file
    from cyberbullying_detection.data.splitting import SplitError
    from cyberbullying_detection.data.validation import DataValidationError, validate_dataframe
    from cyberbullying_detection.utils.io import DataFileError, read_tabular_data, write_text
    from cyberbullying_detection.utils.logging import get_logger

    logger = get_logger("verify_frozen_splits")
    args = parse_args()
    try:
        preprocessing = load_yaml_config("preprocessing.yaml")
        paths = load_yaml_config("paths.yaml")
        input_path = resolve_project_path(args.input)
        output_path = resolve_project_path(args.output)
        processed_dir = configured_path(paths, "processed_data_dir")
        split_dir = configured_path(paths, "split_data_dir")
        reports_dir = configured_path(paths, "report_tables_dir").parent
        if not input_path.is_relative_to(processed_dir):
            raise DataFileError("Split verification input must come from data/processed/.")
        if not output_path.is_relative_to(reports_dir):
            raise DataFileError("Split verification report must be written inside reports/.")

        frame = read_tabular_data(input_path)
        validate_dataframe(
            frame,
            text_column=preprocessing["expected_text_column"],
            label_column=preprocessing["expected_label_column"],
            expected_labels=preprocessing["expected_labels"],
            require_all_labels=True,
        )
        random_split = read_tabular_data(split_dir / "random_split.csv")
        grouped_folds = read_tabular_data(split_dir / "leakage_aware_folds.csv")
        manifest_path = split_dir / "frozen_test_manifest.json"
        with manifest_path.open(encoding="utf-8") as source:
            manifest = json.load(source)

        required_assignment_columns = {
            "row_id",
            "source_row_number",
            preprocessing["expected_label_column"],
            "near_duplicate_group_id",
        }
        if missing := required_assignment_columns - set(random_split.columns):
            raise SplitError(f"Random split is missing columns: {', '.join(sorted(missing))}")
        if missing := required_assignment_columns - set(grouped_folds.columns):
            raise SplitError(f"Grouped folds are missing columns: {', '.join(sorted(missing))}")
        if "split" not in random_split.columns or "fold" not in grouped_folds.columns:
            raise SplitError("Split assignment files have missing partition columns.")

        input_ids = set(frame["row_id"])
        random_ids = set(random_split["row_id"])
        grouped_ids = set(grouped_folds["row_id"])
        if len(random_split) != len(random_ids) or random_ids != input_ids:
            raise SplitError("Random split does not contain every experiment-ready row exactly once.")
        if len(grouped_folds) != len(grouped_ids) or grouped_ids != input_ids:
            raise SplitError("Grouped folds do not contain every experiment-ready row exactly once.")
        if set(random_split["split"]) != {"train", "test"}:
            raise SplitError("Random split must contain train and test partitions only.")
        if set(grouped_folds["fold"]) != {0, 1, 2, 3, 4}:
            raise SplitError("Grouped folds must contain folds 0 through 4.")
        if (grouped_folds.groupby("near_duplicate_group_id")["fold"].nunique() > 1).any():
            raise SplitError("A near-duplicate group crosses grouped fold boundaries.")

        final_test_ids = sorted(grouped_folds.loc[grouped_folds["fold"] == 0, "row_id"])
        expected_checksum = sha256_file(input_path)
        expected_manifest = {
            "schema_version": 1,
            "input_sha256": expected_checksum,
            "random_seed": preprocessing["random_seed"],
            "leakage_aware_n_splits": 5,
            "final_test_fold": 0,
            "final_test_row_ids": final_test_ids,
        }
        if manifest != expected_manifest:
            raise SplitError("Frozen manifest does not match current grouped-fold assignments.")

        report = "\n".join(
            [
                "# Frozen Split Validation",
                "",
                "Status: PASS",
                "",
                "This aggregate report contains no tweet text, usernames, URLs, or row IDs.",
                "",
                "## Verified checks",
                "",
                f"- Experiment-ready rows: {len(frame)}",
                f"- Random split rows: {len(random_split)}",
                f"- Grouped-fold rows: {len(grouped_folds)}",
                f"- Frozen fold-0 test rows: {len(final_test_ids)}",
                f"- Input SHA-256 matches manifest: {manifest['input_sha256'] == expected_checksum}",
                "- Random split covers every experiment-ready row once.",
                "- Grouped folds cover every experiment-ready row once.",
                "- No near-duplicate group crosses grouped fold boundaries.",
                "- Saved manifest exactly matches current input, seed, folds, and fold-0 membership.",
                "",
                "## Operational rule",
                "",
                "Treat grouped fold 0 as frozen final-test membership. Do not use it for preprocessing, feature, threshold, or hyperparameter decisions.",
                "",
            ]
        )
        write_text(report, output_path, overwrite=True)
    except (
        ConfigurationError,
        DataFileError,
        DataValidationError,
        KeyError,
        OSError,
        SplitError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        logger.error("Frozen split verification failed: %s", error)
        return 2

    logger.info("Frozen split validation written to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
