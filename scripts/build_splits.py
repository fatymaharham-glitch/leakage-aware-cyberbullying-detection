"""Create deterministic diagnostic and frozen leakage-aware split assignments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def configure_source_path() -> None:
    """Make the project source package importable when executed as a script."""
    source_directory = str(PROJECT_ROOT / "src")
    if source_directory not in sys.path:
        sys.path.insert(0, source_directory)


def parse_args() -> argparse.Namespace:
    """Parse the project-relative experiment-ready dataset path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Experiment-ready CSV or TSV relative to the repository root.",
    )
    return parser.parse_args()


def frozen_manifest(
    input_checksum: str, final_test_row_ids: list[str], random_seed: int
) -> dict[str, Any]:
    """Build the immutable metadata used to prevent later test-fold replacement."""
    return {
        "schema_version": 1,
        "input_sha256": input_checksum,
        "random_seed": random_seed,
        "leakage_aware_n_splits": 5,
        "final_test_fold": 0,
        "final_test_row_ids": final_test_row_ids,
    }


def verify_existing_manifest(path: Path, expected_manifest: dict[str, Any]) -> bool:
    """Reject reruns that would modify the previously frozen final test membership."""
    if not path.exists():
        return False
    with path.open(encoding="utf-8") as manifest_file:
        existing_manifest = json.load(manifest_file)
    if existing_manifest != expected_manifest:
        raise ValueError(
            "Frozen test manifest differs from this input or assignment. Review the "
            "data change and intentionally replace the manifest only through a new "
            "research decision."
        )
    return True


def main() -> int:
    """Write metadata-only assignments and freeze leakage-aware fold zero as test."""
    configure_source_path()
    from cyberbullying_detection.config import (
        ConfigurationError,
        configured_path,
        load_yaml_config,
        resolve_project_path,
    )
    from cyberbullying_detection.data.audit import sha256_file
    from cyberbullying_detection.data.splitting import (
        SplitError,
        create_frozen_split_assignments,
    )
    from cyberbullying_detection.data.validation import DataValidationError, validate_dataframe
    from cyberbullying_detection.utils.io import (
        DataFileError,
        read_tabular_data,
        write_csv,
        write_json,
    )
    from cyberbullying_detection.utils.logging import get_logger

    logger = get_logger("build_splits")
    args = parse_args()
    try:
        preprocessing = load_yaml_config("preprocessing.yaml")
        paths = load_yaml_config("paths.yaml")
        input_path = resolve_project_path(args.input)
        processed_dir = configured_path(paths, "processed_data_dir")
        split_dir = configured_path(paths, "split_data_dir")
        tables_dir = configured_path(paths, "report_tables_dir")
        if not input_path.is_relative_to(processed_dir):
            raise DataFileError("Split input must come from data/processed/.")

        frame = read_tabular_data(input_path)
        validate_dataframe(
            frame,
            text_column=preprocessing["expected_text_column"],
            label_column=preprocessing["expected_label_column"],
            expected_labels=preprocessing["expected_labels"],
            require_all_labels=True,
        )
        assignments = create_frozen_split_assignments(
            frame,
            row_id_column="row_id",
            source_row_column="source_row_number",
            label_column=preprocessing["expected_label_column"],
            group_column="near_duplicate_group_id",
            expected_labels=preprocessing["expected_labels"],
            random_seed=preprocessing["random_seed"],
        )
        final_test_groups = set(
            assignments.leakage_aware_folds.loc[
                assignments.leakage_aware_folds["fold"] == 0,
                "near_duplicate_group_id",
            ]
        )
        train_groups = set(
            assignments.leakage_aware_folds.loc[
                assignments.leakage_aware_folds["fold"] != 0,
                "near_duplicate_group_id",
            ]
        )
        if final_test_groups & train_groups:
            raise SplitError(
                "A near_duplicate_group_id crosses the frozen train/test boundary."
            )

        manifest_path = split_dir / "frozen_test_manifest.json"
        expected_manifest = frozen_manifest(
            sha256_file(input_path),
            assignments.final_test_row_ids,
            preprocessing["random_seed"],
        )
        manifest_exists = verify_existing_manifest(manifest_path, expected_manifest)
        write_csv(
            assignments.random_split,
            split_dir / "random_split.csv",
            overwrite=True,
        )
        write_csv(
            assignments.leakage_aware_folds,
            split_dir / "leakage_aware_folds.csv",
            overwrite=True,
        )
        write_csv(
            assignments.class_distribution,
            tables_dir / "split_class_distribution.csv",
            overwrite=True,
        )
        if not manifest_exists:
            write_json(expected_manifest, manifest_path, overwrite=False)
    except (
        ConfigurationError,
        DataFileError,
        DataValidationError,
        KeyError,
        SplitError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        logger.error("Split creation not completed: %s", error)
        return 2

    logger.info("Random diagnostic split written to %s", split_dir / "random_split.csv")
    logger.info(
        "Leakage-aware folds written to %s; frozen fold 0 test size: %s",
        split_dir / "leakage_aware_folds.csv",
        len(assignments.final_test_row_ids),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
