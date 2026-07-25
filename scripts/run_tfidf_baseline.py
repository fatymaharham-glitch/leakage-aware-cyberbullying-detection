"""Run leakage-safe grouped-fold word TF-IDF preprocessing baselines."""

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
    """Parse the baseline configuration and explicit overwrite choice."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="tfidf_baseline.yaml",
        help="Configuration filename inside configs/.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing aggregate baseline result files.",
    )
    return parser.parse_args()


def main() -> int:
    """Run non-frozen grouped cross-validation and save aggregate outputs."""
    configure_source_path()
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "data" / "interim" / ".matplotlib"))
    from cyberbullying_detection.config import ConfigurationError, load_yaml_config, resolve_project_path
    from cyberbullying_detection.data.audit import sha256_file
    from cyberbullying_detection.data.validation import DataValidationError, validate_dataframe
    from cyberbullying_detection.experiments.tfidf_baseline import (
        ExperimentError,
        run_grouped_baseline,
        write_baseline_figures,
    )
    from cyberbullying_detection.utils.io import DataFileError, read_tabular_data, write_csv, write_json
    from cyberbullying_detection.utils.logging import get_logger

    logger = get_logger("tfidf_baseline")
    args = parse_args()
    try:
        preprocessing = load_yaml_config("preprocessing.yaml")
        experiment = load_yaml_config(args.config)
        input_path = resolve_project_path(experiment["input_data"])
        assignments_path = resolve_project_path(experiment["fold_assignments"])
        output_dir = resolve_project_path(experiment["outputs"]["directory"])
        reports_dir = resolve_project_path("reports")
        if not output_dir.is_relative_to(reports_dir):
            raise DataFileError("Baseline results must be written inside reports/.")
        frame = read_tabular_data(input_path)
        assignments = read_tabular_data(assignments_path)
        validate_dataframe(
            frame,
            text_column=experiment["text_column"],
            label_column=experiment["label_column"],
            expected_labels=preprocessing["expected_labels"],
            require_all_labels=True,
        )
        results = run_grouped_baseline(
            frame,
            assignments,
            text_column=experiment["text_column"],
            label_column=experiment["label_column"],
            row_id_column=experiment["row_id_column"],
            fold_column=experiment["fold_column"],
            final_test_fold=int(experiment["final_test_fold"]),
            cross_validation_folds=[int(fold) for fold in experiment["cross_validation_folds"]],
            preprocessing_names=experiment["preprocessing_pipelines"],
            tfidf_config=experiment["tfidf"],
            models_config=experiment["models"],
            expected_labels=preprocessing["expected_labels"],
            random_seed=int(preprocessing["random_seed"]),
        )
        manifest = {
            "config_file": args.config,
            "input_data": str(input_path.relative_to(PROJECT_ROOT)),
            "input_sha256": sha256_file(input_path),
            "fold_assignments": str(assignments_path.relative_to(PROJECT_ROOT)),
            "fold_assignments_sha256": sha256_file(assignments_path),
            "final_test_fold": int(experiment["final_test_fold"]),
            "cross_validation_folds": [int(fold) for fold in experiment["cross_validation_folds"]],
            "random_seed": int(preprocessing["random_seed"]),
            "contains_predictions": False,
            "contains_raw_tweet_text": False,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, result in {
            "fold_metrics.csv": results.fold_metrics,
            "summary_metrics.csv": results.summary_metrics,
            "per_class_metrics.csv": results.per_class_metrics,
            "per_class_summary.csv": results.per_class_summary,
            "runtime_metrics.csv": results.runtime_metrics,
        }.items():
            write_csv(result, output_dir / name, overwrite=args.overwrite)
        write_json(manifest, output_dir / "experiment_manifest.json", overwrite=args.overwrite)
        write_baseline_figures(results, output_dir / "figures")
    except (ConfigurationError, DataFileError, DataValidationError, ExperimentError, KeyError, TypeError, ValueError) as error:
        logger.error("TF-IDF baseline not completed: %s", error)
        return 2

    logger.info("Baseline metrics written to %s", output_dir)
    logger.info("Frozen fold %s was not evaluated.", experiment["final_test_fold"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
