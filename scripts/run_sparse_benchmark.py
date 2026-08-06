"""Run a configured leakage-safe sparse representation or classifier benchmark."""

from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Configuration filename inside configs/.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "data" / "interim" / ".matplotlib"))
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from cyberbullying_detection.config import ConfigurationError, load_yaml_config, resolve_project_path
    from cyberbullying_detection.data.audit import sha256_file
    from cyberbullying_detection.data.validation import DataValidationError, validate_dataframe
    from cyberbullying_detection.experiments.sparse_benchmark import (
        build_selection_record,
        ExperimentError,
        run_sparse_benchmark,
        write_summary_plot,
    )
    from cyberbullying_detection.utils.io import DataFileError, read_tabular_data, write_csv, write_json, write_text
    from cyberbullying_detection.utils.logging import get_logger

    logger = get_logger("sparse_benchmark")
    args = parse_args()
    try:
        config = load_yaml_config(args.config)
        preprocessing = load_yaml_config("preprocessing.yaml")
        data_path = resolve_project_path(config["input_data"])
        folds_path = resolve_project_path(config["fold_assignments"])
        output_directory = resolve_project_path(config["outputs"]["directory"])
        reports_directory = resolve_project_path("reports")
        if not output_directory.is_relative_to(reports_directory):
            raise DataFileError("Benchmark results must be written inside reports/.")
        frame = read_tabular_data(data_path)
        assignments = read_tabular_data(folds_path)
        validate_dataframe(
            frame,
            text_column=config["text_column"],
            label_column=config["label_column"],
            expected_labels=preprocessing["expected_labels"],
            require_all_labels=True,
        )
        representations = config.get("representations") or {"selected": config["representation"]}
        results = run_sparse_benchmark(
            frame,
            assignments,
            text_column=config["text_column"],
            label_column=config["label_column"],
            row_id_column=config["row_id_column"],
            fold_column=config["fold_column"],
            final_test_fold=int(config["final_test_fold"]),
            validation_folds=[int(value) for value in config["cross_validation_folds"]],
            preprocessing=config["preprocessing"],
            representations=representations,
            models=config["models"],
            expected_labels=preprocessing["expected_labels"],
            random_seed=int(preprocessing["random_seed"]),
        )
        output_directory.mkdir(parents=True, exist_ok=True)
        tables = {
            "fold_metrics.csv": results.fold_metrics,
            "summary_metrics.csv": results.summary_metrics,
            "per_class_metrics.csv": results.per_class_metrics,
            "per_class_summary.csv": results.per_class_summary,
            "runtime_metrics.csv": results.runtime_metrics,
            "confusion_matrices.csv": results.confusion_matrices,
        }
        for filename, table in tables.items():
            write_csv(table, output_directory / filename, overwrite=args.overwrite)

        try:
            import xgboost

            xgboost_version = xgboost.__version__
        except ImportError:
            xgboost_version = None
        config_path = resolve_project_path(Path("configs") / args.config)
        manifest = {
            "config_file": args.config,
            "config_sha256": sha256_file(config_path),
            "input_data": str(data_path.relative_to(PROJECT_ROOT)),
            "input_sha256": sha256_file(data_path),
            "fold_assignments": str(folds_path.relative_to(PROJECT_ROOT)),
            "fold_assignments_sha256": sha256_file(folds_path),
            "final_test_fold": int(config["final_test_fold"]),
            "cross_validation_folds": [int(value) for value in config["cross_validation_folds"]],
            "random_seed": int(preprocessing["random_seed"]),
            "preprocessing": config["preprocessing"],
            "representations": representations,
            "models": config["models"],
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
                "xgboost": xgboost_version,
                "pyyaml": yaml.__version__,
            },
            "contains_predictions": False,
            "contains_raw_tweet_text": False,
        }
        write_json(manifest, output_directory / "experiment_manifest.json", overwrite=args.overwrite)

        decision = build_selection_record(
            config_file=args.config,
            results=results,
            models=config["models"],
            validation_folds=manifest["cross_validation_folds"],
            final_test_fold=manifest["final_test_fold"],
        )
        write_text(decision, output_directory / "selection_record.md", overwrite=args.overwrite)
        write_summary_plot(
            results.summary_metrics,
            output_directory / "figures" / "macro_f1_comparison.png",
        )
    except (
        ConfigurationError,
        DataFileError,
        DataValidationError,
        ExperimentError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        logger.error("Sparse benchmark not completed: %s", error)
        return 2

    logger.info("Benchmark results written to %s", output_directory)
    logger.info("Frozen fold %s was not evaluated.", config["final_test_fold"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
