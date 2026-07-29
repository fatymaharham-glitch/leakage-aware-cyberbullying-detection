"""Run a configured leakage-safe sparse representation or classifier benchmark."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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
    from cyberbullying_detection.config import load_yaml_config, resolve_project_path
    from cyberbullying_detection.data.validation import validate_dataframe
    from cyberbullying_detection.experiments.sparse_benchmark import run_sparse_benchmark, write_summary_plot
    from cyberbullying_detection.utils.io import read_tabular_data, write_csv

    args = parse_args()
    config = load_yaml_config(args.config)
    preprocessing = load_yaml_config("preprocessing.yaml")
    data_path = resolve_project_path(config["input_data"])
    folds_path = resolve_project_path(config["fold_assignments"])
    output_directory = resolve_project_path(config["outputs"]["directory"])
    frame = read_tabular_data(data_path)
    assignments = read_tabular_data(folds_path)
    validate_dataframe(frame, text_column=config["text_column"], label_column=config["label_column"], expected_labels=preprocessing["expected_labels"], require_all_labels=True)
    representations = config.get("representations") or {"selected": config["representation"]}
    fold_metrics, summary, per_class, runtime = run_sparse_benchmark(
        frame, assignments, text_column=config["text_column"], label_column=config["label_column"],
        row_id_column=config["row_id_column"], fold_column=config["fold_column"], final_test_fold=int(config["final_test_fold"]),
        validation_folds=[int(value) for value in config["cross_validation_folds"]], preprocessing=config["preprocessing"],
        representations=representations, models=config["models"], expected_labels=preprocessing["expected_labels"],
        random_seed=int(preprocessing["random_seed"]),
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    for filename, table in {"fold_metrics.csv": fold_metrics, "summary_metrics.csv": summary, "per_class_metrics.csv": per_class, "runtime_metrics.csv": runtime}.items():
        write_csv(table, output_directory / filename, overwrite=args.overwrite)
    write_summary_plot(summary, output_directory / "figures" / "macro_f1_comparison.png")
    print(f"Benchmark results written to {output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
