"""Compare grouped and random-stratified validation without using frozen fold 0."""

from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="leakage_gap.yaml")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "data/interim/.matplotlib"))
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    import matplotlib.pyplot as plt
    from cyberbullying_detection.config import ConfigurationError, load_yaml_config, resolve_project_path
    from cyberbullying_detection.data.audit import sha256_file
    from cyberbullying_detection.experiments.leakage_gap import (
        build_non_final_random_folds,
        count_groups_crossing_folds,
    )
    from cyberbullying_detection.experiments.sparse_benchmark import (
        ExperimentError,
        run_sparse_benchmark,
    )
    from cyberbullying_detection.utils.io import DataFileError, read_tabular_data, write_csv, write_json, write_text
    from cyberbullying_detection.utils.logging import get_logger

    logger = get_logger("leakage_gap")
    args = parse_args()
    try:
        config = load_yaml_config(args.config)
        preprocessing = load_yaml_config("preprocessing.yaml")
        input_path = resolve_project_path(config["input_data"])
        grouped_path = resolve_project_path(config["grouped_fold_assignments"])
        random_path = resolve_project_path(config["random_fold_assignments"])
        output_directory = resolve_project_path(config["outputs"]["directory"])
        if not output_directory.is_relative_to(resolve_project_path("reports")):
            raise DataFileError("Leakage-gap results must be written inside reports/.")
        frame = read_tabular_data(input_path)
        grouped = read_tabular_data(grouped_path)
        folds = [int(value) for value in config["cross_validation_folds"]]
        final_fold = int(config["final_test_fold"])
        seed = int(preprocessing["random_seed"])
        random_assignments = build_non_final_random_folds(
            frame,
            grouped,
            row_id_column=config["row_id_column"],
            label_column=config["label_column"],
            fold_column=config["fold_column"],
            final_test_fold=final_fold,
            validation_folds=folds,
            random_seed=seed,
        )
        write_csv(random_assignments, random_path, overwrite=args.overwrite)

        common = {
            "text_column": config["text_column"],
            "label_column": config["label_column"],
            "row_id_column": config["row_id_column"],
            "fold_column": config["fold_column"],
            "final_test_fold": final_fold,
            "validation_folds": folds,
            "preprocessing": config["preprocessing"],
            "representations": config["representation"],
            "models": config["model"],
            "expected_labels": preprocessing["expected_labels"],
            "random_seed": seed,
        }
        result_pairs = [
            ("grouped", run_sparse_benchmark(frame, grouped, **common)),
            ("random_stratified", run_sparse_benchmark(frame, random_assignments, **common)),
        ]
        table_names = [
            "fold_metrics",
            "summary_metrics",
            "per_class_metrics",
            "per_class_summary",
            "runtime_metrics",
            "confusion_matrices",
        ]
        output_directory.mkdir(parents=True, exist_ok=True)
        combined_tables: dict[str, pd.DataFrame] = {}
        for table_name in table_names:
            pieces = []
            for strategy, result in result_pairs:
                table = getattr(result, table_name).copy()
                table.insert(0, "split_strategy", strategy)
                pieces.append(table)
            combined_tables[table_name] = pd.concat(pieces, ignore_index=True)
            write_csv(
                combined_tables[table_name],
                output_directory / f"{table_name}.csv",
                overwrite=args.overwrite,
            )

        summary = combined_tables["summary_metrics"].set_index("split_strategy")
        metrics = ["macro_f1_mean", "balanced_accuracy_mean", "mcc_mean", "accuracy_mean"]
        gaps = pd.DataFrame(
            {
                "metric": metrics,
                "grouped": [float(summary.loc["grouped", metric]) for metric in metrics],
                "random_stratified": [
                    float(summary.loc["random_stratified", metric]) for metric in metrics
                ],
            }
        )
        gaps["random_minus_grouped"] = gaps["random_stratified"] - gaps["grouped"]
        write_csv(gaps, output_directory / "credibility_gap.csv", overwrite=args.overwrite)

        grouped_crossing = count_groups_crossing_folds(
            frame,
            grouped,
            row_id_column=config["row_id_column"],
            group_column=config["group_column"],
            fold_column=config["fold_column"],
            final_test_fold=final_fold,
        )
        random_crossing = count_groups_crossing_folds(
            frame,
            random_assignments,
            row_id_column=config["row_id_column"],
            group_column=config["group_column"],
            fold_column=config["fold_column"],
            final_test_fold=final_fold,
        )
        if grouped_crossing != 0:
            raise ExperimentError("Grouped assignments leak near-duplicate groups across folds.")
        manifest = {
            "config_file": args.config,
            "config_sha256": sha256_file(resolve_project_path(Path("configs") / args.config)),
            "input_sha256": sha256_file(input_path),
            "grouped_fold_assignments_sha256": sha256_file(grouped_path),
            "random_fold_assignments_sha256": sha256_file(random_path),
            "final_test_fold": final_fold,
            "cross_validation_folds": folds,
            "random_seed": seed,
            "candidate": config["model"],
            "representation": config["representation"],
            "groups_crossing_grouped_folds": grouped_crossing,
            "groups_crossing_random_folds": random_crossing,
            "python": platform.python_version(),
            "contains_predictions": False,
            "contains_raw_tweet_text": False,
        }
        write_json(manifest, output_directory / "experiment_manifest.json", overwrite=args.overwrite)
        macro_gap = float(gaps.loc[gaps["metric"] == "macro_f1_mean", "random_minus_grouped"].iloc[0])
        record = (
            "# P7 leakage credibility-gap decision\n\n"
            f"The same combined TF-IDF + balanced Logistic Regression candidate was evaluated "
            f"with grouped and random-stratified folds {folds}. Frozen fold {final_fold} remained "
            "unchanged and was never evaluated.\n\n"
            f"Random minus grouped macro-F1: **{macro_gap:+.4f}**. Grouped folds had "
            f"**{grouped_crossing}** near-duplicate groups crossing validation boundaries; random "
            f"folds had **{random_crossing}**. The grouped score remains the selection score because "
            "it better represents performance on genuinely different text.\n"
        )
        write_text(record, output_directory / "decision_record.md", overwrite=args.overwrite)
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.bar(gaps["metric"], gaps["random_minus_grouped"], color="#1f4e79")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_ylabel("Random minus grouped score")
        axis.set_title("Validation credibility gap")
        axis.tick_params(axis="x", rotation=25)
        figure.tight_layout()
        figure_path = output_directory / "figures/credibility_gap.png"
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(figure_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
    except (ConfigurationError, DataFileError, ExperimentError, KeyError, TypeError, ValueError) as error:
        logger.error("Leakage-gap analysis not completed: %s", error)
        return 2
    logger.info("Leakage-gap results written to %s", output_directory)
    logger.info("Frozen fold %s was not evaluated.", final_fold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
