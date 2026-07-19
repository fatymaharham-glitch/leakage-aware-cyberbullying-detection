"""Create the experiment-ready dataset with leakage-aware duplicate controls."""

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
    """Parse source and project-relative processed-output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="UTF-8 CSV or TSV relative to root.")
    parser.add_argument(
        "--output",
        default="data/processed/experiment_ready.csv",
        help="Processed CSV path relative to root; must stay in data/processed/.",
    )
    return parser.parse_args()


def save_figure(figure: object, output_path: Path) -> None:
    """Persist a matplotlib figure atomically and close it after writing."""
    from matplotlib import pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.tmp.png")
    try:
        figure.savefig(temporary_path, format="png", dpi=180, bbox_inches="tight")
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
        plt.close(figure)


def write_duplicate_figures(
    exact_summary: object, near_cluster_summary: object, figures_dir: Path
) -> None:
    """Write aggregate duplicate figures that contain neither tweets nor row IDs."""
    from matplotlib import pyplot as plt

    plot_metrics = {
        "full_duplicate_rows_beyond_first",
        "same_text_same_label_rows_beyond_canonical",
        "conflicting_label_groups",
        "conflicting_label_rows",
    }
    duplicate_plot = exact_summary.loc[exact_summary["metric"].isin(plot_metrics)].copy()
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(
        duplicate_plot["metric"].str.replace("_", " "),
        duplicate_plot["count"],
        color="#cc2529",
    )
    axis.set_title("Exact duplicate and label-conflict audit")
    axis.set_xlabel("Audit metric")
    axis.set_ylabel("Count")
    axis.tick_params(axis="x", rotation=25)
    save_figure(figure, figures_dir / "duplicate_audit.png")

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(
        near_cluster_summary["near_duplicate_group_size"].astype(str),
        near_cluster_summary["group_count"],
        color="#3e9651",
    )
    if (near_cluster_summary["group_count"] > 0).all():
        axis.set_yscale("log")
    axis.set_title("Near-duplicate cluster sizes at threshold 0.85")
    axis.set_xlabel("Near-duplicate group size")
    axis.set_ylabel("Group count (log scale)")
    save_figure(figure, figures_dir / "near_duplicate_cluster_sizes.png")


def main() -> int:
    """Build validated and experiment-ready data without changing raw tweet text."""
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
    from cyberbullying_detection.data.audit import (
        build_validated_dataset,
        data_quality_report_markdown,
    )
    from cyberbullying_detection.data.deduplication import (
        conflicting_label_summary_table,
        exact_duplicate_summary_table,
    )
    from cyberbullying_detection.data.features import FEATURE_COLUMNS, extract_text_features
    from cyberbullying_detection.data.near_duplicates import analyse_near_duplicates
    from cyberbullying_detection.data.validation import DataValidationError
    from cyberbullying_detection.utils.io import (
        DataFileError,
        read_tabular_data,
        write_csv,
        write_json,
        write_text,
    )
    from cyberbullying_detection.utils.logging import get_logger

    logger = get_logger("prepare_dataset")
    args = parse_args()
    try:
        preprocessing = load_yaml_config("preprocessing.yaml")
        paths = load_yaml_config("paths.yaml")
        input_path = resolve_project_path(args.input)
        output_path = resolve_project_path(args.output)
        interim_dir = configured_path(paths, "interim_data_dir")
        processed_dir = configured_path(paths, "processed_data_dir")
        reports_tables_dir = configured_path(paths, "report_tables_dir")
        reports_figures_dir = configured_path(paths, "report_figures_dir")
        reports_dir = reports_tables_dir.parent
        if input_path == output_path:
            raise DataFileError("Output path must differ from the raw input path.")
        if not output_path.is_relative_to(processed_dir):
            raise DataFileError("Experiment-ready data must be written inside data/processed/.")

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

        exact_summary = exact_duplicate_summary_table(validated.duplicate_audit)
        conflict_summary = conflicting_label_summary_table(
            validated.frame, label_column=preprocessing["expected_label_column"]
        )
        write_csv(
            exact_summary,
            reports_tables_dir / "exact_duplicate_summary.csv",
            overwrite=True,
        )
        write_csv(
            conflict_summary,
            reports_tables_dir / "conflicting_label_summary.csv",
            overwrite=True,
        )

        eligible = validated.frame.loc[
            validated.frame["is_experiment_eligible"]
        ].copy()
        if eligible.empty:
            raise DataValidationError("No rows remain after validation and conflict controls.")
        text_column = preprocessing["expected_text_column"]
        label_column = preprocessing["expected_label_column"]
        features = extract_text_features(eligible[text_column])
        eligible = eligible.join(features)
        near_duplicates = analyse_near_duplicates(
            eligible[text_column].tolist(),
            eligible["row_id"].tolist(),
            thresholds=preprocessing["near_duplicate_thresholds"],
            primary_threshold=preprocessing["primary_near_duplicate_threshold"],
            random_seed=preprocessing["random_seed"],
        )
        eligible = eligible.merge(
            near_duplicates.assignments,
            on="row_id",
            how="left",
            validate="one_to_one",
            sort=False,
        )
        if eligible["near_duplicate_group_id"].isna().any():
            raise DataValidationError("Near-duplicate grouping did not assign every row.")

        experiment_columns = [
            "source_row_number",
            "row_id",
            text_column,
            label_column,
            "near_duplicate_group_id",
            "near_duplicate_group_size",
            *FEATURE_COLUMNS,
        ]
        experiment_ready = eligible[experiment_columns].sort_values(
            "source_row_number", ignore_index=True
        )
        if experiment_ready["row_id"].duplicated().any():
            raise DataValidationError("Experiment-ready row_id values must be unique.")
        write_csv(experiment_ready, output_path, overwrite=True)
        write_csv(
            near_duplicates.threshold_analysis,
            reports_tables_dir / "near_duplicate_threshold_analysis.csv",
            overwrite=True,
        )
        write_csv(
            near_duplicates.cluster_summary,
            reports_tables_dir / "near_duplicate_cluster_summary.csv",
            overwrite=True,
        )
        write_duplicate_figures(
            exact_summary, near_duplicates.cluster_summary, reports_figures_dir
        )
    except (ConfigurationError, DataFileError, DataValidationError, KeyError, ValueError) as error:
        logger.error("Dataset preparation not completed: %s", error)
        return 2

    logger.info("Experiment-ready dataset written to %s", output_path)
    logger.info("Rows retained for experiment: %s", len(experiment_ready))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
