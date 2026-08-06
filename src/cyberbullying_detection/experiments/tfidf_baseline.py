"""Grouped-fold TF-IDF baseline utilities that protect frozen test membership."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from cyberbullying_detection.preprocessing.text import preprocess_p0, preprocess_p1, preprocess_p2


class ExperimentError(ValueError):
    """Raised when an experiment configuration or split violates the protocol."""


PREPROCESSORS: dict[str, Callable[[object], str]] = {
    "p0": preprocess_p0,
    "p1": preprocess_p1,
    "p2": preprocess_p2,
}


@dataclass(frozen=True)
class BaselineResults:
    """Aggregate outputs from grouped cross-validation without predictions."""

    fold_metrics: pd.DataFrame
    summary_metrics: pd.DataFrame
    per_class_metrics: pd.DataFrame
    per_class_summary: pd.DataFrame
    runtime_metrics: pd.DataFrame


def _save_figure(figure: plt.Figure, output_path: Path) -> None:
    """Write a baseline figure atomically and close its Matplotlib resources."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.tmp.png")
    try:
        figure.savefig(temporary_path, format="png", dpi=180, bbox_inches="tight")
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
        plt.close(figure)


def write_baseline_figures(results: BaselineResults, output_directory: Path) -> None:
    """Create aggregate, text-free plots for the TF-IDF preprocessing baseline."""
    summary = results.summary_metrics.copy()
    preprocessing_order = ["p0", "p1", "p2"]
    model_order = ["logistic_regression", "linear_svm"]
    model_labels = {"logistic_regression": "Logistic Regression", "linear_svm": "Linear SVM"}
    colors = {"logistic_regression": "#6b7280", "linear_svm": "#1f4e79"}
    positions = np.arange(len(preprocessing_order))
    width = 0.34

    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    for offset, model_name in enumerate(model_order):
        subset = summary.set_index(["preprocessing", "model"])
        means = [subset.loc[(name, model_name), "macro_f1_mean"] for name in preprocessing_order]
        standard_deviations = [
            subset.loc[(name, model_name), "macro_f1_std"] for name in preprocessing_order
        ]
        axis.bar(
            positions + (offset - 0.5) * width,
            means,
            width=width,
            yerr=standard_deviations,
            capsize=4,
            color=colors[model_name],
            label=model_labels[model_name],
        )
    axis.set_title("Word TF-IDF baseline: macro-F1 by preprocessing")
    axis.set_xlabel("Preprocessing pipeline")
    axis.set_ylabel("Mean macro-F1 across grouped validation folds")
    axis.set_xticks(positions, [name.upper() for name in preprocessing_order])
    axis.set_ylim(0, 1)
    axis.legend()
    _save_figure(figure, output_directory / "macro_f1_comparison.png")

    recall = results.per_class_summary.pivot(
        index="label", columns=["preprocessing", "model"], values="recall_mean"
    )
    recall = recall.reindex(columns=pd.MultiIndex.from_product([preprocessing_order, model_order]))
    figure, axis = plt.subplots(figsize=(11, 5.4))
    image = axis.imshow(recall.to_numpy(), vmin=0, vmax=1, cmap="Blues", aspect="auto")
    axis.set_title("Per-class recall across grouped validation folds")
    axis.set_xlabel("Preprocessing and model")
    axis.set_ylabel("Class")
    axis.set_xticks(
        np.arange(len(recall.columns)),
        [f"{pre.upper()}\n{model_labels[model]}" for pre, model in recall.columns],
        rotation=35,
        ha="right",
    )
    axis.set_yticks(np.arange(len(recall.index)), recall.index)
    for row_index, label in enumerate(recall.index):
        for column_index, column in enumerate(recall.columns):
            value = recall.loc[label, column]
            axis.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value >= 0.55 else "black",
                fontsize=8,
            )
    figure.colorbar(image, ax=axis, label="Mean recall")
    _save_figure(figure, output_directory / "per_class_recall.png")

    runtime = (
        results.runtime_metrics.groupby(["preprocessing", "model"], as_index=False)
        .agg(
            elapsed_seconds_mean=("elapsed_seconds", "mean"),
            feature_count_mean=("feature_count", "mean"),
        )
        .set_index(["preprocessing", "model"])
    )
    labels = [f"{pre.upper()}\n{model_labels[model]}" for pre in preprocessing_order for model in model_order]
    elapsed = [runtime.loc[(pre, model), "elapsed_seconds_mean"] for pre in preprocessing_order for model in model_order]
    features = [runtime.loc[(pre, model), "feature_count_mean"] for pre in preprocessing_order for model in model_order]
    figure, (runtime_axis, feature_axis) = plt.subplots(1, 2, figsize=(12, 4.8))
    runtime_axis.bar(labels, elapsed, color="#4b8b3b")
    runtime_axis.set_title("Mean fit-and-predict runtime")
    runtime_axis.set_ylabel("Seconds per grouped validation fold")
    runtime_axis.tick_params(axis="x", rotation=35)
    feature_axis.bar(labels, features, color="#a35d24")
    feature_axis.set_title("Mean TF-IDF vocabulary size")
    feature_axis.set_ylabel("Features")
    feature_axis.tick_params(axis="x", rotation=35)
    figure.suptitle("Word TF-IDF baseline computational profile")
    figure.tight_layout()
    _save_figure(figure, output_directory / "runtime_and_features.png")


def build_pipeline(
    *,
    preprocessing_name: str,
    model_name: str,
    tfidf_config: dict[str, Any],
    model_config: dict[str, Any],
    random_seed: int,
) -> Pipeline:
    """Build a pipeline whose TF-IDF vocabulary is fitted only during ``fit``."""
    if preprocessing_name not in PREPROCESSORS:
        raise ExperimentError(f"Unknown preprocessing pipeline: {preprocessing_name}")
    if model_name == "logistic_regression":
        estimator = LogisticRegression(
            C=float(model_config["C"]),
            max_iter=int(model_config["max_iter"]),
            class_weight=model_config.get("class_weight"),
            random_state=random_seed,
        )
    elif model_name == "linear_svm":
        estimator = LinearSVC(
            C=float(model_config["C"]),
            max_iter=int(model_config["max_iter"]),
            class_weight=model_config.get("class_weight"),
            random_state=random_seed,
        )
    else:
        raise ExperimentError(f"Unsupported baseline model: {model_name}")

    ngram_range = tfidf_config.get("ngram_range")
    if not isinstance(ngram_range, list) or len(ngram_range) != 2:
        raise ExperimentError("tfidf.ngram_range must contain exactly two integers.")
    vectorizer = TfidfVectorizer(
        analyzer=tfidf_config["analyzer"],
        ngram_range=(int(ngram_range[0]), int(ngram_range[1])),
        min_df=int(tfidf_config["min_df"]),
        max_df=float(tfidf_config["max_df"]),
        max_features=int(tfidf_config["max_features"]),
        sublinear_tf=bool(tfidf_config["sublinear_tf"]),
        lowercase=False,
        preprocessor=PREPROCESSORS[preprocessing_name],
    )
    return Pipeline([("tfidf", vectorizer), ("model", estimator)])


def validate_experiment_inputs(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    row_id_column: str,
    label_column: str,
    fold_column: str,
    final_test_fold: int,
    cross_validation_folds: list[int],
) -> pd.DataFrame:
    """Join and validate assignments before any model is fitted."""
    required_data = {row_id_column, label_column}
    required_assignments = {row_id_column, fold_column}
    missing_data = sorted(required_data - set(frame.columns))
    missing_assignments = sorted(required_assignments - set(assignments.columns))
    if missing_data:
        raise ExperimentError(f"Input data missing columns: {', '.join(missing_data)}")
    if missing_assignments:
        raise ExperimentError(
            f"Fold assignments missing columns: {', '.join(missing_assignments)}"
        )
    if frame[row_id_column].duplicated().any() or assignments[row_id_column].duplicated().any():
        raise ExperimentError("row_id must be unique in data and fold assignments.")

    merged = frame.merge(
        assignments[[row_id_column, fold_column]],
        on=row_id_column,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(frame) or len(merged) != len(assignments):
        raise ExperimentError("Data and fold assignments must cover the same row_ids.")
    if final_test_fold in cross_validation_folds:
        raise ExperimentError("Frozen final-test fold cannot be used for cross-validation.")
    available_folds = set(merged[fold_column].astype(int))
    requested_folds = set(cross_validation_folds)
    if not requested_folds.issubset(available_folds):
        raise ExperimentError("Requested cross-validation folds are absent from assignments.")
    if final_test_fold not in available_folds:
        raise ExperimentError("Configured frozen final-test fold is absent from assignments.")
    return merged


def run_grouped_baseline(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    text_column: str,
    label_column: str,
    row_id_column: str,
    fold_column: str,
    final_test_fold: int,
    cross_validation_folds: list[int],
    preprocessing_names: list[str],
    tfidf_config: dict[str, Any],
    models_config: dict[str, dict[str, Any]],
    expected_labels: list[str],
    random_seed: int,
) -> BaselineResults:
    """Run grouped validation on non-frozen folds and return aggregate metrics."""
    merged = validate_experiment_inputs(
        frame,
        assignments,
        row_id_column=row_id_column,
        label_column=label_column,
        fold_column=fold_column,
        final_test_fold=final_test_fold,
        cross_validation_folds=cross_validation_folds,
    )
    if text_column not in merged.columns:
        raise ExperimentError(f"Input data missing text column: {text_column}")

    fold_records: list[dict[str, Any]] = []
    class_records: list[dict[str, Any]] = []
    runtime_records: list[dict[str, Any]] = []
    evaluation_pool = merged.loc[merged[fold_column] != final_test_fold].copy()
    for preprocessing_name in preprocessing_names:
        if preprocessing_name not in PREPROCESSORS:
            raise ExperimentError(f"Unknown preprocessing pipeline: {preprocessing_name}")
        for model_name, model_config in models_config.items():
            for validation_fold in cross_validation_folds:
                train = evaluation_pool.loc[evaluation_pool[fold_column] != validation_fold]
                validation = evaluation_pool.loc[evaluation_pool[fold_column] == validation_fold]
                if train.empty or validation.empty:
                    raise ExperimentError(f"Fold {validation_fold} has no train or validation rows.")
                pipeline = build_pipeline(
                    preprocessing_name=preprocessing_name,
                    model_name=model_name,
                    tfidf_config=tfidf_config,
                    model_config=model_config,
                    random_seed=random_seed,
                )
                started = perf_counter()
                pipeline.fit(train[text_column], train[label_column])
                predictions = pipeline.predict(validation[text_column])
                elapsed_seconds = perf_counter() - started
                precision, recall, f1, support = precision_recall_fscore_support(
                    validation[label_column],
                    predictions,
                    labels=expected_labels,
                    zero_division=0,
                )
                feature_count = len(pipeline.named_steps["tfidf"].vocabulary_)
                common = {
                    "preprocessing": preprocessing_name,
                    "model": model_name,
                    "validation_fold": int(validation_fold),
                    "train_rows": int(len(train)),
                    "validation_rows": int(len(validation)),
                }
                fold_records.append(
                    {
                        **common,
                        "macro_f1": float(
                            f1_score(
                                validation[label_column],
                                predictions,
                                labels=expected_labels,
                                average="macro",
                                zero_division=0,
                            )
                        ),
                        "balanced_accuracy": float(
                            balanced_accuracy_score(validation[label_column], predictions)
                        ),
                        "mcc": float(matthews_corrcoef(validation[label_column], predictions)),
                    }
                )
                runtime_records.append(
                    {**common, "elapsed_seconds": float(elapsed_seconds), "feature_count": feature_count}
                )
                for label, label_precision, label_recall, label_f1, label_support in zip(
                    expected_labels, precision, recall, f1, support, strict=True
                ):
                    class_records.append(
                        {
                            **common,
                            "label": label,
                            "precision": float(label_precision),
                            "recall": float(label_recall),
                            "f1": float(label_f1),
                            "support": int(label_support),
                        }
                    )

    fold_metrics = pd.DataFrame(fold_records).sort_values(
        ["preprocessing", "model", "validation_fold"], ignore_index=True
    )
    per_class_metrics = pd.DataFrame(class_records).sort_values(
        ["preprocessing", "model", "label", "validation_fold"], ignore_index=True
    )
    runtime_metrics = pd.DataFrame(runtime_records).sort_values(
        ["preprocessing", "model", "validation_fold"], ignore_index=True
    )
    summary_metrics = (
        fold_metrics.groupby(["preprocessing", "model"], as_index=False)
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            mcc_mean=("mcc", "mean"),
            mcc_std=("mcc", "std"),
        )
        .sort_values(["macro_f1_mean", "preprocessing", "model"], ascending=[False, True, True], ignore_index=True)
    )
    per_class_summary = (
        per_class_metrics.groupby(["preprocessing", "model", "label"], as_index=False)
        .agg(
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            support_total=("support", "sum"),
        )
        .sort_values(["preprocessing", "model", "label"], ignore_index=True)
    )
    return BaselineResults(
        fold_metrics=fold_metrics,
        summary_metrics=summary_metrics,
        per_class_metrics=per_class_metrics,
        per_class_summary=per_class_summary,
        runtime_metrics=runtime_metrics,
    )
