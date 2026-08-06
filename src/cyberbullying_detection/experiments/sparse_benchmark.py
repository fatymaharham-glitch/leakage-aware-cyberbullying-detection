"""Leakage-safe sparse representation and classifier benchmark utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

from cyberbullying_detection.experiments.tfidf_baseline import (
    ExperimentError,
    PREPROCESSORS,
    validate_experiment_inputs,
)


@dataclass(frozen=True)
class SparseBenchmarkResults:
    """Aggregate benchmark outputs without raw text or row-level predictions."""

    fold_metrics: pd.DataFrame
    summary_metrics: pd.DataFrame
    per_class_metrics: pd.DataFrame
    per_class_summary: pd.DataFrame
    runtime_metrics: pd.DataFrame
    confusion_matrices: pd.DataFrame


def _vectorizer(config: dict[str, Any], preprocessing: str) -> TfidfVectorizer:
    ngrams = config.get("ngram_range")
    if not isinstance(ngrams, list) or len(ngrams) != 2:
        raise ExperimentError("representation.ngram_range must contain exactly two integers.")
    return TfidfVectorizer(
        analyzer=config["analyzer"],
        ngram_range=(int(ngrams[0]), int(ngrams[1])),
        min_df=int(config["min_df"]),
        max_df=float(config["max_df"]),
        max_features=int(config["max_features"]),
        sublinear_tf=True,
        lowercase=False,
        preprocessor=PREPROCESSORS[preprocessing],
    )


def build_representation(name: str, config: dict[str, Any], preprocessing: str):
    """Build word, character, or combined TF-IDF for fold-local fitting."""
    if preprocessing not in PREPROCESSORS:
        raise ExperimentError(f"Unknown preprocessing pipeline: {preprocessing}")
    if name in {"word", "character"}:
        return _vectorizer(config, preprocessing)
    if name == "combined":
        return FeatureUnion(
            [
                ("word", _vectorizer(config["word"], preprocessing)),
                ("character", _vectorizer(config["character"], preprocessing)),
            ]
        )
    raise ExperimentError(f"Unsupported representation: {name}")


def build_model(name: str, config: dict[str, Any], seed: int):
    """Build an approved classifier with explicit resource and seed controls."""
    estimator_name = config.get("estimator", name)
    if estimator_name == "multinomial_nb":
        return MultinomialNB(alpha=float(config["alpha"]))
    if estimator_name == "logistic_regression":
        return LogisticRegression(
            C=float(config["C"]),
            max_iter=int(config["max_iter"]),
            class_weight=config.get("class_weight"),
            random_state=seed,
        )
    if estimator_name == "linear_svm":
        return LinearSVC(
            C=float(config["C"]),
            max_iter=int(config["max_iter"]),
            class_weight=config.get("class_weight"),
            random_state=seed,
        )
    if estimator_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(config["n_estimators"]),
            max_depth=int(config["max_depth"]),
            max_features=config["max_features"],
            class_weight=config.get("class_weight"),
            n_jobs=int(config["n_jobs"]),
            random_state=seed,
        )
    if estimator_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ExperimentError("XGBoost is unavailable. Install requirements.txt.") from error
        return XGBClassifier(
            n_estimators=int(config["n_estimators"]),
            max_depth=int(config["max_depth"]),
            learning_rate=float(config["learning_rate"]),
            subsample=float(config["subsample"]),
            colsample_bytree=float(config["colsample_bytree"]),
            n_jobs=int(config["n_jobs"]),
            random_state=seed,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
        )
    raise ExperimentError(f"Unsupported classifier: {estimator_name}")


def _sparse_mebibytes(matrix: sparse.spmatrix) -> float:
    compressed = matrix.tocsr(copy=False)
    byte_count = compressed.data.nbytes + compressed.indices.nbytes + compressed.indptr.nbytes
    return float(byte_count / (1024**2))


def _feature_count(transformer: TfidfVectorizer | FeatureUnion) -> int:
    if isinstance(transformer, TfidfVectorizer):
        return len(transformer.vocabulary_)
    return sum(len(item[1].vocabulary_) for item in transformer.transformer_list)


def _validate_labels(frame: pd.DataFrame, label_column: str, expected_labels: list[str]) -> None:
    if len(expected_labels) != len(set(expected_labels)):
        raise ExperimentError("Expected labels must be unique.")
    actual_labels = set(frame[label_column].astype(str))
    unknown_labels = sorted(actual_labels - set(expected_labels))
    if unknown_labels:
        raise ExperimentError(f"Input data contain unexpected labels: {', '.join(unknown_labels)}")


def run_sparse_benchmark(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    text_column: str,
    label_column: str,
    row_id_column: str,
    fold_column: str,
    final_test_fold: int,
    validation_folds: list[int],
    preprocessing: str,
    representations: dict[str, dict[str, Any]],
    models: dict[str, dict[str, Any]],
    expected_labels: list[str],
    random_seed: int,
) -> SparseBenchmarkResults:
    """Evaluate sparse candidates across non-frozen grouped folds."""
    merged = validate_experiment_inputs(
        frame,
        assignments,
        row_id_column=row_id_column,
        label_column=label_column,
        fold_column=fold_column,
        final_test_fold=final_test_fold,
        cross_validation_folds=validation_folds,
    )
    if text_column not in merged:
        raise ExperimentError(f"Input data missing text column: {text_column}")
    if not representations or not models:
        raise ExperimentError("At least one representation and model are required.")
    _validate_labels(merged, label_column, expected_labels)

    pool = merged.loc[merged[fold_column] != final_test_fold].copy()
    fold_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    label_to_index = {label: index for index, label in enumerate(expected_labels)}

    for representation_name, representation_config in representations.items():
        for model_name, model_config in models.items():
            for fold in validation_folds:
                train = pool.loc[pool[fold_column] != fold]
                validation = pool.loc[pool[fold_column] == fold]
                if train.empty or validation.empty:
                    raise ExperimentError(f"Fold {fold} has no train or validation rows.")

                representation = build_representation(
                    representation_name, representation_config, preprocessing
                )
                model = build_model(model_name, model_config, random_seed)
                estimator_name = model_config.get("estimator", model_name)
                started = perf_counter()
                if estimator_name == "xgboost":
                    train_features = representation.fit_transform(train[text_column])
                    validation_features = representation.transform(validation[text_column])
                    train_targets = train[label_column].map(label_to_index)
                    if train_targets.isna().any():
                        raise ExperimentError("Training labels could not be encoded for XGBoost.")
                    sample_weight = None
                    if model_config.get("class_weight") == "balanced":
                        sample_weight = compute_sample_weight("balanced", train_targets)
                    model.fit(train_features, train_targets, sample_weight=sample_weight)
                    prediction_indices = np.asarray(model.predict(validation_features), dtype=int)
                    if np.any((prediction_indices < 0) | (prediction_indices >= len(expected_labels))):
                        raise ExperimentError("XGBoost returned an unknown class index.")
                    predictions = np.asarray(expected_labels, dtype=object)[prediction_indices]
                else:
                    pipeline = Pipeline([("representation", representation), ("model", model)])
                    pipeline.fit(train[text_column], train[label_column])
                    predictions = pipeline.predict(validation[text_column])
                    train_features = representation.transform(train[text_column])
                    validation_features = representation.transform(validation[text_column])
                elapsed = perf_counter() - started

                precision, recall, f1, support = precision_recall_fscore_support(
                    validation[label_column], predictions, labels=expected_labels, zero_division=0
                )
                common = {
                    "representation": representation_name,
                    "model": model_name,
                    "validation_fold": int(fold),
                    "train_rows": int(len(train)),
                    "validation_rows": int(len(validation)),
                }
                fold_rows.append(
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
                runtime_rows.append(
                    {
                        **common,
                        "elapsed_seconds": float(elapsed),
                        "feature_count": _feature_count(representation),
                        "train_matrix_mib": _sparse_mebibytes(train_features),
                        "validation_matrix_mib": _sparse_mebibytes(validation_features),
                    }
                )
                class_rows.extend(
                    {
                        **common,
                        "label": label,
                        "precision": float(value_precision),
                        "recall": float(value_recall),
                        "f1": float(value_f1),
                        "support": int(value_support),
                    }
                    for label, value_precision, value_recall, value_f1, value_support in zip(
                        expected_labels, precision, recall, f1, support, strict=True
                    )
                )
                matrix = confusion_matrix(
                    validation[label_column], predictions, labels=expected_labels
                )
                confusion_rows.extend(
                    {
                        **common,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "count": int(matrix[true_index, predicted_index]),
                    }
                    for true_index, true_label in enumerate(expected_labels)
                    for predicted_index, predicted_label in enumerate(expected_labels)
                )

    fold_metrics = pd.DataFrame(fold_rows).sort_values(
        ["representation", "model", "validation_fold"], ignore_index=True
    )
    per_class = pd.DataFrame(class_rows).sort_values(
        ["representation", "model", "label", "validation_fold"], ignore_index=True
    )
    runtime = pd.DataFrame(runtime_rows).sort_values(
        ["representation", "model", "validation_fold"], ignore_index=True
    )
    confusion = pd.DataFrame(confusion_rows).sort_values(
        ["representation", "model", "validation_fold", "true_label", "predicted_label"],
        ignore_index=True,
    )
    summary = (
        fold_metrics.groupby(["representation", "model"], as_index=False)
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            mcc_mean=("mcc", "mean"),
            mcc_std=("mcc", "std"),
        )
        .sort_values(
            ["macro_f1_mean", "representation", "model"],
            ascending=[False, True, True],
            ignore_index=True,
        )
    )
    per_class_summary = (
        per_class.groupby(["representation", "model", "label"], as_index=False)
        .agg(
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            support_total=("support", "sum"),
        )
        .sort_values(["representation", "model", "label"], ignore_index=True)
    )
    return SparseBenchmarkResults(
        fold_metrics=fold_metrics,
        summary_metrics=summary,
        per_class_metrics=per_class,
        per_class_summary=per_class_summary,
        runtime_metrics=runtime,
        confusion_matrices=confusion,
    )


def _save_figure(figure: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.tmp.png")
    try:
        figure.savefig(temporary_path, format="png", dpi=180, bbox_inches="tight")
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
        plt.close(figure)


def write_summary_plot(summary: pd.DataFrame, output_path: Path) -> None:
    """Save an aggregate, text-free macro-F1 comparison plot."""
    labels = [f"{row.representation}\n{row.model}" for row in summary.itertuples()]
    if len(summary) > 8:
        figure, axis = plt.subplots(figsize=(11, max(6.5, len(summary) * 0.55)))
        positions = np.arange(len(summary))
        axis.barh(
            positions,
            summary["macro_f1_mean"],
            xerr=summary["macro_f1_std"],
            capsize=3,
            color="#1f4e79",
        )
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.set_xlim(0, 1)
        axis.set_xlabel("Mean macro-F1")
    else:
        figure, axis = plt.subplots(figsize=(10, 5.5))
        axis.bar(
            labels,
            summary["macro_f1_mean"],
            yerr=summary["macro_f1_std"],
            capsize=4,
            color="#1f4e79",
        )
        axis.set_ylim(0, 1)
        axis.set_ylabel("Mean macro-F1")
        axis.tick_params(axis="x", rotation=35)
    axis.set_title("Leakage-safe sparse benchmark")
    _save_figure(figure, output_path)


def build_selection_record(
    *,
    config_file: str,
    results: SparseBenchmarkResults,
    models: dict[str, dict[str, Any]],
    validation_folds: list[int],
    final_test_fold: int,
) -> str:
    """Build a reproducible, cautious grouped-validation decision record."""
    summary = results.summary_metrics
    winner = summary.iloc[0]
    winner_classes = results.per_class_summary.loc[
        (results.per_class_summary["representation"] == winner["representation"])
        & (results.per_class_summary["model"] == winner["model"])
    ]
    weakest = winner_classes.sort_values("f1_mean").iloc[0]
    lines = [
        "# Benchmark selection record",
        "",
        f"Configuration: `{config_file}`. Evaluation used grouped folds {validation_folds}; "
        f"frozen fold {final_test_fold} was excluded.",
        "",
        "## Ranking",
        "",
        "| Rank | Representation | Configuration | Mean macro-F1 | SD |",
        "|---:|---|---|---:|---:|",
    ]
    for rank, row in enumerate(summary.head(3).itertuples(), start=1):
        lines.append(
            f"| {rank} | {row.representation} | {row.model} | "
            f"{row.macro_f1_mean:.4f} | {row.macro_f1_std:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Highest mean macro-F1: **{winner['representation']} + {winner['model']}** "
            f"({winner['macro_f1_mean']:.4f} ± {winner['macro_f1_std']:.4f}). "
            f"Weakest class by mean F1 for that candidate: **{weakest['label']}** "
            f"({weakest['f1_mean']:.4f}).",
            "",
        ]
    )

    model_families = {
        model_name: model_config.get("estimator", model_name)
        for model_name, model_config in models.items()
    }
    if len(model_families) > 2:
        family_rows = summary.assign(
            estimator=summary["model"].map(model_families)
        ).sort_values("macro_f1_mean", ascending=False)
        linear_shortlist = []
        for estimator in ("logistic_regression", "linear_svm"):
            candidates = family_rows.loc[family_rows["estimator"] == estimator]
            if not candidates.empty:
                linear_shortlist.append(candidates.iloc[0]["model"])
        lines.extend(
            [
                "## Week 6 shortlist",
                "",
                f"Retain **{linear_shortlist[0]}** as primary and **{linear_shortlist[1]}** "
                "as the faster linear comparator. Keep XGBoost only as a non-linear benchmark; "
                "its grouped macro-F1 is lower and its runtime is materially higher.",
                "",
            ]
        )
        if {
            "logistic_regression_c1",
            "logistic_regression_c1_balanced",
        }.issubset(set(summary["model"])):
            indexed = summary.set_index("model")
            weighting_delta = (
                indexed.loc["logistic_regression_c1_balanced", "macro_f1_mean"]
                - indexed.loc["logistic_regression_c1", "macro_f1_mean"]
            )
            lines.extend(
                [
                    f"Balanced class weighting changed Logistic Regression mean macro-F1 by "
                    f"{weighting_delta:+.4f}. Treat this as an ablation result, not proof of "
                    "a population-level improvement.",
                    "",
                ]
            )
    else:
        lines.extend(
            [
                "## Week 6 representation decision",
                "",
                f"Use **{winner['representation']}** TF-IDF for classifier benchmarking. "
                "No auxiliary hand-crafted features were added: incremental value is unproven, "
                "and heuristic identity/social features would widen the sensitivity surface.",
                "",
            ]
        )

    lines.extend(
        [
            "## Limits",
            "",
            "This is grouped-validation screening, not a frozen final-test result. Overlapping "
            "fold variability means small score differences are not yet statistically established; "
            "later statistical, robustness, calibration, and leakage-gap phases must inform final selection.",
            "",
        ]
    )
    return "\n".join(lines)
