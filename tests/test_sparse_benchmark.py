"""Critical protocol and schema tests for sparse benchmarks."""

from __future__ import annotations

import pandas as pd
import pytest

from cyberbullying_detection.experiments import sparse_benchmark
from cyberbullying_detection.experiments.sparse_benchmark import (
    SparseBenchmarkResults,
    build_model,
    build_representation,
    random_oversample_indices,
    run_sparse_benchmark,
)
from cyberbullying_detection.experiments.tfidf_baseline import ExperimentError


def _synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame(
        {
            "row_id": ["f0a", "f0b", "v1a", "v1b", "v2a", "v2b"],
            "tweet_text": [
                "frozenmarker private age",
                "frozenmarker private gender",
                "age alpha",
                "gender beta",
                "age gamma",
                "gender delta",
            ],
            "cyberbullying_type": ["age", "gender", "age", "gender", "age", "gender"],
        }
    )
    assignments = pd.DataFrame(
        {"row_id": frame["row_id"], "fold": [0, 0, 1, 1, 2, 2]}
    )
    return frame, assignments


def test_sparse_benchmark_excludes_frozen_fold_from_vectorizer_fit(monkeypatch) -> None:
    frame, assignments = _synthetic_inputs()
    fitted_representations = []
    original_builder = sparse_benchmark.build_representation

    def recording_builder(name, config, preprocessing):
        representation = original_builder(name, config, preprocessing)
        fitted_representations.append(representation)
        return representation

    monkeypatch.setattr(sparse_benchmark, "build_representation", recording_builder)
    results = run_sparse_benchmark(
        frame,
        assignments,
        text_column="tweet_text",
        label_column="cyberbullying_type",
        row_id_column="row_id",
        fold_column="fold",
        final_test_fold=0,
        validation_folds=[1, 2],
        preprocessing="p1",
        representations={
            "word": {
                "analyzer": "word",
                "ngram_range": [1, 1],
                "min_df": 1,
                "max_df": 1.0,
                "max_features": 100,
            }
        },
        models={"logistic_regression": {"C": 1.0, "max_iter": 100}},
        expected_labels=["age", "gender"],
        random_seed=42,
    )

    assert isinstance(results, SparseBenchmarkResults)
    assert set(results.fold_metrics["validation_fold"]) == {1, 2}
    assert set(results.fold_metrics["train_rows"]) == {2}
    assert len(results.confusion_matrices) == 2 * 2 * 2
    assert results.confusion_matrices["count"].sum() == 4
    assert set(results.per_class_summary["label"]) == {"age", "gender"}
    assert results.fold_metrics["log_loss"].notna().all()
    assert results.fold_metrics["multiclass_brier"].notna().all()
    assert (results.runtime_metrics["train_matrix_mib"] > 0).all()
    assert all("frozenmarker" not in representation.vocabulary_ for representation in fitted_representations)


def test_combined_representation_disables_implicit_lowercasing() -> None:
    config = {
        "word": {
            "analyzer": "word",
            "ngram_range": [1, 1],
            "min_df": 1,
            "max_df": 1.0,
            "max_features": 100,
        },
        "character": {
            "analyzer": "char_wb",
            "ngram_range": [3, 5],
            "min_df": 1,
            "max_df": 1.0,
            "max_features": 100,
        },
    }

    representation = build_representation("combined", config, "p0")

    assert all(transformer.lowercase is False for _, transformer in representation.transformer_list)
    assert all(
        transformer.preprocessor("Hello") == "Hello"
        for _, transformer in representation.transformer_list
    )


def test_models_preserve_explicit_weighting_and_resource_controls() -> None:
    logistic = build_model(
        "logistic_regression_c0_5_balanced",
        {
            "estimator": "logistic_regression",
            "C": 0.5,
            "max_iter": 200,
            "class_weight": "balanced",
        },
        seed=42,
    )
    forest = build_model(
        "random_forest",
        {
            "n_estimators": 10,
            "max_depth": 4,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "n_jobs": 2,
        },
        seed=42,
    )

    assert logistic.class_weight == "balanced"
    assert logistic.random_state == 42
    assert forest.class_weight == "balanced"
    assert forest.n_jobs == 2


def test_sparse_benchmark_rejects_unknown_labels() -> None:
    frame, assignments = _synthetic_inputs()
    frame.loc[2, "cyberbullying_type"] = "unknown"

    with pytest.raises(ExperimentError, match="unexpected labels"):
        run_sparse_benchmark(
            frame,
            assignments,
            text_column="tweet_text",
            label_column="cyberbullying_type",
            row_id_column="row_id",
            fold_column="fold",
            final_test_fold=0,
            validation_folds=[1, 2],
            preprocessing="p1",
            representations={"word": {}},
            models={"logistic_regression": {}},
            expected_labels=["age", "gender"],
            random_seed=42,
        )


def test_random_oversampling_is_balanced_and_deterministic() -> None:
    labels = pd.Series(["age", "age", "age", "gender"])

    first = random_oversample_indices(labels, random_seed=42)
    second = random_oversample_indices(labels, random_seed=42)

    assert first.tolist() == second.tolist()
    assert labels.iloc[first].value_counts().to_dict() == {"age": 3, "gender": 3}


def test_random_oversampling_applies_only_after_fold_split() -> None:
    frame, assignments = _synthetic_inputs()
    results = run_sparse_benchmark(
        frame,
        assignments,
        text_column="tweet_text",
        label_column="cyberbullying_type",
        row_id_column="row_id",
        fold_column="fold",
        final_test_fold=0,
        validation_folds=[1, 2],
        preprocessing="p1",
        representations={
            "word": {
                "analyzer": "word",
                "ngram_range": [1, 1],
                "min_df": 1,
                "max_df": 1.0,
                "max_features": 100,
            }
        },
        models={
            "oversampled": {
                "estimator": "logistic_regression",
                "C": 1.0,
                "max_iter": 100,
                "oversampling": "random",
            }
        },
        expected_labels=["age", "gender"],
        random_seed=42,
    )

    assert set(results.fold_metrics["validation_rows"]) == {2}
    assert set(results.runtime_metrics["fitted_train_rows"]) == {2}
