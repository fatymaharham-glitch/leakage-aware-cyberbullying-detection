"""Tests for frozen-fold-safe TF-IDF baseline setup."""

from __future__ import annotations

import pandas as pd
import pytest

from cyberbullying_detection.experiments.tfidf_baseline import (
    ExperimentError,
    build_pipeline,
    validate_experiment_inputs,
)


def test_pipeline_preserves_p0_case_by_disabling_vectorizer_lowercasing() -> None:
    pipeline = build_pipeline(
        preprocessing_name="p0",
        model_name="logistic_regression",
        tfidf_config={
            "analyzer": "word",
            "ngram_range": [1, 2],
            "min_df": 1,
            "max_df": 1.0,
            "max_features": 100,
            "sublinear_tf": True,
        },
        model_config={"C": 1.0, "max_iter": 100, "class_weight": None},
        random_seed=42,
    )

    assert pipeline.named_steps["tfidf"].lowercase is False
    assert pipeline.named_steps["tfidf"].preprocessor("Hello") == "Hello"


def test_frozen_fold_cannot_be_requested_as_cross_validation_fold() -> None:
    frame = pd.DataFrame(
        {
            "row_id": ["a", "b", "c"],
            "cyberbullying_type": ["age", "gender", "age"],
        }
    )
    assignments = pd.DataFrame({"row_id": ["a", "b", "c"], "fold": [0, 1, 1]})

    with pytest.raises(ExperimentError, match="Frozen final-test fold"):
        validate_experiment_inputs(
            frame,
            assignments,
            row_id_column="row_id",
            label_column="cyberbullying_type",
            fold_column="fold",
            final_test_fold=0,
            cross_validation_folds=[0, 1],
        )
