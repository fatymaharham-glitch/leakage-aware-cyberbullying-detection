"""Critical cache and frozen-fold tests for the contextual baseline."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cyberbullying_detection.experiments.contextual_baseline import (
    ExperimentError,
    evaluate_contextual_embeddings,
    load_embedding_cache,
    prepare_non_final_pool,
    validate_cache_manifest,
    write_embedding_cache,
)


def test_contextual_pool_and_cache_exclude_frozen_rows(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "row_id": ["frozen-a", "frozen-b", "one-a", "one-b", "two-a", "two-b"],
            "cyberbullying_type": ["age", "gender", "age", "gender", "age", "gender"],
        }
    )
    assignments = pd.DataFrame(
        {"row_id": frame["row_id"], "fold": [0, 0, 1, 1, 2, 2]}
    )
    pool = prepare_non_final_pool(
        frame,
        assignments,
        row_id_column="row_id",
        label_column="cyberbullying_type",
        fold_column="fold",
        final_test_fold=0,
        validation_folds=[1, 2],
    )
    row_ids = pool["row_id"].tolist()
    embeddings = np.eye(4, dtype=np.float32)
    cache_path = tmp_path / "embeddings.npz"

    write_embedding_cache(
        cache_path,
        embeddings=embeddings,
        row_ids=row_ids,
        overwrite=False,
    )
    loaded = load_embedding_cache(
        cache_path,
        expected_row_ids=row_ids,
        expected_dimension=4,
    )

    assert set(pool["fold"]) == {1, 2}
    assert not set(row_ids).intersection({"frozen-a", "frozen-b"})
    np.testing.assert_array_equal(loaded, embeddings)


def test_embedding_cache_rejects_row_order_and_manifest_changes(tmp_path) -> None:
    cache_path = tmp_path / "embeddings.npz"
    write_embedding_cache(
        cache_path,
        embeddings=np.ones((2, 3), dtype=np.float32),
        row_ids=["a", "b"],
        overwrite=False,
    )
    with pytest.raises(ExperimentError, match="row order"):
        load_embedding_cache(
            cache_path,
            expected_row_ids=["b", "a"],
            expected_dimension=3,
        )

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"model": "revision-a"}), encoding="utf-8")
    with pytest.raises(ExperimentError, match="manifest mismatch"):
        validate_cache_manifest(manifest_path, {"model": "revision-b"})


def test_contextual_evaluation_returns_complete_aggregate_evidence() -> None:
    pool = pd.DataFrame(
        {
            "cyberbullying_type": ["age", "gender", "age", "gender", "age", "gender"],
            "fold": [1, 1, 2, 2, 3, 3],
        }
    )
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
            [0.1, 0.9],
            [0.8, 0.2],
            [0.2, 0.8],
        ],
        dtype=np.float32,
    )
    results = evaluate_contextual_embeddings(
        pool,
        embeddings,
        label_column="cyberbullying_type",
        fold_column="fold",
        validation_folds=[1, 2, 3],
        models={
            "logistic": {
                "estimator": "logistic_regression",
                "C": 1.0,
                "max_iter": 100,
                "class_weight": None,
            }
        },
        expected_labels=["age", "gender"],
        random_seed=42,
    )

    assert len(results.fold_metrics) == 3
    assert len(results.per_class_metrics) == 6
    assert len(results.confusion_matrices) == 12
    assert results.confusion_matrices["count"].sum() == 6
    assert results.summary_metrics.iloc[0]["macro_f1_mean"] == pytest.approx(1.0)
