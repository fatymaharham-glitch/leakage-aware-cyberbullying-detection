"""Frozen-encoder contextual baseline with protected final-test membership."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.svm import LinearSVC

from cyberbullying_detection.experiments.tfidf_baseline import (
    ExperimentError,
    validate_experiment_inputs,
)


@dataclass(frozen=True)
class ContextualResults:
    """Aggregate contextual results without text or row-level predictions."""

    fold_metrics: pd.DataFrame
    summary_metrics: pd.DataFrame
    per_class_metrics: pd.DataFrame
    per_class_summary: pd.DataFrame
    runtime_metrics: pd.DataFrame
    confusion_matrices: pd.DataFrame


def row_order_sha256(row_ids: list[str]) -> str:
    """Hash an ordered row-ID sequence without exposing its text data."""
    digest = hashlib.sha256()
    for row_id in row_ids:
        digest.update(row_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def prepare_non_final_pool(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    row_id_column: str,
    label_column: str,
    fold_column: str,
    final_test_fold: int,
    validation_folds: list[int],
) -> pd.DataFrame:
    """Return only non-final rows after applying shared split validation."""
    merged = validate_experiment_inputs(
        frame,
        assignments,
        row_id_column=row_id_column,
        label_column=label_column,
        fold_column=fold_column,
        final_test_fold=final_test_fold,
        cross_validation_folds=validation_folds,
    )
    pool = merged.loc[merged[fold_column] != final_test_fold].copy()
    if pool.empty:
        raise ExperimentError("No non-final rows are available for contextual evaluation.")
    return pool


def write_embedding_cache(
    path: Path,
    *,
    embeddings: np.ndarray,
    row_ids: list[str],
    overwrite: bool,
) -> None:
    """Atomically write numeric embeddings aligned to ordered row IDs."""
    if path.exists() and not overwrite:
        raise ExperimentError(f"Refusing to overwrite embedding cache: {path}")
    if embeddings.ndim != 2 or len(embeddings) != len(row_ids):
        raise ExperimentError("Embedding cache row count does not match row IDs.")
    if not np.isfinite(embeddings).all():
        raise ExperimentError("Embedding cache contains non-finite values.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as temporary_file:
        temporary_path = Path(temporary_file.name)
        try:
            np.savez_compressed(
                temporary_file,
                embeddings=np.asarray(embeddings, dtype=np.float32),
                row_ids=np.asarray(row_ids, dtype=str),
            )
            temporary_path.replace(path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise


def load_embedding_cache(
    path: Path,
    *,
    expected_row_ids: list[str],
    expected_dimension: int,
) -> np.ndarray:
    """Load a cache only when shape, order, and values match expectations."""
    if not path.is_file():
        raise ExperimentError(f"Embedding cache not found: {path}")
    try:
        with np.load(path, allow_pickle=False) as payload:
            embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
            row_ids = payload["row_ids"].astype(str).tolist()
    except (KeyError, OSError, ValueError) as error:
        raise ExperimentError(f"Embedding cache cannot be read: {error}") from error
    if row_ids != expected_row_ids:
        raise ExperimentError("Embedding cache row order does not match current non-final data.")
    if embeddings.shape != (len(expected_row_ids), expected_dimension):
        raise ExperimentError(
            "Embedding cache shape does not match current rows and configured dimension."
        )
    if not np.isfinite(embeddings).all():
        raise ExperimentError("Embedding cache contains non-finite values.")
    return embeddings


def validate_cache_manifest(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    """Fail closed when cached embeddings were produced under different inputs."""
    if not path.is_file():
        raise ExperimentError(f"Embedding cache manifest not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExperimentError(f"Embedding cache manifest cannot be read: {error}") from error
    mismatches = [key for key, value in expected.items() if payload.get(key) != value]
    if mismatches:
        raise ExperimentError(
            f"Embedding cache manifest mismatch: {', '.join(sorted(mismatches))}"
        )
    return payload


def build_dense_model(name: str, config: dict[str, Any], seed: int):
    """Build an approved linear classifier for fixed dense embeddings."""
    estimator = config.get("estimator", name)
    if estimator == "logistic_regression":
        return LogisticRegression(
            C=float(config["C"]),
            max_iter=int(config["max_iter"]),
            class_weight=config.get("class_weight"),
            random_state=seed,
        )
    if estimator == "linear_svm":
        return LinearSVC(
            C=float(config["C"]),
            max_iter=int(config["max_iter"]),
            class_weight=config.get("class_weight"),
            random_state=seed,
        )
    raise ExperimentError(f"Unsupported contextual classifier: {estimator}")


def evaluate_contextual_embeddings(
    pool: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    label_column: str,
    fold_column: str,
    validation_folds: list[int],
    models: dict[str, dict[str, Any]],
    expected_labels: list[str],
    random_seed: int,
) -> ContextualResults:
    """Evaluate fixed embeddings across predeclared non-final grouped folds."""
    if embeddings.shape[0] != len(pool):
        raise ExperimentError("Embedding rows do not align with the evaluation pool.")
    if not models:
        raise ExperimentError("At least one contextual classifier is required.")
    fold_values = pool[fold_column].to_numpy(dtype=int)
    labels = pool[label_column].astype(str).to_numpy()
    fold_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []

    for model_name, model_config in models.items():
        for fold in validation_folds:
            train_mask = fold_values != fold
            validation_mask = fold_values == fold
            if not train_mask.any() or not validation_mask.any():
                raise ExperimentError(f"Fold {fold} has no train or validation rows.")
            model = build_dense_model(model_name, model_config, random_seed)
            started = perf_counter()
            model.fit(embeddings[train_mask], labels[train_mask])
            predictions = model.predict(embeddings[validation_mask])
            elapsed_seconds = perf_counter() - started
            validation_labels = labels[validation_mask]
            precision, recall, class_f1, support = precision_recall_fscore_support(
                validation_labels,
                predictions,
                labels=expected_labels,
                zero_division=0,
            )
            common = {
                "representation": "all_minilm_l6_v2",
                "model": model_name,
                "validation_fold": int(fold),
                "train_rows": int(train_mask.sum()),
                "validation_rows": int(validation_mask.sum()),
            }
            fold_rows.append(
                {
                    **common,
                    "macro_f1": float(
                        f1_score(
                            validation_labels,
                            predictions,
                            labels=expected_labels,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(validation_labels, predictions)
                    ),
                    "mcc": float(matthews_corrcoef(validation_labels, predictions)),
                }
            )
            runtime_rows.append(
                {
                    **common,
                    "elapsed_seconds": float(elapsed_seconds),
                    "embedding_dimension": int(embeddings.shape[1]),
                    "train_matrix_mib": float(embeddings[train_mask].nbytes / (1024**2)),
                    "validation_matrix_mib": float(
                        embeddings[validation_mask].nbytes / (1024**2)
                    ),
                }
            )
            class_rows.extend(
                {
                    **common,
                    "label": label,
                    "precision": float(label_precision),
                    "recall": float(label_recall),
                    "f1": float(label_f1),
                    "support": int(label_support),
                }
                for label, label_precision, label_recall, label_f1, label_support in zip(
                    expected_labels, precision, recall, class_f1, support, strict=True
                )
            )
            matrix = confusion_matrix(validation_labels, predictions, labels=expected_labels)
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
        ["model", "validation_fold"], ignore_index=True
    )
    per_class = pd.DataFrame(class_rows).sort_values(
        ["model", "label", "validation_fold"], ignore_index=True
    )
    runtime = pd.DataFrame(runtime_rows).sort_values(
        ["model", "validation_fold"], ignore_index=True
    )
    confusion = pd.DataFrame(confusion_rows).sort_values(
        ["model", "validation_fold", "true_label", "predicted_label"],
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
        .sort_values("macro_f1_mean", ascending=False, ignore_index=True)
    )
    per_class_summary = (
        per_class.groupby(["representation", "model", "label"], as_index=False)
        .agg(
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            support_total=("support", "sum"),
        )
        .sort_values(["model", "label"], ignore_index=True)
    )
    return ContextualResults(
        fold_metrics=fold_metrics,
        summary_metrics=summary,
        per_class_metrics=per_class,
        per_class_summary=per_class_summary,
        runtime_metrics=runtime,
        confusion_matrices=confusion,
    )
