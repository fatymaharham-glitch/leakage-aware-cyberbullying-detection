"""Reproduce classifier, imbalance, contextual, and leakage comparisons."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

from experiments.shared import (
    CV_FOLDS,
    FINAL_FOLD,
    LABELS,
    RANDOM_SEED,
    build_features,
    evaluate_candidate,
    metadata,
    strict_project_data,
    update_registry_metrics,
    write_json,
)

import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parent
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
CACHE = OUTPUT.parents[1] / "data/contextual_embeddings.npz"
CACHE_META = OUTPUT.parents[1] / "data/contextual_embeddings.json"


def compact(evaluation: object, *, include_predictions: bool = False) -> dict[str, object]:
    value = asdict(evaluation)
    if not include_predictions:
        value.pop("truth")
        value.pop("predictions")
        value.pop("confidence")
    return value


def row_order_hash(values: pd.Series) -> str:
    return hashlib.sha256("\n".join(values.astype(str)).encode()).hexdigest()


def contextual_embeddings(frame: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, dict[str, object]]:
    development = frame[frame["fold"] != FINAL_FOLD].reset_index(drop=True)
    expected = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "rows": int(len(development)),
        "row_order_sha256": row_order_hash(development["row_id"]),
        "dimension": 384,
        "normalized": True,
    }
    if CACHE.is_file() and CACHE_META.is_file():
        existing = json.loads(CACHE_META.read_text(encoding="utf-8"))
        if existing != expected:
            raise ValueError("Contextual embedding cache metadata does not match the protected input.")
        with np.load(CACHE) as stored:
            embeddings = stored["embeddings"]
        if embeddings.shape != (len(development), 384) or not np.isfinite(embeddings).all():
            raise ValueError("Contextual embedding cache has an invalid shape or non-finite values.")
        return embeddings, development, {**expected, "cache_reused": True}

    from sentence_transformers import SentenceTransformer

    started = time.perf_counter()
    encoder = SentenceTransformer(
        MODEL_ID,
        revision=MODEL_REVISION,
        device="cpu",
        trust_remote_code=False,
    )
    embeddings = encoder.encode(
        development["tweet_text"].map(lambda value: str(value)).tolist(),
        batch_size=64,
        device="cpu",
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    if embeddings.shape != (len(development), 384) or not np.isfinite(embeddings).all():
        raise ValueError("Contextual encoder produced an invalid matrix.")
    np.savez_compressed(CACHE, embeddings=embeddings)
    write_json(CACHE_META, expected)
    return embeddings, development, {**expected, "cache_reused": False, "embedding_seconds": time.perf_counter() - started}


def evaluate_dense(embeddings: np.ndarray, frame: pd.DataFrame) -> dict[str, object]:
    evidence: dict[str, object] = {}
    models = {
        "logistic_regression_c1_balanced": lambda: LogisticRegression(C=1, max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED),
        "linear_svm_c0_5": lambda: LinearSVC(C=0.5, max_iter=10000, random_state=RANDOM_SEED),
    }
    for model_id, factory in models.items():
        folds = []
        truth_parts = []
        prediction_parts = []
        per_class = []
        for fold in CV_FOLDS:
            train_mask = frame["fold"].to_numpy() != fold
            validation_mask = frame["fold"].to_numpy() == fold
            started = time.perf_counter()
            model = factory()
            model.fit(embeddings[train_mask], frame.loc[train_mask, "cyberbullying_type"])
            predictions = model.predict(embeddings[validation_mask])
            truth = frame.loc[validation_mask, "cyberbullying_type"].to_numpy()
            precision, recall, class_f1, support = precision_recall_fscore_support(truth, predictions, labels=LABELS, zero_division=0)
            folds.append({
                "fold": fold,
                "rows": int(validation_mask.sum()),
                "macro_f1": float(f1_score(truth, predictions, average="macro")),
                "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
                "mcc": float(matthews_corrcoef(truth, predictions)),
                "runtime_seconds": time.perf_counter() - started,
            })
            per_class.extend({"fold": fold, "class": label, "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(class_f1[i]), "support": int(support[i])} for i, label in enumerate(LABELS))
            truth_parts.extend(truth.tolist())
            prediction_parts.extend(predictions.tolist())
        evidence[model_id] = {
            "summary": {
                "macro_f1_mean": float(np.mean([row["macro_f1"] for row in folds])),
                "balanced_accuracy_mean": float(np.mean([row["balanced_accuracy"] for row in folds])),
                "mcc_mean": float(np.mean([row["mcc"] for row in folds])),
                "runtime_seconds": float(sum(row["runtime_seconds"] for row in folds)),
            },
            "folds": folds,
            "per_class": per_class,
            "labels": LABELS,
            "confusion_matrix": confusion_matrix(truth_parts, prediction_parts, labels=LABELS).astype(int).tolist(),
        }
    return evidence


def evaluate_oversampling(frame: pd.DataFrame) -> dict[str, object]:
    fold_rows = []
    truth_parts: list[str] = []
    prediction_parts: list[str] = []
    per_class = []
    for fold in CV_FOLDS:
        train = frame[(frame["fold"] != fold) & (frame["fold"] != FINAL_FOLD)].reset_index(drop=True)
        validation = frame[frame["fold"] == fold]
        random = np.random.default_rng(RANDOM_SEED + fold)
        largest = int(train["cyberbullying_type"].value_counts().max())
        sampled = np.concatenate([
            random.choice(indices, size=largest, replace=len(indices) < largest)
            for label in LABELS
            for indices in [train.index[train["cyberbullying_type"] == label].to_numpy()]
        ])
        representation = build_features()
        started = time.perf_counter()
        train_matrix = representation.fit_transform(train["tweet_text"])
        validation_matrix = representation.transform(validation["tweet_text"])
        model = LogisticRegression(C=1, max_iter=1000, random_state=RANDOM_SEED)
        model.fit(train_matrix[sampled], train.loc[sampled, "cyberbullying_type"])
        predictions = model.predict(validation_matrix)
        truth = validation["cyberbullying_type"].to_numpy()
        precision, recall, class_f1, support = precision_recall_fscore_support(truth, predictions, labels=LABELS, zero_division=0)
        fold_rows.append({"fold": fold, "macro_f1": float(f1_score(truth, predictions, average="macro")), "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)), "mcc": float(matthews_corrcoef(truth, predictions)), "runtime_seconds": time.perf_counter() - started})
        per_class.extend({"fold": fold, "class": label, "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(class_f1[i]), "support": int(support[i])} for i, label in enumerate(LABELS))
        truth_parts.extend(truth.tolist())
        prediction_parts.extend(predictions.tolist())
    return {
        "summary": {"macro_f1_mean": float(np.mean([row["macro_f1"] for row in fold_rows])), "balanced_accuracy_mean": float(np.mean([row["balanced_accuracy"] for row in fold_rows])), "mcc_mean": float(np.mean([row["mcc"] for row in fold_rows])), "runtime_seconds": float(sum(row["runtime_seconds"] for row in fold_rows))},
        "folds": fold_rows,
        "per_class": per_class,
        "labels": LABELS,
        "confusion_matrix": confusion_matrix(truth_parts, prediction_parts, labels=LABELS).astype(int).tolist(),
    }


def random_folds(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    output = frame.copy()
    output["random_fold"] = FINAL_FOLD
    development = output[output["fold"] != FINAL_FOLD]
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_SEED)
    for fold, (_, validation_positions) in zip(CV_FOLDS, splitter.split(development, development["cyberbullying_type"]), strict=True):
        output.loc[development.index[validation_positions], "random_fold"] = fold
    crossing = output[output["fold"] != FINAL_FOLD].groupby("near_duplicate_group_id")["random_fold"].nunique()
    return output, int((crossing > 1).sum())


def main() -> None:
    frame = strict_project_data()
    classifier_ids = [
        "multinomial_nb",
        "logistic_regression_c0_5",
        "logistic_regression_c1",
        "logistic_regression_c2",
        "logistic_regression_c1_balanced",
        "linear_svm_c0_5",
        "linear_svm_c1",
        "linear_svm_c2",
        "linear_svm_c1_balanced",
        "random_forest",
        "xgboost",
    ]
    classifiers = {model_id: compact(evaluate_candidate(frame, model_id)) for model_id in classifier_ids}
    imbalance = {
        "logistic_regression_unweighted": classifiers["logistic_regression_c1"],
        "logistic_regression_balanced": classifiers["logistic_regression_c1_balanced"],
        "logistic_regression_random_oversampling": evaluate_oversampling(frame),
    }
    randomized, groups_crossing = random_folds(frame)
    leakage = {
        "grouped": classifiers["logistic_regression_c1_balanced"],
        "random_stratified": compact(evaluate_candidate(randomized, "logistic_regression_c1_balanced", fold_column="random_fold")),
        "groups_crossing_grouped_folds": 0,
        "groups_crossing_random_folds": groups_crossing,
    }
    embeddings, contextual_frame, contextual_manifest = contextual_embeddings(frame)
    contextual = evaluate_dense(embeddings, contextual_frame)
    primary = "logistic_regression_c1_balanced"
    primary_score = classifiers[primary]["summary"]["macro_f1_mean"]
    eligible_backup = [model_id for model_id in classifier_ids if model_id.startswith("linear_svm") and primary_score - classifiers[model_id]["summary"]["macro_f1_mean"] <= 0.005]
    backup = max(eligible_backup, key=lambda model_id: classifiers[model_id]["summary"]["macro_f1_mean"])
    headline_ids = ["multinomial_nb", primary, backup, "random_forest", "xgboost"]
    headline = {model_id: classifiers[model_id]["summary"]["macro_f1_mean"] for model_id in headline_ids}
    update_registry_metrics(headline)
    results = {
        "experiment": 3,
        "title": "Model comparison and candidate selection",
        "question": "Which bounded model is strongest without using protected fold 0?",
        "choices_tried": {
            "classifiers": classifier_ids,
            "imbalance": list(imbalance),
            "contextual": [MODEL_ID, "Logistic Regression", "Linear SVM"],
            "leakage_check": ["grouped", "random stratified"],
        },
        "solution": "Freeze balanced Logistic Regression as primary and Linear SVM C=0.5 as a different-family backup; retain grouped evaluation as the credible estimate.",
        "results": {
            "headline_models": headline,
            "classifiers": classifiers,
            "imbalance": imbalance,
            "contextual": contextual,
            "contextual_manifest": contextual_manifest,
            "leakage": leakage,
            "selection": {"primary": primary, "backup": backup, "macro_f1_tolerance": 0.005, "used_final_test": False},
        },
        "limitations": [
            "Tree models are bounded cost baselines on high-dimensional sparse text, not exhaustive searches.",
            "MiniLM is frozen, English-oriented, truncated at 256 wordpieces, and evaluated only as a contextual baseline.",
        ],
        "metadata": metadata(),
    }
    write_json(OUTPUT / "results.json", results)

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    pd.Series(headline).sort_values().plot.barh(ax=axes[0, 0], title="Five classifier families", color="#2368a2")
    pd.Series({key: row["summary"]["macro_f1_mean"] for key, row in classifiers.items()}).sort_values().plot.barh(ax=axes[0, 1], title="Bounded linear variants", color="#18745a")
    pd.Series({key: row["summary"]["macro_f1_mean"] for key, row in imbalance.items()}).plot.bar(ax=axes[1, 0], title="Imbalance treatments", color="#9a5b0b")
    controls = {"grouped": leakage["grouped"]["summary"]["macro_f1_mean"], "random": leakage["random_stratified"]["summary"]["macro_f1_mean"], **{f"MiniLM {key}": row["summary"]["macro_f1_mean"] for key, row in contextual.items()}}
    pd.Series(controls).plot.bar(ax=axes[1, 1], title="Leakage and contextual controls", color="#a23b42")
    for axis in axes.flat:
        axis.set_xlabel("Macro-F1")
        axis.grid(axis="y", alpha=0.2)
        axis.tick_params(axis="x", rotation=25)
    figure.suptitle("Experiment 3 — model comparison", fontsize=16)
    figure.tight_layout()
    figure.savefig(OUTPUT / "plot.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
