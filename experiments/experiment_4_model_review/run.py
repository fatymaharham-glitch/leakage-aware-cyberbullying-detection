"""Reproduce robustness, external, confidence, explanation, and error evidence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from experiments.shared import (
    CV_FOLDS,
    EXTERNAL_DATA,
    FINAL_FOLD,
    LABELS,
    RANDOM_SEED,
    build_final_model,
    leetspeak,
    metadata,
    normalize_repeats,
    partial_masking,
    remove_punctuation,
    strict_project_data,
    write_json,
)

import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parent
FINALISTS = {
    "primary_logistic_regression": "logistic_regression_c1_balanced",
    "backup_linear_svm": "linear_svm_c0_5",
}
CHANGES = {
    "clean": str,
    "remove_punctuation": remove_punctuation,
    "normalize_repeats": normalize_repeats,
    "leetspeak": leetspeak,
    "partial_masking": partial_masking,
}


def confidence_scores(model: object, values: pd.Series) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(values)).max(axis=1)
    return softmax(np.asarray(model.decision_function(values)), axis=1).max(axis=1)


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray) -> float:
    value = 0.0
    for low in np.linspace(0, 0.9, 10):
        members = (confidence >= low) & (confidence < low + 0.1)
        if members.any():
            value += float(members.mean() * abs(correct[members].mean() - confidence[members].mean()))
    return value


def bootstrap_difference(truth: np.ndarray, primary: np.ndarray, backup: np.ndarray) -> dict[str, object]:
    random = np.random.default_rng(RANDOM_SEED)
    observed = f1_score(truth, primary, average="macro") - f1_score(truth, backup, average="macro")
    differences = np.empty(1000)
    label_index = {label: index for index, label in enumerate(LABELS)}
    truth_codes = np.fromiter((label_index[value] for value in truth), dtype=np.int8)
    primary_codes = np.fromiter((label_index[value] for value in primary), dtype=np.int8)
    backup_codes = np.fromiter((label_index[value] for value in backup), dtype=np.int8)

    def macro_f1(codes: np.ndarray, predicted: np.ndarray) -> float:
        matrix = np.bincount(codes * len(LABELS) + predicted, minlength=len(LABELS) ** 2).reshape(len(LABELS), len(LABELS))
        true_positive = np.diag(matrix)
        denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
        return float(np.divide(2 * true_positive, denominator, out=np.zeros_like(true_positive, dtype=float), where=denominator != 0).mean())

    for index in range(len(differences)):
        sample = random.integers(0, len(truth), size=len(truth))
        differences[index] = macro_f1(truth_codes[sample], primary_codes[sample]) - macro_f1(truth_codes[sample], backup_codes[sample])
    low, high = np.quantile(differences, [0.025, 0.975])
    return {"difference": float(observed), "low": float(low), "high": float(high), "samples": 1000, "seed": RANDOM_SEED, "unit": "out-of-fold row"}


def main() -> None:
    frame = strict_project_data()
    development = frame[frame["fold"] != FINAL_FOLD]
    truths: list[np.ndarray] = []
    predictions: dict[str, dict[str, list[np.ndarray]]] = {
        finalist: {condition: [] for condition in CHANGES} for finalist in FINALISTS
    }
    confidence: dict[str, list[np.ndarray]] = {finalist: [] for finalist in FINALISTS}
    for fold in CV_FOLDS:
        train = development[development["fold"] != fold]
        validation = development[development["fold"] == fold]
        truths.append(validation["cyberbullying_type"].to_numpy())
        for finalist, model_id in FINALISTS.items():
            model = build_final_model(model_id)
            model.fit(train["tweet_text"], train["cyberbullying_type"])
            for condition, change in CHANGES.items():
                predictions[finalist][condition].append(model.predict(validation["tweet_text"].map(change)))
            confidence[finalist].append(confidence_scores(model, validation["tweet_text"]))
    truth = np.concatenate(truths)
    combined_predictions = {
        finalist: {condition: np.concatenate(parts) for condition, parts in conditions.items()}
        for finalist, conditions in predictions.items()
    }
    robustness = {
        finalist: {
            condition: {
                "macro_f1": float(f1_score(truth, values, average="macro", zero_division=0)),
                "retention": float(f1_score(truth, values, average="macro", zero_division=0) / f1_score(truth, combined_predictions[finalist]["clean"], average="macro", zero_division=0)),
            }
            for condition, values in conditions.items()
        }
        for finalist, conditions in combined_predictions.items()
    }

    if not EXTERNAL_DATA.is_file():
        raise FileNotFoundError("Tracked data/external_validation.csv is required for Experiment 4.")
    external = pd.read_csv(EXTERNAL_DATA)
    true_harmful = external["class"].to_numpy() != 2
    full_models = {}
    external_results = {}
    for finalist, model_id in FINALISTS.items():
        model = build_final_model(model_id)
        model.fit(development["tweet_text"], development["cyberbullying_type"])
        full_models[finalist] = model
        predicted_harmful = model.predict(external["tweet"]) != "not_cyberbullying"
        matrix = confusion_matrix(true_harmful, predicted_harmful, labels=[False, True])
        external_results[finalist] = {
            "rows": int(len(external)),
            "accuracy": float(accuracy_score(true_harmful, predicted_harmful)),
            "binary_macro_f1": float(f1_score(true_harmful, predicted_harmful, average="macro")),
            "harmful_precision": float(precision_score(true_harmful, predicted_harmful)),
            "harmful_recall": float(recall_score(true_harmful, predicted_harmful)),
            "false_positive_rate": float(matrix[0, 1] / matrix[0].sum()),
            "confusion_matrix_false_true": matrix.astype(int).tolist(),
        }

    primary_predictions = combined_predictions["primary_logistic_regression"]["clean"]
    primary_confidence = np.concatenate(confidence["primary_logistic_regression"])
    correct = primary_predictions == truth
    referral_curve = []
    for threshold in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70):
        kept = primary_confidence >= threshold
        referral_curve.append({
            "threshold": threshold,
            "coverage": float(kept.mean()),
            "referred_rate": float((~kept).mean()),
            "accuracy_when_kept": float(accuracy_score(truth[kept], primary_predictions[kept])) if kept.any() else None,
            "macro_f1_when_kept": float(f1_score(truth[kept], primary_predictions[kept], average="macro")) if kept.any() else None,
        })
    confidence_result = {
        "selected_threshold": 0.45,
        "expected_calibration_error": expected_calibration_error(primary_confidence, correct),
        "mean_confidence": float(primary_confidence.mean()),
        "curve": referral_curve,
    }

    bootstrap = bootstrap_difference(
        truth,
        primary_predictions,
        combined_predictions["backup_linear_svm"]["clean"],
    )
    identity_terms = {
        "age": ["young", "old"],
        "ethnicity": ["Asian", "European"],
        "gender": ["woman", "man"],
        "religion": ["Muslim", "Christian"],
    }
    identity_probes = []
    for finalist, model in full_models.items():
        for category, terms in identity_terms.items():
            for term in terms:
                sentence = pd.Series([f"This person is {term}."])
                predicted = str(model.predict(sentence)[0])
                score = float(confidence_scores(model, sentence)[0])
                identity_probes.append({"model": finalist, "category": category, "term": term, "prediction": predicted, "confidence": score})

    primary_model = full_models["primary_logistic_regression"]
    feature_names = primary_model.named_steps["representation"].get_feature_names_out()
    masked_features = []
    for class_name, coefficients in zip(primary_model.named_steps["model"].classes_, primary_model.named_steps["model"].coef_, strict=True):
        for rank, index in enumerate(np.argsort(coefficients)[-5:][::-1], 1):
            prefix, token = str(feature_names[index]).split("__", 1)
            masked = token[:1] + "*" * max(2, len(token) - 1)
            masked_features.append({"class": str(class_name), "rank": rank, "masked_feature": f"{prefix}__{masked}", "coefficient": float(coefficients[index])})

    errors = {}
    for finalist, conditions in combined_predictions.items():
        matrix = confusion_matrix(truth, conditions["clean"], labels=LABELS)
        pairs = [
            {"true": true_label, "predicted": predicted_label, "count": int(matrix[true_index, predicted_index])}
            for true_index, true_label in enumerate(LABELS)
            for predicted_index, predicted_label in enumerate(LABELS)
            if true_index != predicted_index
        ]
        errors[finalist] = sorted(pairs, key=lambda row: row["count"], reverse=True)

    results = {
        "experiment": 4,
        "title": "Robustness and model review",
        "question": "Where do both finalists fail, and how should uncertain predictions be handled?",
        "choices_tried": {
            "finalists": list(FINALISTS),
            "robustness_changes": list(CHANGES),
            "outside_dataset": "Davidson hate/offensive-language Twitter data",
            "referral_thresholds": [row["threshold"] for row in referral_curve],
            "review": ["paired bootstrap", "masked coefficients", "neutral identity probes", "aggregate error pairs"],
        },
        "solution": "Use the simpler probabilistic primary with a 0.45 referral threshold, retain the backup for comparison, and require human review because transfer and obfuscation failures remain substantial.",
        "results": {
            "robustness": robustness,
            "external": external_results,
            "confidence": confidence_result,
            "bootstrap_primary_minus_backup": bootstrap,
            "identity_probes": identity_probes,
            "masked_explanations": masked_features,
            "aggregate_errors": errors,
        },
        "limitations": [
            "External labels are mapped only to harmful/not harmful and cannot validate the six-class task.",
            "Identity probes are small sensitivity diagnostics, not a fairness audit.",
            "Aggregate errors avoid redistributing raw tweets; optional human qualitative coding was not performed.",
            "Linear coefficients describe model associations, not causal explanations.",
        ],
        "metadata": metadata(),
    }
    write_json(OUTPUT / "results.json", results)

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    robustness_frame = pd.DataFrame({finalist: {condition: values["macro_f1"] for condition, values in conditions.items()} for finalist, conditions in robustness.items()})
    robustness_frame.plot.bar(ax=axes[0, 0], title="Robustness changes")
    pd.DataFrame(external_results).T[["accuracy", "binary_macro_f1", "false_positive_rate"]].plot.bar(ax=axes[0, 1], title="External transfer")
    pd.DataFrame(referral_curve).plot(x="threshold", y=["coverage", "accuracy_when_kept"], marker="o", ax=axes[1, 0], title="Confidence referral")
    top_errors = pd.Series({f"{row['true']}→{row['predicted']}": row["count"] for row in errors["primary_logistic_regression"][:6]})
    top_errors.plot.barh(ax=axes[1, 1], title="Largest primary-model errors", color="#a23b42")
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
        axis.tick_params(axis="x", rotation=25)
    figure.suptitle("Experiment 4 — robustness and review", fontsize=16)
    figure.tight_layout()
    figure.savefig(OUTPUT / "plot.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
