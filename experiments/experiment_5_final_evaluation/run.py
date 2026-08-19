"""Reproduce the frozen final evaluation and export its fitted pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_recall_fscore_support

from experiments.shared import LABELS, build_final_model, metadata, save_final_model, strict_project_data, write_json

import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parent
THRESHOLD = 0.45
TOLERANCE = 1e-6


def prior_evidence() -> tuple[list[dict[str, object]], dict[str, float] | None]:
    path = OUTPUT / "results.json"
    if not path.is_file():
        return [], None
    previous = json.loads(path.read_text(encoding="utf-8"))
    history = previous.get("access_history")
    if history is None and "access_record" in previous:
        history = [{**previous["access_record"], "kind": "first_access"}]
    metrics = previous.get("results", {}).get("metrics")
    return list(history or []), metrics


def main() -> None:
    history, prior_metrics = prior_evidence()
    frame = strict_project_data()
    train = frame[frame["fold"] != 0]
    final_test = frame[frame["fold"] == 0]
    model = build_final_model()
    model.fit(train["tweet_text"], train["cyberbullying_type"])
    predictions = model.predict(final_test["tweet_text"])
    probabilities = model.predict_proba(final_test["tweet_text"])
    confidence = probabilities.max(axis=1)
    precision, recall, class_f1, support = precision_recall_fscore_support(
        final_test["cyberbullying_type"], predictions, labels=LABELS, zero_division=0
    )
    matrix = confusion_matrix(final_test["cyberbullying_type"], predictions, labels=LABELS)
    metrics = {
        "rows": int(len(final_test)),
        "macro_f1": float(f1_score(final_test["cyberbullying_type"], predictions, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(final_test["cyberbullying_type"], predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(final_test["cyberbullying_type"], predictions)),
        "mcc": float(matthews_corrcoef(final_test["cyberbullying_type"], predictions)),
        "mean_confidence": float(confidence.mean()),
        "referral_threshold": THRESHOLD,
        "referred_rate": float((confidence < THRESHOLD).mean()),
    }
    comparison = None
    if prior_metrics:
        comparison = {
            key: abs(float(metrics[key]) - float(prior_metrics[key]))
            for key in ("macro_f1", "accuracy", "balanced_accuracy", "mcc")
        }
        if max(comparison.values()) > TOLERANCE:
            raise RuntimeError(f"Final reproduction differs from prior verified evidence beyond {TOLERANCE}: {comparison}")
    now = datetime.now(timezone.utc).isoformat()
    if not history:
        history.append({"evaluated_at_utc": now, "kind": "first_access", "selection_used_final_test": False})
    else:
        history.append({"evaluated_at_utc": now, "kind": "reproduction", "selection_used_final_test": False})
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(class_f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(LABELS)
    }
    result = {
        "experiment": 5,
        "title": "Protected final evaluation and model export",
        "question": "How does the frozen primary perform on data never used for selection?",
        "choices_tried": {
            "model": "balanced Logistic Regression C=1.0 (frozen before access)",
            "features": "P1 combined word and character TF-IDF",
            "training_folds": [1, 2, 3, 4],
            "protected_test_fold": 0,
            "human_review_threshold": THRESHOLD,
        },
        "solution": "Fit the frozen primary once on all development folds, evaluate the exact protected fold-0 membership, record access history, and export that same pipeline.",
        "results": {
            "metrics": metrics,
            "per_class": per_class,
            "class_order": LABELS,
            "confusion_matrix": matrix.astype(int).tolist(),
            "confidence_distribution": {
                "minimum": float(confidence.min()),
                "q25": float(np.quantile(confidence, 0.25)),
                "median": float(np.median(confidence)),
                "q75": float(np.quantile(confidence, 0.75)),
                "maximum": float(confidence.max()),
            },
            "prior_reproduction_absolute_differences": comparison,
        },
        "access_history": history,
        "limitations": [
            "This is a tweet-level research prototype, not proof of intent, repetition, power imbalance, or real-world harm.",
            "Later executions are reproductions of the first protected-fold access, not independent first evaluations.",
        ],
        "metadata": metadata(),
    }
    write_json(OUTPUT / "results.json", result)
    save_final_model(model, THRESHOLD)

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    pd.Series({label: values["f1"] for label, values in per_class.items()}).plot.bar(ax=axes[0, 0], title="Per-class F1", color="#2368a2")
    image = axes[0, 1].imshow(matrix, cmap="Blues")
    axes[0, 1].set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
    axes[0, 1].set_yticks(range(len(LABELS)), LABELS)
    axes[0, 1].set_title("Confusion matrix")
    figure.colorbar(image, ax=axes[0, 1], fraction=0.046)
    axes[1, 0].hist(confidence, bins=20, color="#18745a")
    axes[1, 0].axvline(THRESHOLD, color="#a23b42", linestyle="--", label="human-review threshold")
    axes[1, 0].legend()
    axes[1, 0].set_title("Prediction confidence")
    pd.Series({key: metrics[key] for key in ("macro_f1", "accuracy", "balanced_accuracy", "mcc")}).plot.bar(ax=axes[1, 1], color="#9a5b0b", title="Final headline metrics")
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
        axis.tick_params(axis="x", rotation=30)
    figure.suptitle("Experiment 5 — protected final evaluation", fontsize=16)
    figure.tight_layout()
    figure.savefig(OUTPUT / "plot.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
