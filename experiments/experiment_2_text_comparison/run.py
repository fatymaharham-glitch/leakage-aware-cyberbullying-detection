"""Reproduce P0/P1/P2 and sparse text-representation comparisons."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from experiments.shared import evaluate_candidate, metadata, strict_project_data, write_json

import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parent


def compact(evaluation: object) -> dict[str, object]:
    value = asdict(evaluation)
    value.pop("truth")
    value.pop("predictions")
    value.pop("confidence")
    return value


def main() -> None:
    frame = strict_project_data()
    preprocessing: dict[str, dict[str, object]] = {}
    for cleaner in ("p0", "p1", "p2"):
        for model in ("logistic_regression_c1", "linear_svm_c1"):
            key = f"{cleaner}__{model}"
            preprocessing[key] = compact(
                evaluate_candidate(
                    frame,
                    model,
                    representation="word",
                    preprocessing=cleaner,
                    word_max_features=100_000,
                )
            )
    representations: dict[str, dict[str, object]] = {}
    for representation in ("word", "character", "combined"):
        for model in ("logistic_regression_c1", "linear_svm_c1"):
            key = f"{representation}__{model}"
            representations[key] = compact(
                evaluate_candidate(
                    frame,
                    model,
                    representation=representation,
                    preprocessing="p1",
                    word_max_features=80_000 if representation == "word" else 60_000,
                    character_max_features=80_000 if representation == "character" else 60_000,
                )
            )
    preprocessing_winner = max(preprocessing, key=lambda key: preprocessing[key]["summary"]["macro_f1_mean"])
    representation_winner = max(representations, key=lambda key: representations[key]["summary"]["macro_f1_mean"])
    results = {
        "experiment": 2,
        "title": "Text cleaning and representation comparison",
        "question": "Which text cleaning and TF-IDF representation work best without touching fold 0?",
        "choices_tried": {
            "preprocessing": ["P0 minimal", "P1 social-media-aware", "P2 aggressive"],
            "representations": ["word TF-IDF", "character TF-IDF", "combined word + character TF-IDF"],
            "classifiers": ["Logistic Regression C=1", "Linear SVM C=1"],
        },
        "solution": "Use P1 social-media-aware preprocessing and combined word/character TF-IDF, selected only from grouped folds 1–4.",
        "results": {
            "preprocessing": preprocessing,
            "representations": representations,
            "selected_preprocessing_candidate": preprocessing_winner,
            "selected_representation_candidate": representation_winner,
        },
        "limitations": [
            "English stop words and the source dataset language mix may disadvantage non-English text.",
            "Vocabulary sizes are bounded; larger searches were deliberately not performed after seeing the protected test.",
        ],
        "metadata": metadata(),
    }
    write_json(OUTPUT / "results.json", results)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for axis, title, evidence in (
        (axes[0], "P0/P1/P2 word TF-IDF", preprocessing),
        (axes[1], "Word vs character vs combined", representations),
    ):
        values = pd.Series({key: row["summary"]["macro_f1_mean"] for key, row in evidence.items()}).sort_values()
        values.plot.barh(ax=axis, color="#2368a2")
        axis.set_xlim(0.8, 0.88)
        axis.set_xlabel("Grouped macro-F1")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle("Experiment 2 — text comparison", fontsize=16)
    figure.tight_layout()
    figure.savefig(OUTPUT / "plot.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
