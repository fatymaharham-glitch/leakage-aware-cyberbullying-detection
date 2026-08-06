"""Build and freeze the P8 two-candidate shortlist from P4-P7 evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _candidate_row(
    *,
    candidate_id: str,
    family: str,
    representation: str,
    summary: pd.DataFrame,
    per_class: pd.DataFrame,
    runtime: pd.DataFrame,
    model: str,
    interpretability: str,
    embedding_seconds: float = 0.0,
) -> dict[str, object]:
    metric_rows = summary.loc[summary["model"] == model]
    class_rows = per_class.loc[per_class["model"] == model]
    runtime_rows = runtime.loc[runtime["model"] == model]
    if len(metric_rows) != 1 or class_rows.empty or runtime_rows.empty:
        raise ValueError(f"Incomplete evidence for candidate: {candidate_id}")
    metrics = metric_rows.iloc[0]
    weakest = class_rows.sort_values("f1_mean").iloc[0]
    memory_column = "train_matrix_mib"
    return {
        "candidate_id": candidate_id,
        "family": family,
        "representation": representation,
        "macro_f1_mean": float(metrics["macro_f1_mean"]),
        "macro_f1_std": float(metrics["macro_f1_std"]),
        "balanced_accuracy_mean": float(metrics["balanced_accuracy_mean"]),
        "mcc_mean": float(metrics["mcc_mean"]),
        "weakest_class": str(weakest["label"]),
        "weakest_class_recall": float(weakest["recall_mean"]),
        "weakest_class_f1": float(weakest["f1_mean"]),
        "runtime_seconds_mean": float(runtime_rows["elapsed_seconds"].mean()),
        "one_time_embedding_seconds": float(embedding_seconds),
        "train_matrix_mib_mean": float(runtime_rows[memory_column].mean()),
        "interpretability": interpretability,
    }


def main() -> int:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from cyberbullying_detection.data.audit import sha256_file
    from cyberbullying_detection.experiments.candidate_selection import select_candidates
    from cyberbullying_detection.utils.io import DataFileError, write_csv, write_json, write_text
    from cyberbullying_detection.utils.logging import get_logger

    logger = get_logger("candidate_selection")
    args = parse_args()
    output_directory = PROJECT_ROOT / "reports/model_results/candidate_selection"
    try:
        classifier = PROJECT_ROOT / "reports/model_results/classifier_benchmark"
        imbalance = PROJECT_ROOT / "reports/model_results/imbalance_analysis"
        contextual = PROJECT_ROOT / "reports/model_results/contextual_baseline"
        leakage = PROJECT_ROOT / "reports/model_results/leakage_gap"
        p4_summary = pd.read_csv(classifier / "summary_metrics.csv")
        p4_classes = pd.read_csv(classifier / "per_class_summary.csv")
        p4_runtime = pd.read_csv(classifier / "runtime_metrics.csv")
        p6_summary = pd.read_csv(imbalance / "summary_metrics.csv")
        p6_classes = pd.read_csv(imbalance / "per_class_summary.csv")
        p6_runtime = pd.read_csv(imbalance / "runtime_metrics.csv")
        p5_summary = pd.read_csv(contextual / "summary_metrics.csv")
        p5_classes = pd.read_csv(contextual / "per_class_summary.csv")
        p5_runtime = pd.read_csv(contextual / "runtime_metrics.csv")
        contextual_manifest = json.loads(
            (contextual / "experiment_manifest.json").read_text(encoding="utf-8")
        )
        rows = [
            _candidate_row(
                candidate_id="primary_sparse_logistic_regression",
                family="logistic_regression",
                representation="combined_tfidf",
                summary=p6_summary,
                per_class=p6_classes,
                runtime=p6_runtime,
                model="logistic_regression_balanced",
                interpretability="high_linear_feature_coefficients",
            ),
            _candidate_row(
                candidate_id="sparse_logistic_regression_unweighted",
                family="logistic_regression",
                representation="combined_tfidf",
                summary=p6_summary,
                per_class=p6_classes,
                runtime=p6_runtime,
                model="logistic_regression_unweighted",
                interpretability="high_linear_feature_coefficients",
            ),
            _candidate_row(
                candidate_id="sparse_logistic_regression_oversampled",
                family="logistic_regression",
                representation="combined_tfidf",
                summary=p6_summary,
                per_class=p6_classes,
                runtime=p6_runtime,
                model="logistic_regression_random_oversampling",
                interpretability="high_linear_feature_coefficients",
            ),
            _candidate_row(
                candidate_id="secondary_sparse_linear_svm",
                family="linear_svm",
                representation="combined_tfidf",
                summary=p4_summary,
                per_class=p4_classes,
                runtime=p4_runtime,
                model="linear_svm_c0_5",
                interpretability="high_linear_feature_coefficients",
            ),
            _candidate_row(
                candidate_id="contextual_linear_svm",
                family="linear_svm",
                representation="all_minilm_l6_v2",
                summary=p5_summary,
                per_class=p5_classes,
                runtime=p5_runtime,
                model="linear_svm_c0_5",
                interpretability="limited_embedding_dimensions",
                embedding_seconds=float(
                    contextual_manifest.get("embedding_seconds")
                    or json.loads(
                        (PROJECT_ROOT / "data/processed/contextual_embeddings_non_final_manifest.json").read_text(
                            encoding="utf-8"
                        )
                    )["embedding_seconds"]
                ),
            ),
        ]
        matrix = pd.DataFrame(rows)
        selected = select_candidates(matrix)
        expected = ["primary_sparse_logistic_regression", "secondary_sparse_linear_svm"]
        if selected != expected:
            raise ValueError(f"Frozen config does not match evidence-selected candidates: {selected}")
        matrix["selected_at_p8"] = matrix["candidate_id"].isin(selected)
        leakage_gap = pd.read_csv(leakage / "credibility_gap.csv")
        macro_gap = float(
            leakage_gap.loc[
                leakage_gap["metric"] == "macro_f1_mean", "random_minus_grouped"
            ].iloc[0]
        )
        output_directory.mkdir(parents=True, exist_ok=True)
        write_csv(matrix, output_directory / "selection_matrix.csv", overwrite=args.overwrite)
        frozen_config = PROJECT_ROOT / "configs/final_candidates.yaml"
        manifest = {
            "frozen_candidates_config": str(frozen_config.relative_to(PROJECT_ROOT)),
            "frozen_candidates_sha256": sha256_file(frozen_config),
            "selected_candidate_ids": selected,
            "selection_macro_f1_tolerance": 0.005,
            "random_minus_grouped_macro_f1": macro_gap,
            "final_test_fold": 0,
            "final_test_evaluated": False,
            "evidence_sha256": {
                str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
                for path in [
                    classifier / "summary_metrics.csv",
                    contextual / "summary_metrics.csv",
                    imbalance / "summary_metrics.csv",
                    leakage / "credibility_gap.csv",
                ]
            },
            "contains_predictions": False,
            "contains_raw_tweet_text": False,
        }
        write_json(manifest, output_directory / "selection_manifest.json", overwrite=args.overwrite)
        primary = matrix.loc[matrix["candidate_id"] == selected[0]].iloc[0]
        secondary = matrix.loc[matrix["candidate_id"] == selected[1]].iloc[0]
        record = (
            "# P8 candidate freeze\n\n"
            "Selected using grouped folds 1-4 only. Frozen fold 0 was not evaluated.\n\n"
            f"1. **Balanced Logistic Regression + combined TF-IDF** — primary; grouped macro-F1 "
            f"{primary['macro_f1_mean']:.4f}, best overall and strongest bounded imbalance treatment.\n"
            f"2. **Linear SVM (C=0.5) + combined TF-IDF** — secondary; grouped macro-F1 "
            f"{secondary['macro_f1_mean']:.4f}, competitive different-family comparator with lower fit time.\n\n"
            f"Random validation changed the primary score by {macro_gap:+.4f} versus grouped validation; "
            "grouped evidence controls selection. The contextual baseline was not retained because its "
            "macro-F1 was materially lower and its embedding dimensions are less directly interpretable. "
            "These are P8 candidates, not final-test winners; P9-P15 can still reject either before P16.\n"
        )
        write_text(record, output_directory / "decision_record.md", overwrite=args.overwrite)
    except (DataFileError, FileNotFoundError, KeyError, TypeError, ValueError) as error:
        logger.error("Candidate selection not completed: %s", error)
        return 2
    logger.info("P8 candidates frozen in %s", output_directory)
    logger.info("Frozen fold 0 was not evaluated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
