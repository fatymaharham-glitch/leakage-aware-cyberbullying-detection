"""Generate frozen sentence embeddings and run the P5 grouped baseline."""

from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="contextual_baseline.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def main() -> int:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "data/interim/.matplotlib"))
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "data/processed/huggingface"))
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from cyberbullying_detection.config import ConfigurationError, load_yaml_config, resolve_project_path
    from cyberbullying_detection.data.audit import sha256_file
    from cyberbullying_detection.data.validation import DataValidationError, validate_dataframe
    from cyberbullying_detection.experiments.contextual_baseline import (
        ContextualResults,
        ExperimentError,
        evaluate_contextual_embeddings,
        load_embedding_cache,
        prepare_non_final_pool,
        row_order_sha256,
        validate_cache_manifest,
        write_embedding_cache,
    )
    from cyberbullying_detection.experiments.sparse_benchmark import write_summary_plot
    from cyberbullying_detection.experiments.tfidf_baseline import PREPROCESSORS
    from cyberbullying_detection.utils.io import DataFileError, read_tabular_data, write_csv, write_json, write_text
    from cyberbullying_detection.utils.logging import get_logger

    logger = get_logger("contextual_baseline")
    args = parse_args()
    try:
        config = load_yaml_config(args.config)
        preprocessing_config = load_yaml_config("preprocessing.yaml")
        input_path = resolve_project_path(config["input_data"])
        folds_path = resolve_project_path(config["fold_assignments"])
        output_directory = resolve_project_path(config["outputs"]["directory"])
        cache_path = resolve_project_path(config["cache"]["embeddings"])
        cache_manifest_path = resolve_project_path(config["cache"]["manifest"])
        model_directory = resolve_project_path(config["cache"]["model_directory"])
        if not output_directory.is_relative_to(resolve_project_path("reports")):
            raise DataFileError("Contextual results must be written inside reports/.")

        frame = read_tabular_data(input_path)
        assignments = read_tabular_data(folds_path)
        validate_dataframe(
            frame,
            text_column=config["text_column"],
            label_column=config["label_column"],
            expected_labels=preprocessing_config["expected_labels"],
            require_all_labels=True,
        )
        validation_folds = [int(value) for value in config["cross_validation_folds"]]
        final_test_fold = int(config["final_test_fold"])
        pool = prepare_non_final_pool(
            frame,
            assignments,
            row_id_column=config["row_id_column"],
            label_column=config["label_column"],
            fold_column=config["fold_column"],
            final_test_fold=final_test_fold,
            validation_folds=validation_folds,
        )
        row_ids = pool[config["row_id_column"]].astype(str).tolist()
        expected_dimension = int(config["embedding_model"]["embedding_dimension"])
        cache_expectations = {
            "input_sha256": sha256_file(input_path),
            "fold_assignments_sha256": sha256_file(folds_path),
            "row_order_sha256": row_order_sha256(row_ids),
            "row_count": len(row_ids),
            "excluded_final_test_fold": final_test_fold,
            "preprocessing": config["preprocessing"],
            "embedding_model": config["embedding_model"],
        }
        embedding_seconds = 0.0
        cache_reused = cache_path.is_file() and not args.rebuild_cache
        if cache_reused:
            cache_manifest = validate_cache_manifest(
                cache_manifest_path,
                cache_expectations,
            )
            embeddings = load_embedding_cache(
                cache_path,
                expected_row_ids=row_ids,
                expected_dimension=expected_dimension,
            )
        else:
            from sentence_transformers import SentenceTransformer

            preprocessing_name = config["preprocessing"]
            if preprocessing_name not in PREPROCESSORS:
                raise ExperimentError(f"Unknown preprocessing pipeline: {preprocessing_name}")
            texts = pool[config["text_column"]].map(PREPROCESSORS[preprocessing_name]).tolist()
            model_directory.mkdir(parents=True, exist_ok=True)
            model_config = config["embedding_model"]
            encoder = SentenceTransformer(
                model_config["model_id"],
                revision=model_config["revision"],
                device=model_config["device"],
                cache_folder=str(model_directory),
                trust_remote_code=bool(model_config["trust_remote_code"]),
                local_files_only=bool(model_config["local_files_only"]),
            )
            started = perf_counter()
            embeddings = encoder.encode(
                texts,
                batch_size=int(model_config["batch_size"]),
                convert_to_numpy=True,
                normalize_embeddings=bool(model_config["normalize_embeddings"]),
                show_progress_bar=False,
            )
            embedding_seconds = perf_counter() - started
            write_embedding_cache(
                cache_path,
                embeddings=embeddings,
                row_ids=row_ids,
                overwrite=args.rebuild_cache,
            )
            cache_manifest = {
                "input_data": str(input_path.relative_to(PROJECT_ROOT)),
                "input_sha256": cache_expectations["input_sha256"],
                "fold_assignments": str(folds_path.relative_to(PROJECT_ROOT)),
                "fold_assignments_sha256": cache_expectations["fold_assignments_sha256"],
                "row_order_sha256": cache_expectations["row_order_sha256"],
                "row_count": len(row_ids),
                "excluded_final_test_fold": final_test_fold,
                "preprocessing": preprocessing_name,
                "embedding_model": model_config,
                "embedding_shape": list(embeddings.shape),
                "embedding_dtype": str(embeddings.dtype),
                "embedding_seconds": embedding_seconds,
            }
            write_json(cache_manifest, cache_manifest_path, overwrite=args.rebuild_cache)

        results: ContextualResults = evaluate_contextual_embeddings(
            pool,
            embeddings,
            label_column=config["label_column"],
            fold_column=config["fold_column"],
            validation_folds=validation_folds,
            models=config["models"],
            expected_labels=preprocessing_config["expected_labels"],
            random_seed=int(preprocessing_config["random_seed"]),
        )
        recorded_embedding_seconds = float(
            cache_manifest.get("embedding_seconds", embedding_seconds)
        )
        output_directory.mkdir(parents=True, exist_ok=True)
        tables = {
            "fold_metrics.csv": results.fold_metrics,
            "summary_metrics.csv": results.summary_metrics,
            "per_class_metrics.csv": results.per_class_metrics,
            "per_class_summary.csv": results.per_class_summary,
            "runtime_metrics.csv": results.runtime_metrics,
            "confusion_matrices.csv": results.confusion_matrices,
        }
        for filename, table in tables.items():
            write_csv(table, output_directory / filename, overwrite=args.overwrite)

        import sentence_transformers
        import torch
        import transformers

        manifest = {
            "config_file": args.config,
            "input_data": str(input_path.relative_to(PROJECT_ROOT)),
            "input_sha256": sha256_file(input_path),
            "fold_assignments": str(folds_path.relative_to(PROJECT_ROOT)),
            "fold_assignments_sha256": sha256_file(folds_path),
            "row_order_sha256": row_order_sha256(row_ids),
            "evaluated_rows": len(row_ids),
            "final_test_fold": final_test_fold,
            "cross_validation_folds": validation_folds,
            "embedding_model": config["embedding_model"],
            "models": config["models"],
            "cache_reused": cache_reused,
            "embedding_seconds": recorded_embedding_seconds,
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "sentence_transformers": sentence_transformers.__version__,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "contains_predictions": False,
            "contains_raw_tweet_text": False,
        }
        write_json(manifest, output_directory / "experiment_manifest.json", overwrite=args.overwrite)
        winner = results.summary_metrics.iloc[0]
        sparse_comparison = "Sparse benchmark evidence was unavailable for direct comparison."
        sparse_directory = resolve_project_path("reports/model_results/classifier_benchmark")
        sparse_summary_path = sparse_directory / "summary_metrics.csv"
        sparse_runtime_path = sparse_directory / "runtime_metrics.csv"
        if sparse_summary_path.is_file() and sparse_runtime_path.is_file():
            sparse_summary = pd.read_csv(sparse_summary_path).sort_values(
                "macro_f1_mean", ascending=False
            )
            sparse_winner = sparse_summary.iloc[0]
            sparse_runtime = pd.read_csv(sparse_runtime_path)
            sparse_fit_seconds = sparse_runtime.loc[
                sparse_runtime["model"] == sparse_winner["model"], "elapsed_seconds"
            ].mean()
            contextual_fit_seconds = results.runtime_metrics.loc[
                results.runtime_metrics["model"] == winner["model"], "elapsed_seconds"
            ].mean()
            sparse_comparison = (
                f"The best sparse candidate scored {sparse_winner['macro_f1_mean']:.4f}; "
                f"the contextual result was {winner['macro_f1_mean'] - sparse_winner['macro_f1_mean']:+.4f} "
                f"lower. Mean fold fit time was {contextual_fit_seconds:.2f}s after a one-time "
                f"{recorded_embedding_seconds:.2f}s embedding pass, "
                f"versus {sparse_fit_seconds:.2f}s for the sparse winner. Sparse linear features also "
                "offer more direct coefficient-based explanations."
            )
        record = (
            "# P5 contextual baseline decision\n\n"
            f"Frozen `{config['embedding_model']['model_id']}` embeddings at revision "
            f"`{config['embedding_model']['revision']}` were evaluated on grouped folds "
            f"{validation_folds}. Fold {final_test_fold} was not embedded or evaluated.\n\n"
            f"Best contextual candidate: **{winner['model']}**, mean macro-F1 "
            f"**{winner['macro_f1_mean']:.4f} ± {winner['macro_f1_std']:.4f}**.\n\n"
            f"{sparse_comparison}\n\n"
            "This is a transfer baseline using a frozen general-purpose English encoder, not "
            "fine-tuning. Inputs longer than 256 wordpieces are truncated by the selected model.\n"
        )
        write_text(record, output_directory / "selection_record.md", overwrite=args.overwrite)
        write_summary_plot(
            results.summary_metrics,
            output_directory / "figures/macro_f1_comparison.png",
            title="Frozen sentence-embedding baseline",
        )
    except (
        ConfigurationError,
        DataFileError,
        DataValidationError,
        ExperimentError,
        KeyError,
        TypeError,
        ValueError,
        OSError,
        RuntimeError,
    ) as error:
        logger.error("Contextual baseline not completed: %s", error)
        return 2

    logger.info("Contextual results written to %s", output_directory)
    logger.info("Frozen fold %s was not embedded or evaluated.", final_test_fold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
