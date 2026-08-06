"""Deterministic non-final random folds for leakage credibility analysis."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from cyberbullying_detection.experiments.tfidf_baseline import (
    ExperimentError,
    validate_experiment_inputs,
)


def build_non_final_random_folds(
    frame: pd.DataFrame,
    grouped_assignments: pd.DataFrame,
    *,
    row_id_column: str,
    label_column: str,
    fold_column: str,
    final_test_fold: int,
    validation_folds: list[int],
    random_seed: int,
) -> pd.DataFrame:
    """Stratify only the non-final pool while preserving protected fold membership."""
    merged = validate_experiment_inputs(
        frame,
        grouped_assignments,
        row_id_column=row_id_column,
        label_column=label_column,
        fold_column=fold_column,
        final_test_fold=final_test_fold,
        cross_validation_folds=validation_folds,
    )
    if len(validation_folds) < 2:
        raise ExperimentError("Leakage-gap analysis requires at least two validation folds.")
    output = merged[[row_id_column, fold_column]].copy()
    non_final_mask = output[fold_column] != final_test_fold
    non_final = merged.loc[non_final_mask]
    splitter = StratifiedKFold(
        n_splits=len(validation_folds), shuffle=True, random_state=random_seed
    )
    random_values = output[fold_column].to_numpy(copy=True)
    for fold, (_, validation_indices) in zip(
        validation_folds,
        splitter.split(non_final[row_id_column], non_final[label_column]),
        strict=True,
    ):
        absolute_indices = non_final.index.to_numpy()[validation_indices]
        random_values[absolute_indices] = fold
    output[fold_column] = random_values.astype(int)
    if not output.loc[~non_final_mask, fold_column].eq(final_test_fold).all():
        raise ExperimentError("Random fold construction changed protected final-test membership.")
    if set(output.loc[non_final_mask, fold_column]) != set(validation_folds):
        raise ExperimentError("Random fold construction did not populate every validation fold.")
    return output.reset_index(drop=True)


def count_groups_crossing_folds(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    row_id_column: str,
    group_column: str,
    fold_column: str,
    final_test_fold: int,
) -> int:
    """Count non-final near-duplicate groups assigned to multiple folds."""
    membership = frame[[row_id_column, group_column]].merge(
        assignments[[row_id_column, fold_column]], on=row_id_column, validate="one_to_one"
    )
    membership = membership.loc[membership[fold_column] != final_test_fold]
    return int((membership.groupby(group_column)[fold_column].nunique() > 1).sum())
