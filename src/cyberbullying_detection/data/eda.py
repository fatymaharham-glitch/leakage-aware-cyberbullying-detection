"""Aggregate exploratory analysis and publication-quality, text-free figures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cyberbullying_detection.data.features import FEATURE_COLUMNS, extract_text_features
from cyberbullying_detection.utils.io import write_csv


@dataclass(frozen=True)
class EdaTables:
    """Machine-readable EDA tables and their per-row feature frame."""

    dataset_summary: pd.DataFrame
    class_distribution: pd.DataFrame
    missing_values: pd.DataFrame
    text_length_by_class: pd.DataFrame
    social_media_features_by_class: pd.DataFrame
    features: pd.DataFrame
    labels: pd.Series


def _display_label(value: object) -> str:
    """Return a stable label for aggregate tables, including missing values."""
    if value is None or (pd.api.types.is_scalar(value) and bool(pd.isna(value))):
        return "<missing>"
    return str(value)


def _label_order(observed_labels: pd.Series, expected_labels: list[str]) -> list[str]:
    """Keep configured classes first, then sort unexpected observed labels."""
    unexpected = sorted(set(observed_labels) - set(expected_labels))
    return [*expected_labels, *unexpected]


def build_eda_tables(
    frame: pd.DataFrame,
    *,
    text_column: str,
    label_column: str,
    expected_labels: list[str],
) -> EdaTables:
    """Calculate overall and per-class descriptive tables without exposing tweets."""
    features = extract_text_features(frame[text_column])
    labels = frame[label_column].map(_display_label)
    label_order = _label_order(labels, expected_labels)
    row_count = len(frame)

    class_counts = labels.value_counts().reindex(label_order, fill_value=0)
    class_distribution = class_counts.rename_axis("cyberbullying_type").reset_index(
        name="row_count"
    )
    class_distribution["class_percentage"] = np.where(
        row_count > 0,
        class_distribution["row_count"] / row_count * 100,
        0.0,
    )

    missing_values = frame.isna().sum().rename_axis("column").reset_index(name="missing_count")
    missing_values["missing_percentage"] = np.where(
        row_count > 0, missing_values["missing_count"] / row_count * 100, 0.0
    )

    overall_records: list[tuple[str, float | int]] = [
        ("row_count", row_count),
        ("observed_class_count", int((class_counts > 0).sum())),
    ]
    for column in FEATURE_COLUMNS:
        overall_records.append((f"{column}_mean", float(features[column].mean())))
        overall_records.append((f"{column}_median", float(features[column].median())))
        overall_records.append((f"{column}_total", int(features[column].sum())))
    dataset_summary = pd.DataFrame(overall_records, columns=["metric", "value"])

    feature_frame = features.copy()
    feature_frame["cyberbullying_type"] = labels
    grouped = feature_frame.groupby("cyberbullying_type", sort=False)
    text_length_by_class = grouped.agg(
        row_count=("character_length", "size"),
        character_length_mean=("character_length", "mean"),
        character_length_median=("character_length", "median"),
        character_length_min=("character_length", "min"),
        character_length_max=("character_length", "max"),
        word_count_mean=("word_count", "mean"),
        word_count_median=("word_count", "median"),
        word_count_min=("word_count", "min"),
        word_count_max=("word_count", "max"),
    ).reindex(label_order, fill_value=0).reset_index()

    social_media_features_by_class = grouped.agg(
        row_count=("url_count", "size"),
        url_count_total=("url_count", "sum"),
        url_count_mean=("url_count", "mean"),
        user_mention_count_total=("user_mention_count", "sum"),
        user_mention_count_mean=("user_mention_count", "mean"),
        hashtag_count_total=("hashtag_count", "sum"),
        hashtag_count_mean=("hashtag_count", "mean"),
        emoji_presence_count=("has_emoji", "sum"),
        emoji_presence_rate=("has_emoji", "mean"),
        uppercase_ratio_mean=("uppercase_ratio", "mean"),
        punctuation_count_total=("punctuation_count", "sum"),
        punctuation_count_mean=("punctuation_count", "mean"),
        repeated_character_count_total=("repeated_character_count", "sum"),
        repeated_character_count_mean=("repeated_character_count", "mean"),
    ).reindex(label_order, fill_value=0).reset_index()
    return EdaTables(
        dataset_summary=dataset_summary,
        class_distribution=class_distribution,
        missing_values=missing_values,
        text_length_by_class=text_length_by_class,
        social_media_features_by_class=social_media_features_by_class,
        features=features,
        labels=labels,
    )


def _save_figure(figure: plt.Figure, output_path: Path) -> None:
    """Write a PNG atomically so reruns never leave a partial report image."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.tmp.png")
    try:
        figure.savefig(temporary_path, format="png", dpi=180, bbox_inches="tight")
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
        plt.close(figure)


def write_eda_outputs(tables: EdaTables, *, figures_dir: Path, tables_dir: Path) -> None:
    """Write requested aggregate tables and figures in deterministic filenames."""
    output_tables = {
        "dataset_summary.csv": tables.dataset_summary,
        "class_distribution.csv": tables.class_distribution,
        "missing_values.csv": tables.missing_values,
        "text_length_by_class.csv": tables.text_length_by_class,
        "social_media_features_by_class.csv": tables.social_media_features_by_class,
    }
    for filename, table in output_tables.items():
        write_csv(table, tables_dir / filename, overwrite=True)

    distribution = tables.class_distribution
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(distribution["cyberbullying_type"], distribution["row_count"], color="#3969ac")
    axis.set_title("Class distribution")
    axis.set_xlabel("Cyberbullying class")
    axis.set_ylabel("Tweet count")
    axis.tick_params(axis="x", rotation=30)
    _save_figure(figure, figures_dir / "class_distribution.png")

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(tables.features["character_length"], bins=50, color="#3e9651", edgecolor="white")
    axis.set_title("Text-length distribution")
    axis.set_xlabel("Characters per tweet")
    axis.set_ylabel("Tweet count")
    _save_figure(figure, figures_dir / "text_length_distribution.png")

    label_counts = distribution.set_index("cyberbullying_type")["row_count"]
    non_empty_labels = [
        label
        for label in distribution["cyberbullying_type"].tolist()
        if label_counts[label] > 0
    ]
    _write_by_class_figures(tables, figures_dir, non_empty_labels)

    figure, axis = plt.subplots(figsize=(8, 5))
    missing = tables.missing_values
    axis.bar(missing["column"], missing["missing_count"], color="#cc2529")
    axis.set_title("Missing values by column")
    axis.set_xlabel("Column")
    axis.set_ylabel("Missing rows")
    axis.tick_params(axis="x", rotation=20)
    _save_figure(figure, figures_dir / "missing_values.png")


def _write_by_class_figures(
    tables: EdaTables, figures_dir: Path, labels: list[str]
) -> None:
    """Write class-wise figures from labels retained only in memory."""
    feature_frame = tables.features
    label_series = tables.labels

    character_data = [
        feature_frame.loc[label_series == label, "character_length"].tolist()
        for label in labels
    ]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.boxplot(character_data, tick_labels=labels, showfliers=False)
    axis.set_title("Text length by class")
    axis.set_xlabel("Cyberbullying class")
    axis.set_ylabel("Characters per tweet")
    axis.tick_params(axis="x", rotation=30)
    _save_figure(figure, figures_dir / "text_length_by_class.png")

    word_data = [
        feature_frame.loc[label_series == label, "word_count"].tolist()
        for label in labels
    ]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.boxplot(word_data, tick_labels=labels, showfliers=False)
    axis.set_title("Word count by class")
    axis.set_xlabel("Cyberbullying class")
    axis.set_ylabel("Words per tweet")
    axis.tick_params(axis="x", rotation=30)
    _save_figure(figure, figures_dir / "word_count_by_class.png")

    social = tables.social_media_features_by_class.set_index("cyberbullying_type").loc[labels]
    metrics = [
        "url_count_mean",
        "user_mention_count_mean",
        "hashtag_count_mean",
        "emoji_presence_rate",
        "uppercase_ratio_mean",
        "repeated_character_count_mean",
    ]
    positions = np.arange(len(labels))
    width = 0.13
    figure, axis = plt.subplots(figsize=(13, 6))
    for offset, metric in enumerate(metrics):
        axis.bar(
            positions + (offset - (len(metrics) - 1) / 2) * width,
            social[metric],
            width=width,
            label=metric.replace("_mean", "").replace("_", " "),
        )
    axis.set_title("Social-media features by class")
    axis.set_xlabel("Cyberbullying class")
    axis.set_ylabel("Mean or rate")
    axis.set_xticks(positions, labels, rotation=30)
    axis.legend(fontsize=8, ncol=2)
    _save_figure(figure, figures_dir / "social_media_features_by_class.png")
