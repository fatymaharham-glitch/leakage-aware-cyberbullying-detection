"""Shared, leakage-safe experiment utilities and saved-model compatibility symbols."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
import unicodedata
from typing import Any, Callable

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/cyberbullying-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/cyberbullying-cache")

import emoji
import joblib
import numpy as np
import pandas as pd
import regex
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/experiment_ready.csv"
RAW_DATA = ROOT / "data/cyberbullying_tweets.csv"
FOLDS = ROOT / "data/leakage_aware_folds.csv"
EXTERNAL_DATA = ROOT / "data/external_validation.csv"
LABELS = ["age", "ethnicity", "gender", "religion", "other_cyberbullying", "not_cyberbullying"]
FINAL_FOLD = 0
CV_FOLDS = [1, 2, 3, 4]
RANDOM_SEED = 42
INPUT_SHA256 = "f203129d7c5f761471ab084cd6acac1b16f2039182e1d654a25290b7681af907"
FOLDS_SHA256 = "406f240c9adbbec0a4fa95bade2ba8b3ffbf3432cb8fb1e83c3b918347349dd4"

URL_PATTERN = regex.compile(r'(?i)\b(?:https?://|www\.)[^\s<>"]+')
USER_PATTERN = regex.compile(r"(?<!\w)@[A-Za-z0-9_]+")
HASHTAG_PATTERN = regex.compile(r"(?<!\w)#([\p{L}\p{N}_]+)")


def _coerce_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def normalize(value: object) -> str:
    text = unicodedata.normalize("NFKC", _coerce_text(value))
    return regex.sub(r"\s+", " ", text).strip()


def preprocess_p0(value: object) -> str:
    text = URL_PATTERN.sub(" <URL> ", normalize(value))
    return regex.sub(r"\s+", " ", USER_PATTERN.sub(" <USER> ", text)).strip()


def preprocess_p1(value: object) -> str:
    text = normalize(value).lower()
    text = URL_PATTERN.sub(" <url> ", text)
    text = USER_PATTERN.sub(" <user> ", text)
    text = HASHTAG_PATTERN.sub(r" hashtag_\1 ", text)
    return regex.sub(r"\s+", " ", emoji.demojize(text, delimiters=(" ", " "))).strip()


def preprocess_p2(value: object) -> str:
    text = HASHTAG_PATTERN.sub(r" \1 ", USER_PATTERN.sub(" ", URL_PATTERN.sub(" ", normalize(value).lower())))
    tokens = regex.sub(r"[^\p{L}\p{N}\s]", " ", text).split()
    return " ".join(token for token in tokens if token not in ENGLISH_STOP_WORDS)


PREPROCESSORS: dict[str, Callable[[object], str]] = {
    "p0": preprocess_p0,
    "p1": preprocess_p1,
    "p2": preprocess_p2,
}


class EncodedXGBoost(BaseEstimator, ClassifierMixin):
    """XGBoost adapter retained at this module path for existing Joblib files."""

    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state

    def fit(self, features: Any, labels: Any) -> "EncodedXGBoost":
        from xgboost import XGBClassifier

        self.encoder_ = LabelEncoder().fit(labels)
        self.model_ = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.5,
            n_jobs=4,
            random_state=self.random_state,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
        )
        self.model_.fit(features, self.encoder_.transform(labels))
        self.classes_ = self.encoder_.classes_
        return self

    def predict(self, features: Any) -> np.ndarray:
        return self.encoder_.inverse_transform(self.model_.predict(features).astype(int))

    def predict_proba(self, features: Any) -> np.ndarray:
        return self.model_.predict_proba(features)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_project_data() -> pd.DataFrame:
    """Load data only when row membership, folds, hashes, and groups are intact."""
    if not DATA.is_file() or not FOLDS.is_file():
        raise FileNotFoundError("Expected local data/experiment_ready.csv and tracked data/leakage_aware_folds.csv.")
    if sha256_file(DATA) != INPUT_SHA256:
        raise ValueError("Prepared dataset checksum differs from the protected experiment input.")
    if sha256_file(FOLDS) != FOLDS_SHA256:
        raise ValueError("Fold-assignment checksum differs from the protected experiment input.")
    data = pd.read_csv(DATA)
    folds = pd.read_csv(FOLDS)
    required_data = {"row_id", "tweet_text", "cyberbullying_type", "near_duplicate_group_id"}
    required_folds = {"row_id", "fold"}
    if not required_data.issubset(data.columns) or not required_folds.issubset(folds.columns):
        raise ValueError("Prepared data or fold assignments are missing required columns.")
    if data["row_id"].duplicated().any() or folds["row_id"].duplicated().any():
        raise ValueError("Row identifiers must be unique in data and fold assignments.")
    if set(data["row_id"]) != set(folds["row_id"]):
        raise ValueError("Data and fold assignments do not contain exactly the same row identifiers.")
    if set(folds["fold"].astype(int)) != {0, 1, 2, 3, 4}:
        raise ValueError("Fold assignments must contain exactly folds 0 through 4.")
    if set(data["cyberbullying_type"].unique()) != set(LABELS):
        raise ValueError("Project labels do not match the expected six-class contract.")
    merged = data.merge(folds[["row_id", "fold"]], on="row_id", validate="one_to_one")
    overlap = merged.groupby("near_duplicate_group_id", dropna=False)["fold"].nunique()
    if (overlap > 1).any():
        raise ValueError("At least one near-duplicate group crosses protected folds.")
    return merged


def build_features(
    kind: str = "combined",
    preprocessing: str = "p1",
    *,
    word_max_features: int = 60_000,
    character_max_features: int = 60_000,
) -> TfidfVectorizer | FeatureUnion:
    cleaner = PREPROCESSORS[preprocessing]
    word = TfidfVectorizer(
        preprocessor=cleaner,
        tokenizer=None,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=word_max_features,
        sublinear_tf=True,
        lowercase=False,
    )
    character = TfidfVectorizer(
        preprocessor=cleaner,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
        max_features=character_max_features,
        sublinear_tf=True,
        lowercase=False,
    )
    if kind == "word":
        return word
    if kind == "character":
        return character
    if kind != "combined":
        raise ValueError(f"Unknown representation: {kind}")
    return FeatureUnion([("word", word), ("character", character)])


def build_model(model_id: str) -> Any:
    models: dict[str, Callable[[], Any]] = {
        "multinomial_nb": lambda: MultinomialNB(alpha=1.0),
        "logistic_regression_c0_5": lambda: LogisticRegression(C=0.5, max_iter=1000, random_state=RANDOM_SEED),
        "logistic_regression_c1": lambda: LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED),
        "logistic_regression_c2": lambda: LogisticRegression(C=2.0, max_iter=1000, random_state=RANDOM_SEED),
        "logistic_regression_c1_balanced": lambda: LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED),
        "linear_svm_c0_5": lambda: LinearSVC(C=0.5, max_iter=10000, random_state=RANDOM_SEED),
        "linear_svm_c1": lambda: LinearSVC(C=1.0, max_iter=10000, random_state=RANDOM_SEED),
        "linear_svm_c2": lambda: LinearSVC(C=2.0, max_iter=10000, random_state=RANDOM_SEED),
        "linear_svm_c1_balanced": lambda: LinearSVC(C=1.0, max_iter=10000, class_weight="balanced", random_state=RANDOM_SEED),
        "random_forest": lambda: RandomForestClassifier(n_estimators=75, max_depth=30, max_features="sqrt", n_jobs=4, random_state=RANDOM_SEED),
        "xgboost": lambda: EncodedXGBoost(),
    }
    try:
        return models[model_id]()
    except KeyError as error:
        raise ValueError(f"Unknown model: {model_id}") from error


def build_final_model(model_id: str = "logistic_regression_c1_balanced") -> Pipeline:
    return Pipeline([("representation", build_features()), ("model", build_model(model_id))])


@dataclass
class Evaluation:
    summary: dict[str, float]
    folds: list[dict[str, Any]]
    per_class: list[dict[str, Any]]
    confusion_matrix: list[list[int]]
    labels: list[str]
    truth: list[str]
    predictions: list[str]
    confidence: list[float]


def evaluate_candidate(
    frame: pd.DataFrame,
    model_id: str,
    *,
    representation: str = "combined",
    preprocessing: str = "p1",
    word_max_features: int = 60_000,
    character_max_features: int = 60_000,
    fold_column: str = "fold",
    folds: list[int] | None = None,
) -> Evaluation:
    validation_folds = folds or CV_FOLDS
    all_truth: list[str] = []
    all_predictions: list[str] = []
    all_confidence: list[float] = []
    fold_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for fold in validation_folds:
        train = frame[(frame[fold_column] != fold) & (frame["fold"] != FINAL_FOLD)]
        validation = frame[frame[fold_column] == fold]
        if train.empty or validation.empty:
            raise ValueError(f"Fold {fold} has an empty train or validation partition.")
        representation_model = build_features(
            representation,
            preprocessing,
            word_max_features=word_max_features,
            character_max_features=character_max_features,
        )
        started = time.perf_counter()
        train_matrix = representation_model.fit_transform(train["tweet_text"])
        validation_matrix = representation_model.transform(validation["tweet_text"])
        fit_started = time.perf_counter()
        model = build_model(model_id)
        model.fit(train_matrix, train["cyberbullying_type"])
        fit_seconds = time.perf_counter() - fit_started
        predict_started = time.perf_counter()
        predictions = np.asarray(model.predict(validation_matrix), dtype=str)
        predict_seconds = time.perf_counter() - predict_started
        if hasattr(model, "predict_proba"):
            confidence = np.asarray(model.predict_proba(validation_matrix)).max(axis=1)
        elif hasattr(model, "decision_function"):
            confidence = softmax(np.asarray(model.decision_function(validation_matrix)), axis=1).max(axis=1)
        else:
            confidence = np.full(len(validation), np.nan)
        truth = validation["cyberbullying_type"].astype(str).to_numpy()
        precision, recall, class_f1, support = precision_recall_fscore_support(
            truth, predictions, labels=LABELS, zero_division=0
        )
        fold_rows.append(
            {
                "fold": int(fold),
                "rows": int(len(validation)),
                "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
                "accuracy": float(accuracy_score(truth, predictions)),
                "mcc": float(matthews_corrcoef(truth, predictions)),
                "feature_count": int(train_matrix.shape[1]),
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "total_seconds": time.perf_counter() - started,
            }
        )
        per_class_rows.extend(
            {
                "fold": int(fold),
                "class": label,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(class_f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        )
        all_truth.extend(truth.tolist())
        all_predictions.extend(predictions.tolist())
        all_confidence.extend(confidence.astype(float).tolist())
    fold_frame = pd.DataFrame(fold_rows)
    truth_array = np.asarray(all_truth)
    prediction_array = np.asarray(all_predictions)
    summary = {
        "macro_f1_mean": float(fold_frame["macro_f1"].mean()),
        "macro_f1_std": float(fold_frame["macro_f1"].std(ddof=0)),
        "balanced_accuracy_mean": float(fold_frame["balanced_accuracy"].mean()),
        "accuracy_mean": float(fold_frame["accuracy"].mean()),
        "mcc_mean": float(fold_frame["mcc"].mean()),
        "runtime_seconds": float(fold_frame["total_seconds"].sum()),
    }
    return Evaluation(
        summary=summary,
        folds=fold_rows,
        per_class=per_class_rows,
        confusion_matrix=confusion_matrix(truth_array, prediction_array, labels=LABELS).astype(int).tolist(),
        labels=LABELS,
        truth=all_truth,
        predictions=all_predictions,
        confidence=all_confidence,
    )


def metadata(*, source: str = "fresh_reproduction") -> dict[str, Any]:
    packages = ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "xgboost", "sentence-transformers"]
    versions: dict[str, str] = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return {
        "evidence_source": source,
        "random_seed": RANDOM_SEED,
        "input_sha256": sha256_file(DATA),
        "fold_assignments_sha256": sha256_file(FOLDS),
        "final_test_fold": FINAL_FOLD,
        "development_folds": CV_FOLDS,
        "class_order": LABELS,
        "versions": versions,
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def update_registry_metrics(scores: dict[str, float]) -> None:
    registry_path = ROOT / "demo/models/registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    score_ids = {
        "naive_bayes": "multinomial_nb",
        "logistic_regression": "logistic_regression_c1_balanced",
        "linear_svm": "linear_svm_c0_5",
        "random_forest": "random_forest",
        "xgboost": "xgboost",
    }
    for item in registry["models"]:
        model_path = registry_path.parent / item["file"]
        item["macro_f1"] = scores[score_ids[item["id"]]]
        item["sha256"] = sha256_file(model_path)
        item["size_mib"] = round(model_path.stat().st_size / 1024**2, 2)
    write_json(registry_path, registry)


def save_final_model(model: Pipeline, threshold: float) -> None:
    model_path = ROOT / "demo/models/logistic_regression.joblib"
    joblib.dump(model, model_path, compress=3)
    registry_path = model_path.parent / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    for item in registry["models"]:
        if item["id"] == "logistic_regression":
            item["sha256"] = sha256_file(model_path)
            item["size_mib"] = round(model_path.stat().st_size / 1024**2, 2)
            item["referral_threshold"] = threshold
    write_json(registry_path, registry)


def remove_punctuation(text: object) -> str:
    return regex.sub(r"[\p{P}\p{S}]", " ", _coerce_text(text))


def normalize_repeats(text: object) -> str:
    return regex.sub(r"(.)\1{2,}", r"\1", _coerce_text(text))


def leetspeak(text: object) -> str:
    return _coerce_text(text).translate(str.maketrans({"a": "4", "e": "3", "i": "1", "o": "0", "s": "5"}))


def partial_masking(text: object) -> str:
    def mask(match: regex.Match) -> str:
        token = match.group(0)
        return token[:2] + "*" * max(1, len(token) - 3) + token[-1:]

    return regex.sub(r"\p{L}{5,}", mask, _coerce_text(text))
