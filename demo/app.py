"""Local project dashboard and final cyberbullying classifier demo."""

from __future__ import annotations

from contextlib import asynccontextmanager
import hashlib
import json
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse
import joblib
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

DEMO_DIRECTORY = Path(__file__).resolve().parent
PROJECT_DIRECTORY = DEMO_DIRECTORY.parent
EXPERIMENT_DIRECTORY = PROJECT_DIRECTORY / "experiments"
MODEL_DIRECTORY = DEMO_DIRECTORY / "models"
REGISTRY_PATH = MODEL_DIRECTORY / "registry.json"
INDEX_PATH = DEMO_DIRECTORY / "static/index.html"

EXPERIMENT_NAMES = (
    "experiment_1_data_preparation",
    "experiment_2_text_comparison",
    "experiment_3_model_comparison",
    "experiment_4_model_review",
    "experiment_5_final_evaluation",
)
PLOTS = {name: EXPERIMENT_DIRECTORY / name / "plot.png" for name in EXPERIMENT_NAMES}
PLOT_ALIASES = {
    "data-class-distribution": "experiment_1_data_preparation",
    "data-duplicate-audit": "experiment_1_data_preparation",
    "data-near-duplicates": "experiment_1_data_preparation",
    "text-preprocessing": "experiment_2_text_comparison",
    "text-class-recall": "experiment_2_text_comparison",
    "text-representations": "experiment_2_text_comparison",
    "model-classifiers": "experiment_3_model_comparison",
    "model-contextual": "experiment_3_model_comparison",
    "model-imbalance": "experiment_3_model_comparison",
    "model-leakage": "experiment_3_model_comparison",
    "review-robustness": "experiment_4_model_review",
    "final-evaluation": "experiment_5_final_evaluation",
}
EXPECTED_CLASSES = {"age", "ethnicity", "gender", "not_cyberbullying", "other_cyberbullying", "religion"}


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: Annotated[str, Field(min_length=1, max_length=500)]
    model: str = "logistic_regression"


class PredictionResponse(BaseModel):
    model: str
    model_name: str
    category: str
    confidence: float
    confidence_kind: str
    needs_human_review: bool
    message: str


class HealthResponse(BaseModel):
    ready: bool
    model: str


class ProjectResponse(BaseModel):
    project: dict[str, Any]
    headline: dict[str, float]
    experiments: list[dict[str, Any]]
    data: dict[str, Any]
    comparisons: dict[str, list[dict[str, Any]]]
    final: dict[str, Any]
    safety: dict[str, Any]
    evidence: list[dict[str, str]]
    files: dict[str, Any]
    limitations: list[str]
    models: list[dict[str, Any]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def project_file_counts() -> dict[str, int]:
    excluded = {".git", ".venv", ".pytest_cache", ".ruff_cache", "__pycache__", ".codex-local"}
    files = [
        path
        for path in PROJECT_DIRECTORY.rglob("*")
        if path.is_file()
        and not excluded.intersection(path.relative_to(PROJECT_DIRECTORY).parts)
        and path.name != ".DS_Store"
        and path.name not in {"cyberbullying_tweets.csv", "experiment_ready.csv", "contextual_embeddings.npz", "contextual_embeddings.json"}
    ]
    return {
        "total": len(files),
        "tests": 0,
        "csv": sum(path.suffix == ".csv" for path in files),
        "json": sum(path.suffix == ".json" for path in files),
        "plots": sum(path.suffix == ".png" for path in files),
    }


def build_project_dashboard() -> dict[str, Any]:
    experiment_data = [read_json(EXPERIMENT_DIRECTORY / name / "results.json") for name in EXPERIMENT_NAMES]
    data_result, text_result, model_result, review_result, final_result = experiment_data
    final_metrics = final_result["results"]["metrics"]
    external = review_result["results"]["external"]["primary_logistic_regression"]
    confidence = review_result["results"]["confidence"]
    class_counts = data_result["results"]["prepared"]["classes"]
    row_count = data_result["results"]["prepared"]["rows"]
    per_class = final_result["results"]["per_class"]
    robustness = review_result["results"]["robustness"]
    bootstrap = review_result["results"]["bootstrap_primary_minus_backup"]
    registry = read_json(REGISTRY_PATH)
    counts = project_file_counts()

    experiments = [
        {
            "stage": f"Experiment {item['experiment']}",
            "title": item["title"],
            "status": "complete",
            "result": item["solution"],
            "why": item["question"],
            "settings": item["choices_tried"],
            "approach": item["solution"],
            "details": item["results"],
            "limitations": item["limitations"],
            "metadata": item["metadata"],
        }
        for item in experiment_data
    ]

    def chart_rows(values: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"label": label.replace("_", " "), "value": row["summary"]["macro_f1_mean"]} for label, row in values.items()]

    return {
        "project": {
            "title": "Multi-Class Cyberbullying Detection on Twitter",
            "purpose": "Suggest one of six tweet-level categories to support a human moderator.",
            "status": "Complete research prototype",
            "selected_model": "logistic_regression",
            "final_test_was_protected": all(record["selection_used_final_test"] is False for record in final_result["access_history"]),
        },
        "headline": {
            "macro_f1": float(final_metrics["macro_f1"]),
            "accuracy": float(final_metrics["accuracy"]),
            "balanced_accuracy": float(final_metrics["balanced_accuracy"]),
            "mcc": float(final_metrics["mcc"]),
            "human_review_rate": float(final_metrics["referred_rate"]),
        },
        "experiments": experiments,
        "data": {
            "rows": row_count,
            "classes": [{"label": label.replace("_", " "), "count": count, "share": count / row_count} for label, count in class_counts.items()],
            "external_rows": external["rows"],
            "external_source": "Davidson et al. hate/offensive-language Twitter dataset (MIT licence)",
        },
        "comparisons": {
            "preprocessing": chart_rows(text_result["results"]["preprocessing"]),
            "representations": chart_rows(text_result["results"]["representations"]),
            "classifiers": [{"label": key.replace("_", " "), "value": value} for key, value in model_result["results"]["headline_models"].items()],
            "contextual": chart_rows(model_result["results"]["contextual"]),
            "imbalance": chart_rows(model_result["results"]["imbalance"]),
        },
        "final": {
            "per_class": [{"label": label.replace("_", " "), **metrics} for label, metrics in per_class.items()],
            "robustness": [{"candidate": candidate.replace("_", " "), "condition": condition, **values} for candidate, conditions in robustness.items() for condition, values in conditions.items()],
            "bootstrap": {
                "difference": bootstrap["difference"],
                "low": bootstrap["low"],
                "high": bootstrap["high"],
                "samples": bootstrap["samples"],
            },
        },
        "safety": {
            "confidence": {"referral_threshold": confidence["selected_threshold"], "mean_confidence": final_metrics["mean_confidence"], "observed_accuracy": final_metrics["accuracy"], "expected_calibration_error": confidence["expected_calibration_error"]},
            "referral_curve": confidence["curve"],
            "external": external,
            "identity_probes": [row for row in review_result["results"]["identity_probes"] if row["model"] == "primary_logistic_regression"],
            "errors": review_result["results"]["aggregate_errors"]["primary_logistic_regression"],
            "explanations": [{"class": row["class"].replace("_", " "), "rank": row["rank"], "feature": row["masked_feature"], "weight": row["coefficient"]} for row in review_result["results"]["masked_explanations"]],
        },
        "evidence": [
            {"id": plot_id, "title": plot_id.replace("-", " ").title(), "url": f"/plots/{plot_id}"}
            for plot_id in PLOTS
        ],
        "files": {
            "git_visible_total": counts["total"],
            "tests": counts["tests"],
            "csv": counts["csv"],
            "json": counts["json"],
            "plots": counts["plots"],
            "groups": [
                {"path": "demo/", "does": "Tabbed dashboard, prediction API, and five saved models."},
                {"path": "experiments/", "does": "Five four-file experiment records plus shared leakage-safe logic."},
                {"path": "data/", "does": "Tracked protected folds and external validation data; main text stays local."},
                {"path": "PRESENTATION_GUIDE.md", "does": "Beginner explanation, speaking notes, evidence, and limitations."},
            ],
        },
        "limitations": [
            "A single tweet cannot prove intent, repetition, power imbalance, or a complete cyberbullying event.",
            "Outside-dataset labels do not match the six project classes, so that result is only a binary transfer check.",
            "The outside-dataset false-positive rate is high; predictions must not trigger automatic punishment.",
            "Leetspeak and partly masked words caused major performance drops.",
            "Identity probes are small diagnostic examples, not a complete fairness audit.",
            "The original project dataset source and licence still need final confirmation before redistribution.",
        ],
        "models": registry["models"],
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not REGISTRY_PATH.is_file():
        raise RuntimeError("Demo model registry is missing. Run make experiment-3.")
    registry = read_json(REGISTRY_PATH)
    model_rows = registry.get("models", [])
    model_ids = [item.get("id") for item in model_rows]
    if len(model_ids) != 5 or len(set(model_ids)) != 5 or registry.get("default_model") not in model_ids:
        raise RuntimeError("Demo registry must contain five unique models and a valid default.")
    models = {}
    metadata = {}
    for item in registry["models"]:
        model_path = MODEL_DIRECTORY / item["file"]
        if not model_path.is_file() or sha256_file(model_path) != item["sha256"]:
            raise RuntimeError(f"Demo model is missing or invalid: {item['id']}")
        models[item["id"]] = joblib.load(model_path)
        classes = set(np.asarray(models[item["id"]].named_steps["model"].classes_, dtype=str))
        if classes != EXPECTED_CLASSES:
            raise RuntimeError(f"Demo model class contract is invalid: {item['id']}")
        metadata[item["id"]] = item
    app.state.models = models
    app.state.model_metadata = metadata
    app.state.default_model = registry["default_model"]
    default_metadata = metadata[app.state.default_model]
    threshold = default_metadata.get("referral_threshold")
    if not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
        raise RuntimeError("Default model registry entry needs a referral threshold between zero and one.")
    app.state.referral_threshold = float(threshold)
    app.state.dashboard = build_project_dashboard()
    yield
    app.state.models = {}


app = FastAPI(title="Cyberbullying Research Dashboard", lifespan=lifespan)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["127.0.0.1", "localhost", "testserver"],
)


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(INDEX_PATH)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    return HealthResponse(
        ready=bool(request.app.state.models),
        model=f"{len(request.app.state.models)} selectable models",
    )


@app.get("/project", response_model=ProjectResponse)
def project(request: Request) -> ProjectResponse:
    return ProjectResponse.model_validate(request.app.state.dashboard)


@app.get("/plots/{plot_id}", include_in_schema=False)
def plot(plot_id: str) -> FileResponse:
    path = PLOTS.get(PLOT_ALIASES.get(plot_id, plot_id))
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(path, media_type="image/png")


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    text = payload.text.strip()
    if not text:
        return PredictionResponse(
            model=payload.model,
            model_name="Invalid input",
            category="invalid",
            confidence=0.0,
            confidence_kind="none",
            needs_human_review=True,
            message="Enter some text.",
        )
    model = request.app.state.models.get(payload.model)
    metadata = request.app.state.model_metadata.get(payload.model)
    if model is None or metadata is None:
        raise HTTPException(status_code=404, detail="Unknown model")
    category = str(model.predict([text])[0])
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba([text])[0], dtype=float)
    else:
        decision = np.asarray(model.decision_function([text])[0], dtype=float)
        decision -= decision.max()
        probabilities = np.exp(decision) / np.exp(decision).sum()
    classes = np.asarray(model.named_steps["model"].classes_, dtype=str)
    class_index = int(np.flatnonzero(classes == category)[0])
    confidence = float(probabilities[class_index])
    is_final_model = payload.model == request.app.state.default_model
    needs_review = confidence < request.app.state.referral_threshold if is_final_model else True
    return PredictionResponse(
        model=payload.model,
        model_name=metadata["name"],
        category=category,
        confidence=round(confidence, 4),
        confidence_kind=metadata["confidence_kind"],
        needs_human_review=needs_review,
        message=(
            "Comparison model only; its confidence was not used to set the 45% referral rule."
            if not is_final_model
            else "Low confidence: ask a human moderator to review this tweet."
            if needs_review
            else "Model suggestion only; a human should make the final decision."
        ),
    )
