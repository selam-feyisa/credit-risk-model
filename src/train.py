"""Train, tune, track, and register credit risk models with MLflow."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from src.data_processing import (
    RANDOM_STATE,
    TARGET_COLUMN,
    ModelReadyTransformer,
    TransactionFeatureEngineer,
)


EXPERIMENT_NAME = "credit-risk-proxy-model"
REGISTERED_MODEL_NAME = "CreditRiskBestModel"
DEFAULT_TRACKING_URI = "file:mlruns"


def load_training_data(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_training_frame(raw_data: pd.DataFrame) -> pd.DataFrame:
    engineer = TransactionFeatureEngineer(include_target=True)
    return engineer.fit_transform(raw_data)


def evaluate_classifier(
    model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> dict:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def candidate_models() -> dict[str, tuple[Pipeline, dict]]:
    return {
        "logistic_regression": (
            Pipeline(
                steps=[
                    ("preprocess", ModelReadyTransformer()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=1000,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__solver": ["liblinear"],
            },
        ),
        "random_forest": (
            Pipeline(
                steps=[
                    ("preprocess", ModelReadyTransformer()),
                    (
                        "model",
                        RandomForestClassifier(
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [5, 10, None],
                "model__min_samples_leaf": [1, 5],
            },
        ),
    }


def train_and_track(raw_data: pd.DataFrame, artifact_dir: str | Path = "artifacts"):
    engineered = prepare_training_frame(raw_data)
    X = engineered.drop(columns=[TARGET_COLUMN])
    y = engineered[TARGET_COLUMN].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    mlflow.set_experiment(EXPERIMENT_NAME)
    best_run = None
    best_model = None
    best_metrics = None
    best_score = -1.0

    for model_name, (pipeline, param_grid) in candidate_models().items():
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=3,
            n_jobs=1,
        )
        with mlflow.start_run(run_name=model_name) as run:
            search.fit(X_train, y_train)
            metrics = evaluate_classifier(search.best_estimator_, X_test, y_test)
            mlflow.log_params(search.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(search.best_estimator_, artifact_path="model")

            if metrics["roc_auc"] > best_score:
                best_score = metrics["roc_auc"]
                best_run = run.info.run_id
                best_model = search.best_estimator_
                best_metrics = metrics

    if best_model is None or best_run is None:
        raise RuntimeError("No model was trained.")

    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, artifact_path / "best_model.joblib")

    model_uri = f"runs:/{best_run}/model"
    try:
        mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
    except Exception as exc:
        print(f"Model registry step skipped: {exc}")

    return best_model, best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train credit risk models.")
    parser.add_argument("--input", default="data/raw/training.csv")
    parser.add_argument("--artifact-dir", default="artifacts")
    args = parser.parse_args()

    raw_data = load_training_data(args.input)
    _, metrics = train_and_track(raw_data, artifact_dir=args.artifact_dir)
    print("Best model metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
