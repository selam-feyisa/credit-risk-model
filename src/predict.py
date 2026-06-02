"""Shared prediction helpers for the API and batch inference."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import mlflow.pyfunc
import pandas as pd


DEFAULT_MODEL_URI = "models:/CreditRiskBestModel/Production"
LOCAL_MODEL_PATH = Path("artifacts/best_model.joblib")


def load_model(model_uri: str | None = None) -> Any:
    """Load the registered MLflow model, falling back to a local artifact."""
    uri = model_uri or os.getenv("MODEL_URI", DEFAULT_MODEL_URI)
    try:
        return mlflow.pyfunc.load_model(uri)
    except Exception:
        if LOCAL_MODEL_PATH.exists():
            return joblib.load(LOCAL_MODEL_PATH)
        raise


def predict_risk_probability(model: Any, features: dict) -> float:
    """Return the probability that a customer belongs to the high-risk class."""
    frame = pd.DataFrame([features])
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(frame)[0][1]
    else:
        prediction = model.predict(frame)
        if isinstance(prediction, pd.DataFrame):
            probability = prediction.iloc[0, -1]
        else:
            probability = prediction[0]
    return float(probability)
