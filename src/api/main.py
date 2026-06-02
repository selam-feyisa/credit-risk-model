"""FastAPI application for serving the credit risk model."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from src.api.pydantic_models import PredictionRequest, PredictionResponse
from src.predict import load_model, predict_risk_probability


app = FastAPI(
    title="Bati Bank Credit Risk API",
    version="1.0.0",
    description="Scores alternative-data customers using the registered MLflow model.",
)

model = None


@app.on_event("startup")
def startup_event() -> None:
    global model
    model = load_model()


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    probability = predict_risk_probability(model, request.features)
    label = "high_risk" if probability >= 0.5 else "low_risk"
    return PredictionResponse(risk_probability=probability, risk_label=label)
