"""Pydantic request and response schemas for the credit risk API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Model-ready customer features for a single prediction."""

    model_config = ConfigDict(extra="allow")

    features: dict[str, Any] = Field(
        ...,
        description="Model feature names and values for one customer.",
    )


class PredictionResponse(BaseModel):
    risk_probability: float = Field(..., ge=0, le=1)
    risk_label: str
