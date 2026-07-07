"""API request/response schemas (the transport contract)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from autopredict.core.enums import Asset


class PredictRequest(BaseModel):
    """Feature payload for a single prediction."""

    model_config = ConfigDict(extra="forbid")

    asset: Asset
    features: dict[str, float] = Field(
        ..., description="Engineered feature values keyed by feature name."
    )


class PredictResponse(BaseModel):
    """A served prediction."""

    asset: Asset
    direction: str = Field(description="UP or DOWN")
    probability_up: float = Field(ge=0.0, le=1.0)
    model_version: str
    horizon_hours: int
    created_at: datetime


class HealthResponse(BaseModel):
    """Liveness/readiness payload."""

    status: str
    environment: str
    version: str
    models_loaded: list[str]


class ErrorResponse(BaseModel):
    """Uniform error envelope."""

    detail: str
    error_type: str
