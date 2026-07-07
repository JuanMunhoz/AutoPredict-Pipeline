"""Immutable domain entities.

These are plain, validated data structures with no knowledge of pandas,
FastAPI, MLflow or any adapter. They form the vocabulary the whole system
speaks.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)

from autopredict.core.enums import Asset, Direction


class OHLCV(BaseModel):
    """A single open/high/low/close/volume candle."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    open: float = Field(ge=0)
    high: float = Field(ge=0)
    low: float = Field(ge=0)
    close: float = Field(ge=0)
    volume: float = Field(ge=0)


class SentimentSnapshot(BaseModel):
    """Market-wide sentiment at a point in time (e.g. Fear & Greed Index)."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    fear_greed_value: int = Field(ge=0, le=100)
    classification: str


class PredictionRequest(BaseModel):
    """Inputs required to produce a prediction for one asset."""

    asset: Asset
    features: dict[str, float]


class Prediction(BaseModel):
    """The model's output for a single request."""

    model_config = ConfigDict(frozen=True)

    asset: Asset
    direction: Direction
    probability_up: float = Field(ge=0.0, le=1.0)
    model_version: str
    horizon_hours: int
    created_at: datetime = Field(default_factory=_utcnow)

    @property
    def label(self) -> str:
        """Human-readable direction."""
        return "UP" if self.direction is Direction.UP else "DOWN"


class ModelMetrics(BaseModel):
    """Evaluation metrics for a trained candidate."""

    model_config = ConfigDict(frozen=True)

    roc_auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    log_loss: float
    n_samples: int

    def as_dict(self) -> dict[str, float]:
        """Flat mapping suitable for MLflow logging."""
        return {
            "roc_auc": self.roc_auc,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "log_loss": self.log_loss,
        }


class TrainedModel(BaseModel):
    """A candidate produced by a training run, before any promotion decision."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    asset: Asset
    estimator: object  # fitted sklearn-compatible estimator
    feature_names: list[str]
    metrics: ModelMetrics
    params: dict[str, object]
    trained_at: datetime = Field(default_factory=_utcnow)


class PromotionDecision(BaseModel):
    """Outcome of the promotion gate for a candidate model."""

    model_config = ConfigDict(frozen=True)

    asset: Asset
    promoted: bool
    reason: str
    candidate_auc: float
    incumbent_auc: float | None = None
