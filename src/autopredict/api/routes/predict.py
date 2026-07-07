"""Prediction endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from autopredict.api.dependencies import PredictorRegistry
from autopredict.api.schemas import PredictRequest, PredictResponse
from autopredict.core.exceptions import FeatureEngineeringError, ModelNotFoundError
from autopredict.monitoring import get_logger
from autopredict.monitoring.metrics import PREDICTION_COUNT, PROMETHEUS_AVAILABLE

logger = get_logger(__name__)
router = APIRouter(tags=["inference"])


@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    """Return the predicted 4h price direction for one asset."""
    registry: PredictorRegistry = request.app.state.registry
    try:
        predictor = registry.get(payload.asset)
        prediction = predictor.predict(payload.asset, payload.features)
    except ModelNotFoundError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except FeatureEngineeringError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    if PROMETHEUS_AVAILABLE:
        PREDICTION_COUNT.labels(payload.asset.value, prediction.label).inc()

    logger.info(
        "prediction_served",
        asset=payload.asset.value,
        direction=prediction.label,
        probability_up=round(prediction.probability_up, 4),
        model_version=prediction.model_version,
    )
    return PredictResponse(
        asset=prediction.asset,
        direction=prediction.label,
        probability_up=prediction.probability_up,
        model_version=prediction.model_version,
        horizon_hours=prediction.horizon_hours,
        created_at=prediction.created_at,
    )
