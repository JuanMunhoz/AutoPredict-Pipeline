"""Health & readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request

from autopredict import __version__
from autopredict.api.dependencies import PredictorRegistry
from autopredict.api.schemas import HealthResponse
from autopredict.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Liveness probe — the process is up."""
    registry: PredictorRegistry = request.app.state.registry
    settings = get_settings()
    return HealthResponse(
        status="ok",
        environment=settings.env.value,
        version=__version__,
        models_loaded=registry.loaded_assets,
    )


@router.get("/ready", response_model=HealthResponse)
def ready(request: Request) -> HealthResponse:
    """Readiness probe — at least one model is loaded and servable."""
    registry: PredictorRegistry = request.app.state.registry
    settings = get_settings()
    status = "ok" if registry.loaded_assets else "degraded"
    return HealthResponse(
        status=status,
        environment=settings.env.value,
        version=__version__,
        models_loaded=registry.loaded_assets,
    )
