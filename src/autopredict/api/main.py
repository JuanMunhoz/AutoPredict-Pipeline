"""FastAPI application factory.

Wires middleware, routes, exception handling and a ``/metrics`` endpoint.
Predictors are loaded once during the lifespan startup and stored on
``app.state`` so requests never pay model-load cost.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from autopredict import __version__
from autopredict.api.dependencies import build_registry
from autopredict.api.middleware import ObservabilityMiddleware
from autopredict.api.routes import health, predict
from autopredict.api.schemas import ErrorResponse
from autopredict.config import get_settings
from autopredict.core.exceptions import AutoPredictError
from autopredict.monitoring import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json)
    app.state.registry = build_registry(settings)
    logger.info("api_startup", models_loaded=app.state.registry.loaded_assets)
    yield
    logger.info("api_shutdown")


def create_app() -> FastAPI:
    """Application factory — importable by tests and ASGI servers."""
    settings = get_settings()
    app = FastAPI(
        title="AutoPredict API",
        version=__version__,
        description="Predicts the 4h price direction (UP/DOWN) for BTC and SOL.",
        lifespan=lifespan,
    )
    app.add_middleware(ObservabilityMiddleware)
    app.include_router(health.router)
    app.include_router(predict.router)

    @app.exception_handler(AutoPredictError)
    async def _domain_error_handler(_: Request, exc: AutoPredictError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(detail=str(exc), error_type=type(exc).__name__).model_dump(),
        )

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> PlainTextResponse:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {"service": "autopredict", "version": __version__, "docs": "/docs"}

    logger.info("app_created", env=settings.env.value)
    return app


app = create_app()
