"""HTTP middleware: structured access logs + Prometheus metrics + request id.

Every request is timed, tagged with a correlation id (bound into the structlog
context so all logs during the request carry it), counted and observed. Errors
are logged with the request id for traceability.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from autopredict.monitoring import get_logger
from autopredict.monitoring.metrics import (
    PROMETHEUS_AVAILABLE,
    REQUEST_COUNT,
    REQUEST_LATENCY,
)

logger = get_logger(__name__)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Bind a request id, time the request and emit metrics + access logs."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        structlog.contextvars.bind_contextvars(request_id=request_id, path=request.url.path)
        start = time.perf_counter()
        endpoint = request.url.path
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            status = 500
            logger.exception("request_unhandled_error", method=request.method)
            raise
        finally:
            elapsed = time.perf_counter() - start
            if PROMETHEUS_AVAILABLE:
                REQUEST_COUNT.labels(request.method, endpoint, str(status)).inc()
                REQUEST_LATENCY.labels(endpoint).observe(elapsed)
            logger.info(
                "request_completed",
                method=request.method,
                status=status,
                duration_ms=round(elapsed * 1000, 2),
            )
            structlog.contextvars.clear_contextvars()

        response.headers["x-request-id"] = request_id
        response.headers["x-process-time-ms"] = f"{elapsed * 1000:.2f}"
        return response
