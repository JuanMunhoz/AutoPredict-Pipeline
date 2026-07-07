"""Prometheus metrics for the API.

Exposes request counters, latency histograms and a prediction-direction counter
so the served model can be observed in production (throughput, tail latency,
class balance drift). Imported lazily-safe: if ``prometheus_client`` is absent,
no-op stand-ins keep the API importable.
"""

from __future__ import annotations

try:
    from prometheus_client import Counter, Histogram

    REQUEST_COUNT = Counter(
        "autopredict_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status"],
    )
    REQUEST_LATENCY = Histogram(
        "autopredict_request_latency_seconds",
        "Request latency in seconds",
        ["endpoint"],
    )
    PREDICTION_COUNT = Counter(
        "autopredict_predictions_total",
        "Predictions served",
        ["asset", "direction"],
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover
    PROMETHEUS_AVAILABLE = False
    REQUEST_COUNT = REQUEST_LATENCY = PREDICTION_COUNT = None  # type: ignore[assignment]
