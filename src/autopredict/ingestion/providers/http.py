"""Shared resilient HTTP client used by all providers.

Public crypto APIs rate-limit and occasionally 5xx. ``tenacity`` gives us
exponential backoff with jitter; ``httpx`` gives us timeouts and connection
pooling. Every provider composes this instead of calling ``httpx`` directly,
so retry/timeout policy lives in exactly one place.
"""

from __future__ import annotations

from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from autopredict.core.exceptions import DataIngestionError
from autopredict.monitoring import get_logger

logger = get_logger(__name__)

_RETRYABLE = (httpx.TransportError, httpx.HTTPStatusError)


class HttpClient:
    """Thin wrapper around ``httpx.Client`` with retries and JSON parsing."""

    def __init__(self, base_url: str, *, timeout: float = 15.0, max_retries: int = 4) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=timeout,
            headers={"Accept": "application/json", "User-Agent": "autopredict/0.1"},
        )

    def get_json(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        """GET ``path`` and return parsed JSON, retrying transient failures."""

        @retry(
            reraise=True,
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential_jitter(initial=1, max=20),
            retry=retry_if_exception_type(_RETRYABLE),
        )
        def _do_get() -> Any:
            response = self._client.get(path, params=params)
            response.raise_for_status()
            return response.json()

        try:
            return _do_get()
        except httpx.HTTPError as exc:  # pragma: no cover - network path
            logger.error("http_request_failed", url=f"{self._base_url}{path}", error=str(exc))
            raise DataIngestionError(f"GET {self._base_url}{path} failed: {exc}") from exc

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
