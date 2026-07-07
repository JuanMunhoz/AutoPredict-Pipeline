"""Crypto Fear & Greed Index adapter (alternative.me).

Free, keyless, daily-granularity market sentiment. We forward-fill it onto the
hourly OHLCV grid during feature engineering.
"""

from __future__ import annotations

import pandas as pd

from autopredict.core.exceptions import DataIngestionError
from autopredict.ingestion.providers.http import HttpClient
from autopredict.monitoring import get_logger

logger = get_logger(__name__)


class FearGreedProvider:
    """Implements :class:`autopredict.core.ports.SentimentProvider`."""

    def __init__(self, base_url: str, **client_kwargs: object) -> None:
        self._client = HttpClient(base_url, **client_kwargs)  # type: ignore[arg-type]

    def fetch_sentiment(self, history_days: int) -> pd.DataFrame:
        payload = self._client.get_json("/fng/", params={"limit": history_days, "format": "json"})
        data = payload.get("data") if isinstance(payload, dict) else None
        if not data:
            raise DataIngestionError("Fear & Greed API returned no data")

        frame = pd.DataFrame(data)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"].astype(int), unit="s", utc=True)
        frame["fear_greed_value"] = frame["value"].astype(int)
        frame = (
            frame[["timestamp", "fear_greed_value", "value_classification"]]
            .rename(columns={"value_classification": "fear_greed_class"})
            .drop_duplicates(subset="timestamp")
            .set_index("timestamp")
            .sort_index()
        )
        logger.info("fear_greed_fetched", records=len(frame))
        return frame
