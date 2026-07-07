"""The ingestion use-case.

``DataCollector`` orchestrates providers (via ports), validates the result and
persists raw datasets per asset. It depends only on abstractions, so in tests
we inject fakes and in production we inject Binance + Fear&Greed adapters.
"""

from __future__ import annotations

import pandas as pd

from autopredict.core.enums import Asset
from autopredict.core.exceptions import DataIngestionError
from autopredict.core.ports import (
    DatasetRepository,
    MarketDataProvider,
    SentimentProvider,
)
from autopredict.ingestion.validation import validate_ohlcv
from autopredict.monitoring import get_logger

logger = get_logger(__name__)


class DataCollector:
    """Collect, validate and persist raw market + sentiment data per asset."""

    def __init__(
        self,
        market_provider: MarketDataProvider,
        repository: DatasetRepository,
        sentiment_provider: SentimentProvider | None = None,
    ) -> None:
        self._market = market_provider
        self._repo = repository
        self._sentiment = sentiment_provider

    def collect(self, asset: Asset, interval: str, history_days: int) -> pd.DataFrame:
        """Fetch, validate and persist raw data for a single asset."""
        logger.info("ingestion_started", asset=asset.value, history_days=history_days)

        ohlcv = self._market.fetch_ohlcv(asset, interval=interval, history_days=history_days)
        ohlcv = validate_ohlcv(ohlcv)

        if self._sentiment is not None:
            ohlcv = self._attach_sentiment(ohlcv, history_days)

        self._repo.save(ohlcv, name=f"raw_{asset.value.lower()}")
        logger.info("ingestion_completed", asset=asset.value, rows=len(ohlcv))
        return ohlcv

    def collect_all(
        self, assets: list[Asset], interval: str, history_days: int
    ) -> dict[Asset, pd.DataFrame]:
        """Collect every asset, isolating per-asset failures."""
        results: dict[Asset, pd.DataFrame] = {}
        errors: dict[Asset, str] = {}
        for asset in assets:
            try:
                results[asset] = self.collect(asset, interval, history_days)
            except DataIngestionError as exc:
                errors[asset] = str(exc)
                logger.error("ingestion_failed", asset=asset.value, error=str(exc))
        if not results:
            raise DataIngestionError(f"All ingestion attempts failed: {errors}")
        return results

    def _attach_sentiment(self, ohlcv: pd.DataFrame, history_days: int) -> pd.DataFrame:
        assert self._sentiment is not None
        try:
            sentiment = self._sentiment.fetch_sentiment(history_days)
        except DataIngestionError as exc:  # sentiment is enrichment, not critical
            logger.warning("sentiment_unavailable", error=str(exc))
            return ohlcv
        # Daily sentiment forward-filled onto the hourly grid.
        merged = ohlcv.join(sentiment.reindex(ohlcv.index, method="ffill"))
        return merged
