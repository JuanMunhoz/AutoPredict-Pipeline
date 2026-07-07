"""CoinGecko adapter — secondary/fallback OHLC source and enrichment.

The free ``/coins/{id}/ohlc`` endpoint returns coarse candles for a bounded
window. We keep it as a fallback provider and as a cross-source sanity check
against Binance.
"""

from __future__ import annotations

import pandas as pd

from autopredict.core.enums import Asset
from autopredict.core.exceptions import DataIngestionError
from autopredict.ingestion.providers.http import HttpClient
from autopredict.monitoring import get_logger

logger = get_logger(__name__)
_COLUMNS = ["open", "high", "low", "close"]


class CoinGeckoProvider:
    """Implements :class:`autopredict.core.ports.MarketDataProvider` (close-only volume=0)."""

    def __init__(self, base_url: str, id_map: dict[Asset, str], **client_kwargs: object) -> None:
        self._id_map = id_map
        self._client = HttpClient(base_url, **client_kwargs)  # type: ignore[arg-type]

    def fetch_ohlcv(self, asset: Asset, interval: str, history_days: int) -> pd.DataFrame:
        coin_id = self._id_map.get(asset)
        if coin_id is None:
            raise DataIngestionError(f"No CoinGecko id mapped for {asset}")

        # CoinGecko caps free OHLC history; clamp the request window.
        days = min(history_days, 90)
        payload = self._client.get_json(
            f"/coins/{coin_id}/ohlc",
            params={"vs_currency": "usd", "days": days},
        )
        if not payload:
            raise DataIngestionError(f"CoinGecko returned no OHLC for {coin_id}")

        frame = pd.DataFrame(payload, columns=["ts", *_COLUMNS])
        frame["timestamp"] = pd.to_datetime(frame["ts"], unit="ms", utc=True)
        frame["volume"] = 0.0  # not provided by this endpoint
        frame = (
            frame[["timestamp", *_COLUMNS, "volume"]]
            .drop_duplicates(subset="timestamp")
            .set_index("timestamp")
            .sort_index()
        )
        logger.info("coingecko_ohlc_fetched", asset=asset.value, candles=len(frame))
        return frame
