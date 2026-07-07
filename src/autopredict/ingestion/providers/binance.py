"""Binance public REST adapter — primary OHLCV source.

Uses the ``/api/v3/klines`` endpoint, which is free and keyless. Binance caps
each response at 1000 candles, so we page backwards from *now* until we have
enough history.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from autopredict.core.enums import Asset
from autopredict.core.exceptions import DataIngestionError
from autopredict.ingestion.providers.http import HttpClient
from autopredict.monitoring import get_logger

logger = get_logger(__name__)

# Milliseconds per candle for the intervals we support.
_INTERVAL_MS: dict[str, int] = {
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}
_MAX_LIMIT = 1000
_COLUMNS = ["open", "high", "low", "close", "volume"]


class BinanceProvider:
    """Implements :class:`autopredict.core.ports.MarketDataProvider`."""

    def __init__(self, base_url: str, symbol_map: dict[Asset, str], **client_kwargs: object) -> None:
        self._symbol_map = symbol_map
        self._client = HttpClient(base_url, **client_kwargs)  # type: ignore[arg-type]

    def fetch_ohlcv(self, asset: Asset, interval: str, history_days: int) -> pd.DataFrame:
        """Return an OHLCV DataFrame indexed by UTC timestamp (ascending)."""
        if interval not in _INTERVAL_MS:
            raise DataIngestionError(f"Unsupported interval: {interval}")
        symbol = self._symbol_map.get(asset)
        if symbol is None:
            raise DataIngestionError(f"No Binance symbol mapped for {asset}")

        step_ms = _INTERVAL_MS[interval]
        end_ms = int(datetime.now(tz=UTC).timestamp() * 1000)
        start_ms = int((datetime.now(tz=UTC) - timedelta(days=history_days)).timestamp() * 1000)

        rows: list[list[float]] = []
        cursor = start_ms
        while cursor < end_ms:
            batch = self._client.get_json(
                "/api/v3/klines",
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": cursor,
                    "limit": _MAX_LIMIT,
                },
            )
            if not batch:
                break
            rows.extend(batch)
            cursor = int(batch[-1][0]) + step_ms
            if len(batch) < _MAX_LIMIT:
                break

        if not rows:
            raise DataIngestionError(f"Binance returned no candles for {symbol}")

        frame = self._to_frame(rows)
        logger.info(
            "binance_ohlcv_fetched",
            asset=asset.value,
            symbol=symbol,
            candles=len(frame),
            interval=interval,
        )
        return frame

    @staticmethod
    def _to_frame(rows: list[list[float]]) -> pd.DataFrame:
        frame = pd.DataFrame(
            rows,
            columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades",
                "taker_base", "taker_quote", "ignore",
            ],
        )
        frame["timestamp"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
        frame = frame[["timestamp", *_COLUMNS]].astype({c: "float64" for c in _COLUMNS})
        frame = frame.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()
        return frame
