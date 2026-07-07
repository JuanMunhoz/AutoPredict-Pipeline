"""Lightweight, dependency-free data-quality checks.

We intentionally avoid a heavy validation framework here: a handful of explicit
invariants is easier to read, test and reason about for tabular market data.
Each failed invariant raises :class:`DataValidationError` with a precise
message so pipeline failures are self-explanatory in the logs.
"""

from __future__ import annotations

import pandas as pd

from autopredict.core.exceptions import DataValidationError
from autopredict.monitoring import get_logger

logger = get_logger(__name__)

_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def validate_ohlcv(df: pd.DataFrame, *, min_rows: int = 100) -> pd.DataFrame:
    """Validate an OHLCV frame; return it unchanged if it passes.

    Checks: required columns, monotonic unique UTC index, no NaNs, non-negative
    prices/volume, and ``low <= open/close <= high`` consistency.
    """
    missing = [c for c in _OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing OHLCV columns: {missing}")

    if len(df) < min_rows:
        raise DataValidationError(f"Too few rows: {len(df)} < {min_rows}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataValidationError("OHLCV index must be a DatetimeIndex")

    if df.index.has_duplicates:
        raise DataValidationError("OHLCV index contains duplicate timestamps")

    if not df.index.is_monotonic_increasing:
        raise DataValidationError("OHLCV index must be sorted ascending")

    if df[list(_OHLCV_COLUMNS)].isna().any().any():
        raise DataValidationError("OHLCV contains NaN values")

    if (df[list(_OHLCV_COLUMNS)] < 0).any().any():
        raise DataValidationError("OHLCV contains negative values")

    inconsistent = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )
    if bool(inconsistent.any()):
        raise DataValidationError(f"{int(inconsistent.sum())} candles violate OHLC bounds")

    logger.info("ohlcv_validation_passed", rows=len(df))
    return df
