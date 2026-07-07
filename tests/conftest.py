"""Shared pytest fixtures.

We synthesise a realistic-but-deterministic OHLCV series (geometric random walk)
so unit tests never touch the network and always produce the same result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """~800 hourly candles from a seeded geometric random walk."""
    n = 800
    rng = np.random.default_rng(7)
    index = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    returns = rng.normal(0, 0.01, size=n)
    close = 30_000 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    # Keep OHLC bounds consistent.
    high = np.maximum.reduce([high, open_, close])
    low = np.minimum.reduce([low, open_, close])
    volume = np.abs(rng.normal(1000, 200, n))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    ).rename_axis("timestamp")
