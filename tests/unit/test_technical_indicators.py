"""Unit tests for technical indicators against known-good values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autopredict.features import technical_indicators as ti


def test_sma_matches_manual_mean() -> None:
    s = pd.Series([1, 2, 3, 4, 5], dtype="float64")
    result = ti.sma(s, window=2)
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(1.5)
    assert result.iloc[4] == pytest.approx(4.5)


def test_rsi_bounded_between_0_and_100(synthetic_ohlcv: pd.DataFrame) -> None:
    rsi = ti.rsi(synthetic_ohlcv["close"], period=14).dropna()
    assert rsi.between(0, 100).all()


def test_rsi_all_gains_approaches_100() -> None:
    s = pd.Series(np.arange(1, 50, dtype="float64"))
    rsi = ti.rsi(s, period=14).dropna()
    assert rsi.iloc[-1] > 99


def test_macd_histogram_is_line_minus_signal(synthetic_ohlcv: pd.DataFrame) -> None:
    macd = ti.macd(synthetic_ohlcv["close"]).dropna()
    reconstructed = macd["macd"] - macd["macd_signal"]
    pd.testing.assert_series_equal(macd["macd_hist"], reconstructed, check_names=False)


def test_bollinger_pct_within_bands_mostly(synthetic_ohlcv: pd.DataFrame) -> None:
    bb = ti.bollinger_bands(synthetic_ohlcv["close"]).dropna()
    # %B between 0 and 1 means price sits inside the bands.
    inside = bb["bb_pct"].between(0, 1).mean()
    assert inside > 0.8


def test_atr_is_non_negative(synthetic_ohlcv: pd.DataFrame) -> None:
    atr = ti.atr(synthetic_ohlcv, period=14).dropna()
    assert (atr >= 0).all()
