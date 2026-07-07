"""Technical indicators implemented in pure pandas/numpy.

Implementing these ourselves (rather than pulling a TA library) keeps the
dependency surface small, makes the maths auditable, and lets us unit-test each
indicator against hand-computed expectations. All functions are pure and return
a Series/DataFrame aligned to the input index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def returns(close: pd.Series, periods: int = 1) -> pd.Series:
    """Simple percentage return over ``periods`` candles."""
    return close.pct_change(periods=periods)


def log_returns(close: pd.Series, periods: int = 1) -> pd.Series:
    """Log return over ``periods`` candles."""
    return np.log(close / close.shift(periods))


def sma(close: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return close.rolling(window=window, min_periods=window).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return close.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    result = 100.0 - (100.0 / (1.0 + rs))
    # No losses in the window => maximal strength. Preserve NaN during warmup.
    return result.where(avg_loss != 0.0, 100.0).where(avg_loss.notna(), np.nan)


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """Moving Average Convergence Divergence: line, signal and histogram."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": histogram}
    )


def bollinger_bands(
    close: pd.Series, window: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """Bollinger Bands plus the normalised %B position within the bands."""
    mid = sma(close, window)
    std = close.rolling(window=window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    pct_b = (close - lower) / (upper - lower)
    return pd.DataFrame(
        {"bb_upper": upper, "bb_lower": lower, "bb_pct": pct_b, "bb_width": (upper - lower) / mid}
    )


def rolling_volatility(close: pd.Series, window: int = 12) -> pd.Series:
    """Rolling standard deviation of log returns (realised volatility proxy)."""
    return log_returns(close).rolling(window=window, min_periods=window).std()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — volatility that accounts for gaps."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
