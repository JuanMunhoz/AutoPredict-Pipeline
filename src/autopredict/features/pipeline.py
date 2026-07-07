"""Feature matrix construction and target labelling.

The pipeline turns a raw OHLCV(+sentiment) frame into a supervised learning
table: engineered features ``X`` plus a binary target ``y`` = direction of the
close ``horizon`` candles into the future.

Leakage safety: the label uses a *future* close, so we drop the last
``horizon`` rows (whose future is unknown). Features only ever use past/current
information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd

from autopredict.core.enums import Direction
from autopredict.core.exceptions import FeatureEngineeringError
from autopredict.features import technical_indicators as ti
from autopredict.monitoring import get_logger

logger = get_logger(__name__)

TARGET_COLUMN = "target_direction"


@dataclass(frozen=True)
class FeatureConfig:
    """Parameters controlling feature construction (loaded from YAML)."""

    rsi_period: int = 14
    ema_windows: tuple[int, ...] = (9, 21, 50)
    sma_windows: tuple[int, ...] = (20, 50, 100)
    macd: dict[str, int] = field(default_factory=lambda: {"fast": 12, "slow": 26, "signal": 9})
    bollinger: dict[str, float] = field(default_factory=lambda: {"window": 20, "num_std": 2})
    return_lags: tuple[int, ...] = (1, 2, 3, 6, 12, 24)
    volatility_windows: tuple[int, ...] = (6, 12, 24)
    horizon_hours: int = 4
    deadband: float = 0.0

    @classmethod
    def from_yaml(cls, cfg: dict[str, Any]) -> FeatureConfig:
        """Build from the parsed ``configs/config.yaml`` mapping."""
        feats = cfg.get("features", {})
        label = cfg.get("label", {})
        return cls(
            rsi_period=feats.get("rsi_period", 14),
            ema_windows=tuple(feats.get("ema_windows", (9, 21, 50))),
            sma_windows=tuple(feats.get("sma_windows", (20, 50, 100))),
            macd=feats.get("macd", {"fast": 12, "slow": 26, "signal": 9}),
            bollinger=feats.get("bollinger", {"window": 20, "num_std": 2}),
            return_lags=tuple(feats.get("return_lags", (1, 2, 3, 6, 12, 24))),
            volatility_windows=tuple(feats.get("volatility_windows", (6, 12, 24))),
            horizon_hours=label.get("horizon_hours", 4),
            deadband=label.get("deadband", 0.0),
        )


class FeaturePipeline:
    """Stateless transformer: raw OHLCV -> (features, target)."""

    def __init__(self, config: FeatureConfig) -> None:
        self._cfg = config

    def build_features(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Engineer the feature columns (no target). Index preserved."""
        if "close" not in raw.columns:
            raise FeatureEngineeringError("Raw frame is missing a 'close' column")

        close = raw["close"]
        out = pd.DataFrame(index=raw.index)

        # Price transforms & momentum
        for lag in self._cfg.return_lags:
            out[f"ret_{lag}"] = ti.returns(close, lag)
            out[f"logret_{lag}"] = ti.log_returns(close, lag)

        for window in self._cfg.sma_windows:
            out[f"sma_{window}_ratio"] = close / ti.sma(close, window) - 1.0
        for span in self._cfg.ema_windows:
            out[f"ema_{span}_ratio"] = close / ti.ema(close, span) - 1.0

        out["rsi"] = ti.rsi(close, self._cfg.rsi_period)

        macd_df = ti.macd(close, **self._cfg.macd)
        out = out.join(macd_df)

        bb_df = ti.bollinger_bands(
            close, int(self._cfg.bollinger["window"]), float(self._cfg.bollinger["num_std"])
        )
        out = out.join(bb_df[["bb_pct", "bb_width"]])

        for window in self._cfg.volatility_windows:
            out[f"vol_{window}"] = ti.rolling_volatility(close, window)

        if {"high", "low", "close"}.issubset(raw.columns):
            out["atr_14"] = ti.atr(raw, 14) / close  # normalised ATR

        # Volume dynamics
        if "volume" in raw.columns and raw["volume"].abs().sum() > 0:
            out["vol_change_1"] = raw["volume"].pct_change()
            out["vol_zscore_24"] = _zscore(raw["volume"], 24)

        # Sentiment enrichment (already forward-filled by the collector)
        if "fear_greed_value" in raw.columns:
            out["fear_greed"] = raw["fear_greed_value"].astype(float)
            out["fear_greed_change"] = raw["fear_greed_value"].astype(float).diff()

        # Calendar features (crypto trades 24/7 but flow differs by hour/day)
        idx = cast("pd.DatetimeIndex", raw.index)
        out["hour"] = idx.hour
        out["dayofweek"] = idx.dayofweek

        return out.replace([np.inf, -np.inf], np.nan)

    def make_label(self, raw: pd.DataFrame) -> pd.Series:
        """Binary target: UP if the close ``horizon`` candles ahead rose."""
        horizon = self._cfg.horizon_hours
        future_return = raw["close"].shift(-horizon) / raw["close"] - 1.0
        label = future_return.apply(
            lambda r: Direction.from_return(r, self._cfg.deadband).value if pd.notna(r) else np.nan
        )
        return cast("pd.Series", label.rename(TARGET_COLUMN))

    def build_supervised(self, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Produce an aligned, leakage-safe (X, y) ready for training."""
        features = self.build_features(raw)
        target = self.make_label(raw)

        dataset = features.join(target).dropna()
        if dataset.empty:
            raise FeatureEngineeringError("Feature matrix is empty after dropping NaNs")

        y = dataset[TARGET_COLUMN].astype(int)
        x = dataset.drop(columns=[TARGET_COLUMN])
        logger.info("features_built", rows=len(x), features=x.shape[1])
        return x, y


def _zscore(series: pd.Series, window: int) -> pd.Series:
    rolling = series.rolling(window=window, min_periods=window)
    return (series - rolling.mean()) / rolling.std()
