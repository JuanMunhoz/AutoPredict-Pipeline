"""Unit tests for the feature pipeline and leakage safety."""

from __future__ import annotations

import pandas as pd

from autopredict.core.enums import Direction
from autopredict.features import FeatureConfig, FeaturePipeline
from autopredict.features.pipeline import TARGET_COLUMN


def _pipeline() -> FeaturePipeline:
    return FeaturePipeline(FeatureConfig(horizon_hours=4))


def test_build_supervised_returns_aligned_xy(synthetic_ohlcv: pd.DataFrame) -> None:
    x, y = _pipeline().build_supervised(synthetic_ohlcv)
    assert len(x) == len(y)
    assert not x.isna().any().any()
    assert set(y.unique()).issubset({0, 1})


def test_label_matches_future_return(synthetic_ohlcv: pd.DataFrame) -> None:
    pipeline = _pipeline()
    label = pipeline.make_label(synthetic_ohlcv).dropna()
    horizon = 4
    close = synthetic_ohlcv["close"]
    expected = (close.shift(-horizon) > close).astype(int).loc[label.index]
    pd.testing.assert_series_equal(
        label.astype(int), expected, check_names=False
    )


def test_no_target_column_leaks_into_features(synthetic_ohlcv: pd.DataFrame) -> None:
    x, _ = _pipeline().build_supervised(synthetic_ohlcv)
    assert TARGET_COLUMN not in x.columns


def test_direction_from_return() -> None:
    assert Direction.from_return(0.01) is Direction.UP
    assert Direction.from_return(-0.01) is Direction.DOWN
    assert Direction.from_return(0.0) is Direction.DOWN
