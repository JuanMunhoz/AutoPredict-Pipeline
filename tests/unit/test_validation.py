"""Unit tests for OHLCV data-quality validation."""

from __future__ import annotations

import pandas as pd
import pytest

from autopredict.core.exceptions import DataValidationError
from autopredict.ingestion.validation import validate_ohlcv


def test_valid_frame_passes(synthetic_ohlcv: pd.DataFrame) -> None:
    assert validate_ohlcv(synthetic_ohlcv) is synthetic_ohlcv


def test_missing_column_raises(synthetic_ohlcv: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="Missing OHLCV"):
        validate_ohlcv(synthetic_ohlcv.drop(columns=["volume"]))


def test_negative_value_raises(synthetic_ohlcv: pd.DataFrame) -> None:
    bad = synthetic_ohlcv.copy()
    bad.iloc[0, bad.columns.get_loc("close")] = -1
    with pytest.raises(DataValidationError):
        validate_ohlcv(bad)


def test_duplicate_index_raises(synthetic_ohlcv: pd.DataFrame) -> None:
    bad = pd.concat([synthetic_ohlcv, synthetic_ohlcv.iloc[[0]]])
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_ohlcv(bad)


def test_too_few_rows_raises(synthetic_ohlcv: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="Too few"):
        validate_ohlcv(synthetic_ohlcv.head(10))
