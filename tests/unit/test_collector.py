"""Unit tests for the ingestion use-case using in-memory fakes (ports)."""

from __future__ import annotations

import pandas as pd
import pytest

from autopredict.core.enums import Asset
from autopredict.core.exceptions import DataIngestionError
from autopredict.ingestion import DataCollector


class FakeMarketProvider:
    def __init__(self, frame: pd.DataFrame, fail_for: set[Asset] | None = None) -> None:
        self._frame = frame
        self._fail_for = fail_for or set()

    def fetch_ohlcv(self, asset: Asset, interval: str, history_days: int) -> pd.DataFrame:
        if asset in self._fail_for:
            raise DataIngestionError(f"boom for {asset}")
        return self._frame.copy()


class InMemoryRepo:
    def __init__(self) -> None:
        self.saved: dict[str, pd.DataFrame] = {}

    def save(self, df: pd.DataFrame, name: str) -> str:
        self.saved[name] = df
        return name

    def load(self, name: str) -> pd.DataFrame:
        return self.saved[name]


def test_collect_persists_validated_data(synthetic_ohlcv: pd.DataFrame) -> None:
    repo = InMemoryRepo()
    collector = DataCollector(FakeMarketProvider(synthetic_ohlcv), repo)
    result = collector.collect(Asset.BTC, "1h", 30)
    assert "raw_btc" in repo.saved
    assert len(result) == len(synthetic_ohlcv)


def test_collect_all_isolates_failures(synthetic_ohlcv: pd.DataFrame) -> None:
    repo = InMemoryRepo()
    provider = FakeMarketProvider(synthetic_ohlcv, fail_for={Asset.SOL})
    collector = DataCollector(provider, repo)
    results = collector.collect_all([Asset.BTC, Asset.SOL], "1h", 30)
    assert Asset.BTC in results
    assert Asset.SOL not in results


def test_collect_all_raises_when_everything_fails(synthetic_ohlcv: pd.DataFrame) -> None:
    repo = InMemoryRepo()
    provider = FakeMarketProvider(synthetic_ohlcv, fail_for={Asset.BTC, Asset.SOL})
    collector = DataCollector(provider, repo)
    with pytest.raises(DataIngestionError):
        collector.collect_all([Asset.BTC, Asset.SOL], "1h", 30)
