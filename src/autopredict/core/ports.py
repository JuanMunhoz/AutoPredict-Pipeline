"""Ports — abstract boundaries the domain depends on.

Adapters (CoinGecko client, MLflow registry, S3 storage, ...) implement these
Protocols. Use-cases depend on the Protocol, never the concrete class, so any
implementation can be swapped or mocked in tests (Dependency Inversion).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from autopredict.core.entities import Prediction, TrainedModel
from autopredict.core.enums import Asset, ModelStage


@runtime_checkable
class MarketDataProvider(Protocol):
    """Source of OHLCV candle data for an asset."""

    def fetch_ohlcv(self, asset: Asset, interval: str, history_days: int) -> pd.DataFrame:
        """Return a DataFrame indexed by UTC timestamp with OHLCV columns."""
        ...


@runtime_checkable
class SentimentProvider(Protocol):
    """Source of market sentiment (e.g. Fear & Greed Index)."""

    def fetch_sentiment(self, history_days: int) -> pd.DataFrame:
        """Return a DataFrame indexed by UTC timestamp with a sentiment column."""
        ...


@runtime_checkable
class DatasetRepository(Protocol):
    """Persistence for tabular datasets (raw candles, feature matrices)."""

    def save(self, df: pd.DataFrame, name: str) -> str:
        """Persist a DataFrame and return its resolved location."""
        ...

    def load(self, name: str) -> pd.DataFrame:
        """Load a previously saved DataFrame."""
        ...


@runtime_checkable
class ModelRegistry(Protocol):
    """Abstraction over an experiment tracker + model registry (e.g. MLflow)."""

    def log_candidate(self, model: TrainedModel, run_name: str) -> str:
        """Register a candidate and return its version identifier."""
        ...

    def get_production_metrics(self, asset: Asset) -> dict[str, float] | None:
        """Return metrics of the current Production model, or None if absent."""
        ...

    def promote(self, asset: Asset, version: str, stage: ModelStage) -> None:
        """Transition a model version to the given stage."""
        ...


@runtime_checkable
class ModelLoader(Protocol):
    """Loads a servable model for inference."""

    def load(self, asset: Asset) -> Predictor:
        """Return a ready-to-serve predictor for the asset."""
        ...


@runtime_checkable
class Predictor(Protocol):
    """A callable inference object."""

    version: str
    feature_names: list[str]

    def predict(self, asset: Asset, features: dict[str, float]) -> Prediction:
        """Produce a prediction from a feature mapping."""
        ...
