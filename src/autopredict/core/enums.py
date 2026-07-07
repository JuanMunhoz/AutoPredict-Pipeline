"""Enumerations shared across the domain."""

from __future__ import annotations

from enum import Enum, StrEnum


class Asset(StrEnum):
    """Supported crypto assets."""

    BTC = "BTC"
    SOL = "SOL"


class Direction(int, Enum):
    """Prediction target: price direction over the forecast horizon."""

    DOWN = 0
    UP = 1

    @classmethod
    def from_return(cls, forward_return: float, deadband: float = 0.0) -> Direction:
        """Map a forward return to a class label given a symmetric deadband."""
        return cls.UP if forward_return > deadband else cls.DOWN


class Environment(StrEnum):
    """Runtime environment."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class ModelStage(StrEnum):
    """MLflow-style registry stages."""

    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"
