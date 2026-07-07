"""Application settings.

Two layers, deliberately separated:

* **Secrets & environment** come from ``.env`` / real env vars via
  ``pydantic-settings`` (prefix ``APP_``). These must never be committed.
* **Non-secret, versioned config** (indicator windows, gate thresholds, asset
  metadata) comes from ``configs/*.yaml`` and is loaded on demand.

``get_settings()`` is cached so the environment is parsed once per process.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from autopredict.core.enums import Asset, Environment
from autopredict.core.exceptions import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "configs"


class Settings(BaseSettings):
    """Typed environment-driven settings."""

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # General
    env: Environment = Environment.DEVELOPMENT
    log_level: str = "INFO"
    log_json: bool = False

    # Assets / target
    assets: list[Asset] = Field(default_factory=lambda: [Asset.BTC, Asset.SOL])
    prediction_horizon_hours: int = 4
    ohlcv_interval: str = "1h"

    # Providers
    coingecko_base_url: str = "https://api.coingecko.com/api/v3"
    binance_base_url: str = "https://api.binance.com"
    fear_greed_base_url: str = "https://api.alternative.me"
    alphavantage_api_key: str | None = None
    http_timeout_seconds: float = 15.0
    http_max_retries: int = 4

    # Storage
    s3_endpoint_url: str | None = None
    s3_bucket: str = "autopredict"
    s3_region: str = "us-east-1"

    # MLflow
    mlflow_experiment: str = "autopredict"
    model_registry_stage: str = "Production"

    # Promotion gate (env can override YAML)
    min_roc_auc: float = 0.55
    min_accuracy: float = 0.52
    promotion_improvement: float = 0.005

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    model_dir: Path = PROJECT_ROOT / "models"

    @field_validator("assets", mode="before")
    @classmethod
    def _split_assets(cls, value: Any) -> Any:
        """Allow ``APP_ASSETS=BTC,SOL`` comma-separated strings."""
        if isinstance(value, str):
            return [item.strip().upper() for item in value.split(",") if item.strip()]
        return value

    @property
    def is_production(self) -> bool:
        return self.env is Environment.PRODUCTION

    def load_yaml_config(self, name: str = "config.yaml") -> dict[str, Any]:
        """Load and parse a versioned YAML config file from ``configs/``."""
        path = CONFIG_DIR / name
        if not path.exists():
            raise ConfigurationError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise ConfigurationError(f"Config {name} must be a mapping, got {type(data)!r}")
        return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
