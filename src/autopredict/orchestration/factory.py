"""Composition helpers that build concrete adapters from settings + YAML.

This is the only place that knows how to wire providers, repositories and the
registry together. Keeping construction here (and out of use-cases) means the
business logic never depends on concrete adapters — swap Binance for another
exchange by editing one function.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autopredict.config import Settings
from autopredict.core.enums import Asset
from autopredict.ingestion import DataCollector
from autopredict.ingestion.providers import (
    BinanceProvider,
    FearGreedProvider,
)
from autopredict.ingestion.repository import ParquetDatasetRepository

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"


def _asset_maps(yaml_cfg: dict[str, Any]) -> tuple[dict[Asset, str], dict[Asset, str]]:
    """Return (binance_symbol_map, coingecko_id_map) from the YAML asset table."""
    binance: dict[Asset, str] = {}
    coingecko: dict[Asset, str] = {}
    for entry in yaml_cfg.get("assets", []):
        asset = Asset(entry["symbol"])
        binance[asset] = entry["binance_symbol"]
        coingecko[asset] = entry["coingecko_id"]
    return binance, coingecko


def build_collector(settings: Settings, yaml_cfg: dict[str, Any]) -> DataCollector:
    """Wire the ingestion use-case with Binance + Fear&Greed adapters."""
    binance_map, _ = _asset_maps(yaml_cfg)
    client_kwargs = {
        "timeout": settings.http_timeout_seconds,
        "max_retries": settings.http_max_retries,
    }
    market = BinanceProvider(settings.binance_base_url, binance_map, **client_kwargs)
    sentiment = (
        FearGreedProvider(settings.fear_greed_base_url, **client_kwargs)
        if yaml_cfg.get("ingestion", {}).get("include_fear_greed", True)
        else None
    )
    repo = ParquetDatasetRepository(DATA_DIR / "raw")
    return DataCollector(market, repo, sentiment)


def raw_repository() -> ParquetDatasetRepository:
    return ParquetDatasetRepository(DATA_DIR / "raw")


def processed_repository() -> ParquetDatasetRepository:
    return ParquetDatasetRepository(DATA_DIR / "processed")
