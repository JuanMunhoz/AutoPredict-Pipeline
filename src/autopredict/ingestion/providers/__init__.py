"""Concrete data-provider adapters implementing the domain ports."""

from autopredict.ingestion.providers.binance import BinanceProvider
from autopredict.ingestion.providers.coingecko import CoinGeckoProvider
from autopredict.ingestion.providers.fear_greed import FearGreedProvider

__all__ = ["BinanceProvider", "CoinGeckoProvider", "FearGreedProvider"]
