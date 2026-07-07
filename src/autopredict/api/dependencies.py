"""Dependency wiring for the API (composition root for inference).

Loads every configured asset's predictor once at startup into a registry, then
hands predictors to routes via FastAPI ``Depends``. This keeps route handlers
free of construction logic and makes them trivially testable with overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from autopredict.config import Settings, get_settings
from autopredict.core.enums import Asset
from autopredict.core.exceptions import ModelNotFoundError
from autopredict.deployment import ModelStore, SklearnPredictor
from autopredict.monitoring import get_logger

logger = get_logger(__name__)


@dataclass
class PredictorRegistry:
    """In-memory map of asset -> loaded predictor."""

    predictors: dict[Asset, SklearnPredictor] = field(default_factory=dict)

    def get(self, asset: Asset) -> SklearnPredictor:
        predictor = self.predictors.get(asset)
        if predictor is None:
            raise ModelNotFoundError(f"No model loaded for {asset.value}")
        return predictor

    @property
    def loaded_assets(self) -> list[str]:
        return [a.value for a in self.predictors]


def build_registry(settings: Settings | None = None) -> PredictorRegistry:
    """Load predictors for every configured asset (best-effort per asset)."""
    settings = settings or get_settings()
    store = ModelStore(settings.model_dir)
    registry = PredictorRegistry()
    for asset in settings.assets:
        try:
            registry.predictors[asset] = store.load(asset)
            logger.info("predictor_loaded", asset=asset.value)
        except ModelNotFoundError as exc:
            logger.warning("predictor_missing", asset=asset.value, detail=str(exc))
    return registry
