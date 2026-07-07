"""Servable predictor wrapping a fitted estimator.

Implements the ``Predictor`` port. It enforces the feature contract: incoming
feature mappings are projected onto the exact training feature order, missing
features raise, and the output is a validated domain ``Prediction``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from autopredict.core.entities import Prediction
from autopredict.core.enums import Asset, Direction
from autopredict.core.exceptions import FeatureEngineeringError


class SklearnPredictor:
    """Adapt any sklearn-compatible classifier to the ``Predictor`` port."""

    def __init__(
        self,
        estimator: Any,
        feature_names: list[str],
        version: str,
        horizon_hours: int = 4,
    ) -> None:
        self._estimator = estimator
        self.feature_names = feature_names
        self.version = version
        self._horizon = horizon_hours

    def predict(self, asset: Asset, features: dict[str, float]) -> Prediction:
        """Produce a :class:`Prediction` from a feature mapping."""
        missing = [f for f in self.feature_names if f not in features]
        if missing:
            raise FeatureEngineeringError(f"Missing features for inference: {missing}")

        row = pd.DataFrame([[features[f] for f in self.feature_names]], columns=self.feature_names)
        proba_up = float(np.asarray(self._estimator.predict_proba(row))[0, 1])
        direction = Direction.UP if proba_up >= 0.5 else Direction.DOWN
        return Prediction(
            asset=asset,
            direction=direction,
            probability_up=proba_up,
            model_version=self.version,
            horizon_hours=self._horizon,
        )
