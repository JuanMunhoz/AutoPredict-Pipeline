"""Estimator factory (Factory pattern).

Maps the ``estimator`` key in ``configs/model.yaml`` to a fitted-ready
sklearn-compatible pipeline. Adding a new model means adding one builder here —
the rest of the training code is agnostic to the concrete estimator.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from autopredict.core.exceptions import ModelTrainingError


def _build_lightgbm(params: dict[str, Any]) -> BaseEstimator:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(**params, verbose=-1)


def _build_xgboost(params: dict[str, Any]) -> BaseEstimator:
    from xgboost import XGBClassifier

    return XGBClassifier(**params, use_label_encoder=False)


def _build_logreg(params: dict[str, Any]) -> BaseEstimator:
    # Linear models need scaling; wrap in a pipeline so callers stay uniform.
    return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(**params))])


_BUILDERS: dict[str, Callable[[dict[str, Any]], BaseEstimator]] = {
    "lightgbm": _build_lightgbm,
    "xgboost": _build_xgboost,
    "logistic_regression": _build_logreg,
}


def build_estimator(model_config: dict[str, Any]) -> tuple[BaseEstimator, dict[str, Any]]:
    """Return ``(estimator, resolved_params)`` from a parsed model config."""
    name = model_config.get("estimator", "lightgbm")
    if name not in _BUILDERS:
        raise ModelTrainingError(f"Unknown estimator '{name}'. Available: {sorted(_BUILDERS)}")
    params = dict(model_config.get(name, {}))
    estimator = _BUILDERS[name](params)
    return estimator, {"estimator": name, **params}
