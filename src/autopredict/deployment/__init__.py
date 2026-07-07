"""Deployment: local model artifact store and the servable predictor."""

from autopredict.deployment.model_store import ModelStore
from autopredict.deployment.predictor import SklearnPredictor

__all__ = ["ModelStore", "SklearnPredictor"]
