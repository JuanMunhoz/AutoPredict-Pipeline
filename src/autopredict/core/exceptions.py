"""Domain-specific exception hierarchy.

Keeping a single root (`AutoPredictError`) lets callers catch everything the
application raises without also swallowing unrelated stdlib errors.
"""

from __future__ import annotations


class AutoPredictError(Exception):
    """Base class for all application errors."""


class ConfigurationError(AutoPredictError):
    """Invalid or missing configuration."""


class DataIngestionError(AutoPredictError):
    """A data provider failed to return usable data."""


class DataValidationError(AutoPredictError):
    """Ingested or engineered data failed a quality/schema check."""


class FeatureEngineeringError(AutoPredictError):
    """Feature construction failed."""


class ModelTrainingError(AutoPredictError):
    """Training could not complete."""


class ModelNotFoundError(AutoPredictError):
    """No model artifact available to serve or evaluate."""


class PromotionRejectedError(AutoPredictError):
    """A newly trained model failed the promotion gate."""
