"""The training use-case.

``ModelTrainer`` builds an estimator, fits it on a chronological training split,
evaluates it on the holdout, and returns a domain ``TrainedModel``. It is pure
w.r.t. tracking/registry concerns — logging to MLflow and promotion are handled
by callers, keeping this class single-responsibility and easy to unit-test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from autopredict.core.entities import TrainedModel
from autopredict.core.enums import Asset
from autopredict.core.exceptions import ModelTrainingError
from autopredict.evaluation.metrics import compute_metrics
from autopredict.monitoring import get_logger
from autopredict.training.dataset import chronological_split, walk_forward_splitter
from autopredict.training.model_factory import build_estimator

logger = get_logger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameters not tied to a specific estimator."""

    test_size: float = 0.2
    n_splits: int = 5
    random_state: int = 42


class ModelTrainer:
    """Fit and evaluate a single-asset direction classifier."""

    def __init__(self, model_config: dict[str, Any], training_config: TrainingConfig) -> None:
        self._model_config = model_config
        self._cfg = training_config

    def train(self, asset: Asset, x: pd.DataFrame, y: pd.Series) -> TrainedModel:
        """Train on a chronological split and evaluate on the holdout."""
        if len(x) != len(y):
            raise ModelTrainingError("X and y have mismatched lengths")
        if y.nunique() < 2:
            raise ModelTrainingError("Target has a single class; cannot train a classifier")

        x_train, x_test, y_train, y_test = chronological_split(x, y, self._cfg.test_size)
        estimator, resolved_params = build_estimator(self._model_config)

        logger.info(
            "training_started",
            asset=asset.value,
            estimator=resolved_params.get("estimator"),
            train_rows=len(x_train),
            test_rows=len(x_test),
        )
        estimator.fit(x_train, y_train)

        proba = _predict_proba_up(estimator, x_test)
        metrics = compute_metrics(y_test.to_numpy(), proba)

        cv_auc = self._cross_validate(estimator, x_train, y_train)
        logger.info(
            "training_completed",
            asset=asset.value,
            holdout_auc=metrics.roc_auc,
            holdout_acc=metrics.accuracy,
            cv_mean_auc=cv_auc,
        )
        return TrainedModel(
            asset=asset,
            estimator=estimator,
            feature_names=list(x.columns),
            metrics=metrics,
            params={**resolved_params, "cv_mean_auc": cv_auc},
        )

    def _cross_validate(self, estimator: Any, x: pd.DataFrame, y: pd.Series) -> float:
        """Walk-forward CV mean AUC on the training portion (for logging)."""
        from sklearn.base import clone

        splitter = walk_forward_splitter(self._cfg.n_splits)
        aucs: list[float] = []
        for train_idx, val_idx in splitter.split(x):
            model = clone(estimator)
            model.fit(x.iloc[train_idx], y.iloc[train_idx])
            proba = _predict_proba_up(model, x.iloc[val_idx])
            fold = compute_metrics(y.iloc[val_idx].to_numpy(), proba)
            if not np.isnan(fold.roc_auc):
                aucs.append(fold.roc_auc)
        return float(np.mean(aucs)) if aucs else float("nan")


def _predict_proba_up(estimator: Any, x: pd.DataFrame) -> np.ndarray:
    """Return P(class == UP) as a 1-D array."""
    proba = estimator.predict_proba(x)
    return np.asarray(proba)[:, 1]
