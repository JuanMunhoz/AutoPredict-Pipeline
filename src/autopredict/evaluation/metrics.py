"""Classification metrics computed into a domain ``ModelMetrics`` object."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from autopredict.core.entities import ModelMetrics


def compute_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> ModelMetrics:
    """Compute the full metric set from ground truth and P(UP)."""
    y_pred = (y_proba >= threshold).astype(int)
    # roc_auc / log_loss are undefined if only one class is present in y_true.
    single_class = len(np.unique(y_true)) < 2
    return ModelMetrics(
        roc_auc=float("nan") if single_class else float(roc_auc_score(y_true, y_proba)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        log_loss=float("nan") if single_class else float(log_loss(y_true, y_proba)),
        n_samples=int(len(y_true)),
    )
