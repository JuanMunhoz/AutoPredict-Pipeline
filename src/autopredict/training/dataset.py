"""Time-aware dataset splitting.

Financial series are autocorrelated, so a random split leaks the future into
training. We therefore use a strictly chronological holdout and a walk-forward
cross-validation splitter (sklearn's ``TimeSeriesSplit``).
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from autopredict.core.exceptions import ModelTrainingError


def chronological_split(
    x: pd.DataFrame, y: pd.Series, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split by time: the last ``test_size`` fraction becomes the holdout."""
    if not 0.0 < test_size < 1.0:
        raise ModelTrainingError(f"test_size must be in (0, 1), got {test_size}")
    n = len(x)
    split_idx = int(n * (1.0 - test_size))
    if split_idx <= 0 or split_idx >= n:
        raise ModelTrainingError("Not enough rows for the requested split")
    return (
        x.iloc[:split_idx],
        x.iloc[split_idx:],
        y.iloc[:split_idx],
        y.iloc[split_idx:],
    )


def walk_forward_splitter(n_splits: int = 5) -> TimeSeriesSplit:
    """Return a walk-forward CV splitter for hyperparameter validation."""
    return TimeSeriesSplit(n_splits=n_splits)
