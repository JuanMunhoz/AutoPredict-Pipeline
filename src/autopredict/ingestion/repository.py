"""Parquet-backed dataset repository.

Implements :class:`autopredict.core.ports.DatasetRepository`. Files land under
``data/`` which is tracked by **DVC**, not git — so large datasets are
versioned and pushed to the S3-compatible remote (Supabase) while git stays
lean. Parquet keeps dtypes and the datetime index intact.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from autopredict.core.exceptions import DataValidationError
from autopredict.monitoring import get_logger

logger = get_logger(__name__)


class ParquetDatasetRepository:
    """Store/load DataFrames as Parquet under a base directory."""

    def __init__(self, base_dir: Path | str) -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self._base / f"{name}.parquet"

    def save(self, df: pd.DataFrame, name: str) -> str:
        path = self._path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, engine="pyarrow", index=True)
        logger.info("dataset_saved", name=name, path=str(path), rows=len(df))
        return str(path)

    def load(self, name: str) -> pd.DataFrame:
        path = self._path(name)
        if not path.exists():
            raise DataValidationError(f"Dataset not found: {path}")
        df = pd.read_parquet(path, engine="pyarrow")
        logger.info("dataset_loaded", name=name, rows=len(df))
        return df

    def exists(self, name: str) -> bool:
        return self._path(name).exists()
