"""Local model artifact store.

Persists a fitted estimator (``joblib``) alongside a JSON sidecar describing
its feature contract, metrics and version. This is what gets baked into the
Docker image and loaded by the API. It also implements the ``ModelLoader`` port
so the API depends on an abstraction, not on joblib details.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from autopredict.core.entities import TrainedModel
from autopredict.core.enums import Asset
from autopredict.core.exceptions import ModelNotFoundError
from autopredict.deployment.predictor import SklearnPredictor
from autopredict.monitoring import get_logger

logger = get_logger(__name__)


class ModelStore:
    """Read/write model artifacts under ``<base>/<asset>/``."""

    def __init__(self, base_dir: Path | str) -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    def _dir(self, asset: Asset) -> Path:
        return self._base / asset.value.lower()

    def save(self, model: TrainedModel, version: str | None = None) -> str:
        """Persist the estimator + metadata; return the resolved version."""
        version = version or datetime.now(tz=UTC).strftime("%Y%m%d%H%M%S")
        target = self._dir(model.asset)
        target.mkdir(parents=True, exist_ok=True)

        joblib.dump(model.estimator, target / "model.joblib")
        metadata: dict[str, Any] = {
            "asset": model.asset.value,
            "version": version,
            "feature_names": model.feature_names,
            "metrics": model.metrics.as_dict(),
            "params": {k: _jsonable(v) for k, v in model.params.items()},
            "trained_at": model.trained_at.isoformat(),
            "saved_at": datetime.now(tz=UTC).isoformat(),
        }
        (target / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.info("model_saved", asset=model.asset.value, version=version, path=str(target))
        return version

    def load(self, asset: Asset) -> SklearnPredictor:
        """Load the current servable predictor for ``asset`` (ModelLoader port)."""
        target = self._dir(asset)
        model_path = target / "model.joblib"
        meta_path = target / "metadata.json"
        if not model_path.exists() or not meta_path.exists():
            raise ModelNotFoundError(f"No model artifact for {asset.value} in {target}")

        estimator = joblib.load(model_path)
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        return SklearnPredictor(
            estimator=estimator,
            feature_names=metadata["feature_names"],
            version=metadata["version"],
            horizon_hours=int(metadata["params"].get("horizon_hours", 4)),
        )

    def metadata(self, asset: Asset) -> dict[str, Any]:
        meta_path = self._dir(asset) / "metadata.json"
        if not meta_path.exists():
            raise ModelNotFoundError(f"No metadata for {asset.value}")
        return json.loads(meta_path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of param values to JSON-serialisable types."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
