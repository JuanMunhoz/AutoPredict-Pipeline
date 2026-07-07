"""MLflow adapter implementing the ``ModelRegistry`` port.

Wraps experiment tracking (params/metrics/artifacts) and the model registry
(stage transitions) behind our domain interface, so the training use-case never
imports MLflow directly. ``mlflow`` is an optional dependency; importing this
module without it installed raises a clear error only when used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from autopredict.core.entities import TrainedModel
from autopredict.core.enums import Asset, ModelStage
from autopredict.monitoring import get_logger

if TYPE_CHECKING:  # pragma: no cover
    pass

logger = get_logger(__name__)


class MLflowModelRegistry:
    """Experiment tracker + model registry backed by MLflow."""

    def __init__(self, experiment: str, tracking_uri: str | None = None) -> None:
        import mlflow

        self._mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        self._client = mlflow.tracking.MlflowClient()

    def _registered_name(self, asset: Asset) -> str:
        return f"autopredict-{asset.value.lower()}"

    def log_candidate(self, model: TrainedModel, run_name: str) -> str:
        """Log params/metrics + the model artifact; register a new version."""
        import mlflow

        name = self._registered_name(model.asset)
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params({k: v for k, v in model.params.items()})
            mlflow.log_metrics(model.metrics.as_dict())
            mlflow.set_tags(
                {"asset": model.asset.value, "feature_count": len(model.feature_names)}
            )
            info = mlflow.sklearn.log_model(
                sk_model=model.estimator,
                artifact_path="model",
                registered_model_name=name,
            )
            version = self._resolve_version(name, run.info.run_id, info)
            logger.info("candidate_logged", asset=model.asset.value, version=version)
            return version

    def get_production_metrics(self, asset: Asset) -> dict[str, float] | None:
        """Return the Production model's metrics, or ``None`` if none exists."""
        name = self._registered_name(asset)
        try:
            versions = self._client.get_latest_versions(name, stages=[ModelStage.PRODUCTION.value])
        except Exception as exc:  # noqa: BLE001 - registry may not exist yet
            logger.info("no_production_model", asset=asset.value, detail=str(exc))
            return None
        if not versions:
            return None
        run = self._client.get_run(versions[0].run_id)
        return dict(run.data.metrics)

    def promote(self, asset: Asset, version: str, stage: ModelStage) -> None:
        """Transition a version to a stage, archiving prior Production models."""
        name = self._registered_name(asset)
        self._client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage.value,
            archive_existing_versions=(stage is ModelStage.PRODUCTION),
        )
        logger.info("model_promoted", asset=asset.value, version=version, stage=stage.value)

    def _resolve_version(self, name: str, run_id: str, info: object) -> str:
        # model_info may carry the registered version directly on newer MLflow.
        version = getattr(info, "registered_model_version", None)
        if version is not None:
            return str(version)
        versions = self._client.search_model_versions(f"run_id='{run_id}'")
        return str(versions[0].version) if versions else "1"
