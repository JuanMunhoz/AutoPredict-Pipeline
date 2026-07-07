"""End-to-end pipeline steps (the composition root).

Each function is a self-contained stage that the CLI and the Prefect flows call.
They orchestrate use-cases and adapters but contain no business rules of their
own — the rules live in the domain/use-case layers. MLflow is optional: if a
tracking server is unreachable, training still runs and persists locally.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from autopredict.config import Settings, get_settings
from autopredict.core.entities import PromotionDecision
from autopredict.core.enums import Asset, ModelStage
from autopredict.evaluation.gate import GateThresholds, PromotionGate
from autopredict.features import FeatureConfig, FeaturePipeline
from autopredict.monitoring import get_logger
from autopredict.monitoring.drift import DriftMonitor, DriftSummary
from autopredict.orchestration import factory
from autopredict.training import ModelTrainer, TrainingConfig

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Stage 1 — ingestion
# --------------------------------------------------------------------------- #
def run_ingestion(settings: Settings | None = None) -> dict[Asset, pd.DataFrame]:
    """Collect and persist raw data for every configured asset."""
    settings = settings or get_settings()
    yaml_cfg = settings.load_yaml_config()
    collector = factory.build_collector(settings, yaml_cfg)
    history_days = yaml_cfg.get("ingestion", {}).get("history_days", 365)
    return collector.collect_all(settings.assets, settings.ohlcv_interval, history_days)


# --------------------------------------------------------------------------- #
# Stage 2 — feature engineering
# --------------------------------------------------------------------------- #
def run_feature_build(settings: Settings | None = None) -> dict[Asset, tuple[int, int]]:
    """Build and persist the supervised feature matrix for each asset."""
    settings = settings or get_settings()
    yaml_cfg = settings.load_yaml_config()
    fcfg = FeatureConfig.from_yaml(yaml_cfg)
    pipeline = FeaturePipeline(fcfg)
    raw_repo = factory.raw_repository()
    proc_repo = factory.processed_repository()

    shapes: dict[Asset, tuple[int, int]] = {}
    for asset in settings.assets:
        raw = raw_repo.load(f"raw_{asset.value.lower()}")
        x, y = pipeline.build_supervised(raw)
        dataset = x.copy()
        dataset["target_direction"] = y.to_numpy()
        proc_repo.save(dataset, f"features_{asset.value.lower()}")
        shapes[asset] = (x.shape[0], x.shape[1])
        logger.info("feature_build_done", asset=asset.value, rows=x.shape[0], cols=x.shape[1])
    return shapes


# --------------------------------------------------------------------------- #
# Stage 3 — training + evaluation + gated promotion
# --------------------------------------------------------------------------- #
def run_training(settings: Settings | None = None) -> dict[Asset, PromotionDecision]:
    """Train a candidate per asset, evaluate the gate and promote if it passes."""
    settings = settings or get_settings()
    yaml_cfg = settings.load_yaml_config()
    model_cfg = settings.load_yaml_config("model.yaml")

    trainer = ModelTrainer(model_cfg, _training_config(yaml_cfg))
    gate = PromotionGate(_gate_thresholds(settings, yaml_cfg))
    registry = _maybe_registry(settings)
    from autopredict.deployment import ModelStore

    store = ModelStore(settings.model_dir)
    proc_repo = factory.processed_repository()

    decisions: dict[Asset, PromotionDecision] = {}
    for asset in settings.assets:
        dataset = proc_repo.load(f"features_{asset.value.lower()}")
        y = dataset["target_direction"].astype(int)
        x = dataset.drop(columns=["target_direction"])
        # Ensure the label horizon is captured in params for the served model.
        candidate = trainer.train(asset, x, y)
        candidate.params["horizon_hours"] = settings.prediction_horizon_hours

        incumbent_auc = _incumbent_auc(registry, asset)
        decision = gate.evaluate(asset, candidate.metrics, incumbent_auc)
        decisions[asset] = decision

        version = _track_and_maybe_promote(registry, store, candidate, decision)
        logger.info(
            "training_stage_done",
            asset=asset.value,
            promoted=decision.promoted,
            reason=decision.reason,
            version=version,
        )
    return decisions


# --------------------------------------------------------------------------- #
# Stage 4 — drift monitoring
# --------------------------------------------------------------------------- #
def run_drift_report(settings: Settings | None = None) -> dict[Asset, DriftSummary]:
    """Compare recent features vs the training reference and emit HTML reports."""
    settings = settings or get_settings()
    yaml_cfg = settings.load_yaml_config()
    monitor = DriftMonitor(
        yaml_cfg.get("monitoring", {}).get("reports_dir", "reports"),
        yaml_cfg.get("monitoring", {}).get("drift_threshold", 0.5),
    )
    proc_repo = factory.processed_repository()

    summaries: dict[Asset, DriftSummary] = {}
    for asset in settings.assets:
        dataset = proc_repo.load(f"features_{asset.value.lower()}").drop(
            columns=["target_direction"], errors="ignore"
        )
        # Reference = older 70%, current = most recent 30% (proxy for prod drift).
        split = int(len(dataset) * 0.7)
        reference, current = dataset.iloc[:split], dataset.iloc[split:]
        summaries[asset] = monitor.run(asset.value, reference, current)
    return summaries


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _training_config(yaml_cfg: dict[str, Any]) -> TrainingConfig:
    t = yaml_cfg.get("training", {})
    return TrainingConfig(
        test_size=t.get("test_size", 0.2),
        n_splits=t.get("n_splits", 5),
        random_state=t.get("random_state", 42),
    )


def _gate_thresholds(settings: Settings, yaml_cfg: dict[str, Any]) -> GateThresholds:
    g = yaml_cfg.get("promotion_gate", {})
    return GateThresholds(
        min_roc_auc=settings.min_roc_auc or g.get("min_roc_auc", 0.55),
        min_accuracy=settings.min_accuracy or g.get("min_accuracy", 0.52),
        improvement_margin=settings.promotion_improvement or g.get("improvement_margin", 0.005),
    )


def _maybe_registry(settings: Settings) -> Any | None:
    """Return an MLflow registry adapter, or None if MLflow is unavailable."""
    try:
        from autopredict.evaluation.mlflow_registry import MLflowModelRegistry

        return MLflowModelRegistry(settings.mlflow_experiment)
    except Exception as exc:  # noqa: BLE001 - tracking is optional
        logger.warning("mlflow_unavailable", detail=str(exc))
        return None


def _incumbent_auc(registry: Any | None, asset: Asset) -> float | None:
    if registry is None:
        return None
    metrics = registry.get_production_metrics(asset)
    return metrics.get("roc_auc") if metrics else None


def _track_and_maybe_promote(
    registry: Any | None, store: Any, candidate: Any, decision: PromotionDecision
) -> str:
    run_name = f"{candidate.asset.value}-{datetime.now(tz=UTC):%Y%m%d-%H%M%S}"
    version = "local"
    if registry is not None:
        version = registry.log_candidate(candidate, run_name)
    if decision.promoted:
        # The served artifact is always the promoted candidate.
        store.save(candidate, version=None if version == "local" else version)
        if registry is not None:
            registry.promote(candidate.asset, version, ModelStage.PRODUCTION)
    return version
