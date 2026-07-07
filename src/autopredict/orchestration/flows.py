"""Prefect flows.

Thin orchestration wrappers over the composition-root pipeline functions. They
add retries, logging and scheduling metadata without duplicating logic. Prefect
is an optional dependency; import this module only where it is installed.

Deploy/schedule example (every 4h) once a Prefect worker is running::

    prefect deployment build src/autopredict/orchestration/flows.py:full_pipeline \
        -n autopredict-4h --cron "0 */4 * * *" --apply
"""

from __future__ import annotations

from prefect import flow, task

from autopredict.monitoring import get_logger
from autopredict.orchestration import pipeline

logger = get_logger(__name__)


@task(retries=3, retry_delay_seconds=30, name="ingest")
def ingest_task() -> None:
    pipeline.run_ingestion()


@task(name="build-features")
def features_task() -> None:
    pipeline.run_feature_build()


@task(name="train-and-gate")
def train_task() -> dict[str, bool]:
    decisions = pipeline.run_training()
    return {a.value: d.promoted for a, d in decisions.items()}


@task(name="drift-report")
def drift_task() -> None:
    pipeline.run_drift_report()


@flow(name="autopredict-full-pipeline", log_prints=True)
def full_pipeline() -> dict[str, bool]:
    """Ingest -> features -> train/gate -> drift, in order."""
    ingest_task()
    features_task()
    promoted = train_task()
    drift_task()
    logger.info("full_pipeline_completed", promoted=promoted)
    return promoted


if __name__ == "__main__":  # pragma: no cover
    full_pipeline()
