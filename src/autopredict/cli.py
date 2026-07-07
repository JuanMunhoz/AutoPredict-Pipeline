"""Typer CLI — the operator entrypoint for every pipeline stage.

Exposed as the ``autopredict`` console script. Each command maps to one
composition-root function so the pipeline can be driven from a terminal,
a Makefile, GitHub Actions or a container CMD identically.
"""

from __future__ import annotations

import typer

from autopredict.config import get_settings
from autopredict.monitoring import configure_logging, get_logger

app = typer.Typer(
    name="autopredict",
    help="Automated MLOps pipeline for crypto price-direction forecasting.",
    add_completion=False,
    no_args_is_help=True,
)
logger = get_logger(__name__)


@app.callback()
def _bootstrap() -> None:
    """Configure logging before any command runs."""
    settings = get_settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json)


@app.command()
def ingest() -> None:
    """Collect and persist raw market + sentiment data."""
    from autopredict.orchestration import run_ingestion

    results = run_ingestion()
    for asset, df in results.items():
        typer.echo(f"[ingest] {asset.value}: {len(df)} rows")


@app.command("build-features")
def build_features() -> None:
    """Engineer features and labels; persist the supervised dataset."""
    from autopredict.orchestration import run_feature_build

    shapes = run_feature_build()
    for asset, (rows, cols) in shapes.items():
        typer.echo(f"[features] {asset.value}: {rows} rows x {cols} features")


@app.command()
def train() -> None:
    """Train, evaluate the promotion gate and promote the winner."""
    from autopredict.orchestration import run_training

    decisions = run_training()
    for asset, decision in decisions.items():
        status = "PROMOTED" if decision.promoted else "REJECTED"
        typer.echo(f"[train] {asset.value}: {status} — {decision.reason}")


@app.command("drift-report")
def drift_report() -> None:
    """Generate Evidently drift reports (HTML) per asset."""
    from autopredict.orchestration import run_drift_report

    summaries = run_drift_report()
    for asset, summary in summaries.items():
        typer.echo(
            f"[drift] {asset.value}: drift={summary.dataset_drift} "
            f"share={summary.drift_share:.2f} -> {summary.report_path}"
        )


@app.command("run-all")
def run_all() -> None:
    """Run the full pipeline end-to-end (ingest -> features -> train -> drift)."""
    ingest()
    build_features()
    train()
    drift_report()


if __name__ == "__main__":  # pragma: no cover
    app()
