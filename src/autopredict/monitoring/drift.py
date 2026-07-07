"""Data & feature drift detection with Evidently.

Compares a *reference* dataset (typically the training features) against
*current* production features, produces an HTML report saved to ``reports/``,
and returns a compact summary the pipeline can act on (e.g. trigger retrain).
``evidently`` is an optional dependency, imported lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from autopredict.monitoring import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class DriftSummary:
    """Machine-readable outcome of a drift run."""

    asset: str
    dataset_drift: bool
    drift_share: float
    n_drifted_features: int
    n_features: int
    report_path: str


class DriftMonitor:
    """Generate Evidently drift reports and summarise them."""

    def __init__(self, reports_dir: Path | str, drift_threshold: float = 0.5) -> None:
        self._dir = Path(reports_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._threshold = drift_threshold

    def run(self, asset: str, reference: pd.DataFrame, current: pd.DataFrame) -> DriftSummary:
        """Compute drift between reference and current feature frames."""
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        common = [c for c in reference.columns if c in current.columns]
        ref, cur = reference[common], current[common]

        report = Report(metrics=[DataDriftPreset(drift_share=self._threshold)])
        report.run(reference_data=ref, current_data=cur)

        stamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        report_path = self._dir / f"drift_{asset.lower()}_{stamp}.html"
        report.save_html(str(report_path))

        result = report.as_dict()
        drift = result["metrics"][0]["result"]
        summary = DriftSummary(
            asset=asset,
            dataset_drift=bool(drift["dataset_drift"]),
            drift_share=float(drift["share_of_drifted_columns"]),
            n_drifted_features=int(drift["number_of_drifted_columns"]),
            n_features=int(drift["number_of_columns"]),
            report_path=str(report_path),
        )
        logger.info(
            "drift_report_generated",
            asset=asset,
            dataset_drift=summary.dataset_drift,
            drift_share=summary.drift_share,
            path=summary.report_path,
        )
        return summary
