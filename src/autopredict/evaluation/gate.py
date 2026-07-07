"""The promotion gate — the automated decision that guards production.

A candidate is promoted only if it (a) clears absolute minimum thresholds and
(b) beats the incumbent Production model's AUC by a configured margin. If there
is no incumbent, clearing the absolute thresholds is enough. This encodes the
"the model may only be promoted if it surpasses the minimum metrics" rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from autopredict.core.entities import ModelMetrics, PromotionDecision
from autopredict.core.enums import Asset
from autopredict.monitoring import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class GateThresholds:
    """Configurable promotion criteria."""

    min_roc_auc: float = 0.55
    min_accuracy: float = 0.52
    improvement_margin: float = 0.005


class PromotionGate:
    """Decide whether a candidate model should become Production."""

    def __init__(self, thresholds: GateThresholds) -> None:
        self._t = thresholds

    def evaluate(
        self,
        asset: Asset,
        candidate: ModelMetrics,
        incumbent_auc: float | None,
    ) -> PromotionDecision:
        """Return a :class:`PromotionDecision` for the candidate."""
        if math.isnan(candidate.roc_auc):
            return self._reject(asset, candidate, incumbent_auc, "candidate AUC is undefined")

        if candidate.roc_auc < self._t.min_roc_auc:
            return self._reject(
                asset,
                candidate,
                incumbent_auc,
                f"AUC {candidate.roc_auc:.4f} < min {self._t.min_roc_auc}",
            )
        if candidate.accuracy < self._t.min_accuracy:
            return self._reject(
                asset,
                candidate,
                incumbent_auc,
                f"accuracy {candidate.accuracy:.4f} < min {self._t.min_accuracy}",
            )

        if incumbent_auc is None:
            return self._accept(asset, candidate, None, "no incumbent; thresholds met")

        required = incumbent_auc + self._t.improvement_margin
        if candidate.roc_auc < required:
            return self._reject(
                asset,
                candidate,
                incumbent_auc,
                f"AUC {candidate.roc_auc:.4f} < incumbent+margin {required:.4f}",
            )
        return self._accept(
            asset,
            candidate,
            incumbent_auc,
            f"AUC {candidate.roc_auc:.4f} beats incumbent {incumbent_auc:.4f}",
        )

    def _accept(
        self, asset: Asset, m: ModelMetrics, incumbent: float | None, reason: str
    ) -> PromotionDecision:
        logger.info("promotion_accepted", asset=asset.value, reason=reason, auc=m.roc_auc)
        return PromotionDecision(
            asset=asset,
            promoted=True,
            reason=reason,
            candidate_auc=m.roc_auc,
            incumbent_auc=incumbent,
        )

    def _reject(
        self, asset: Asset, m: ModelMetrics, incumbent: float | None, reason: str
    ) -> PromotionDecision:
        logger.warning("promotion_rejected", asset=asset.value, reason=reason, auc=m.roc_auc)
        return PromotionDecision(
            asset=asset,
            promoted=False,
            reason=reason,
            candidate_auc=m.roc_auc,
            incumbent_auc=incumbent,
        )
