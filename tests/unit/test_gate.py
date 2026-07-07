"""Unit tests for the promotion gate — the production guardrail."""

from __future__ import annotations

import pytest

from autopredict.core.entities import ModelMetrics
from autopredict.core.enums import Asset
from autopredict.evaluation.gate import GateThresholds, PromotionGate


def _metrics(auc: float, acc: float = 0.6) -> ModelMetrics:
    return ModelMetrics(
        roc_auc=auc, accuracy=acc, precision=0.6, recall=0.6, f1=0.6,
        log_loss=0.6, n_samples=100,
    )


@pytest.fixture
def gate() -> PromotionGate:
    return PromotionGate(GateThresholds(min_roc_auc=0.55, min_accuracy=0.52, improvement_margin=0.01))


def test_promotes_when_no_incumbent_and_thresholds_met(gate: PromotionGate) -> None:
    decision = gate.evaluate(Asset.BTC, _metrics(0.60), incumbent_auc=None)
    assert decision.promoted is True


def test_rejects_below_min_auc(gate: PromotionGate) -> None:
    decision = gate.evaluate(Asset.BTC, _metrics(0.50), incumbent_auc=None)
    assert decision.promoted is False
    assert "min" in decision.reason


def test_rejects_when_not_beating_incumbent_margin(gate: PromotionGate) -> None:
    decision = gate.evaluate(Asset.SOL, _metrics(0.605), incumbent_auc=0.60)
    assert decision.promoted is False


def test_promotes_when_beating_incumbent_margin(gate: PromotionGate) -> None:
    decision = gate.evaluate(Asset.SOL, _metrics(0.62), incumbent_auc=0.60)
    assert decision.promoted is True


def test_rejects_nan_auc(gate: PromotionGate) -> None:
    decision = gate.evaluate(Asset.BTC, _metrics(float("nan")), incumbent_auc=None)
    assert decision.promoted is False
