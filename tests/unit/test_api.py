"""API tests with a stubbed predictor registry (no model artifacts needed)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from autopredict.api.dependencies import PredictorRegistry
from autopredict.api.main import create_app
from autopredict.core.entities import Prediction
from autopredict.core.enums import Asset, Direction


class StubPredictor:
    version = "test-1"
    feature_names = ["rsi", "macd"]

    def predict(self, asset: Asset, features: dict[str, float]) -> Prediction:
        return Prediction(
            asset=asset,
            direction=Direction.UP,
            probability_up=0.73,
            model_version=self.version,
            horizon_hours=4,
            created_at=datetime.now(UTC),
        )


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    with TestClient(app) as test_client:
        # Override the (empty) registry the lifespan loaded — no artifacts on disk.
        test_client.app.state.registry = PredictorRegistry(
            predictors={Asset.BTC: StubPredictor()}  # type: ignore[dict-item]
        )
        yield test_client


def test_health_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert "BTC" in resp.json()["models_loaded"]


def test_predict_returns_direction(client: TestClient) -> None:
    resp = client.post("/predict", json={"asset": "BTC", "features": {"rsi": 55.0, "macd": 0.1}})
    assert resp.status_code == 200
    body = resp.json()
    assert body["direction"] == "UP"
    assert body["probability_up"] == pytest.approx(0.73)
    assert body["model_version"] == "test-1"


def test_predict_unloaded_asset_returns_503(client: TestClient) -> None:
    resp = client.post("/predict", json={"asset": "SOL", "features": {"rsi": 55.0}})
    assert resp.status_code == 503


def test_request_id_header_present(client: TestClient) -> None:
    resp = client.get("/health")
    assert "x-request-id" in resp.headers
    assert "x-process-time-ms" in resp.headers
