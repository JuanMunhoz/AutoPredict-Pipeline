"""Integration-ish unit tests: train a real (small) model and serve it."""

from __future__ import annotations

import pandas as pd
import pytest

from autopredict.core.enums import Asset
from autopredict.core.exceptions import FeatureEngineeringError
from autopredict.deployment import ModelStore
from autopredict.features import FeatureConfig, FeaturePipeline
from autopredict.training import ModelTrainer, TrainingConfig

MODEL_CONFIG = {
    "estimator": "logistic_regression",
    "logistic_regression": {"C": 1.0, "max_iter": 500},
}


@pytest.fixture
def xy(synthetic_ohlcv: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    pipeline = FeaturePipeline(FeatureConfig(horizon_hours=4))
    return pipeline.build_supervised(synthetic_ohlcv)


def test_trainer_produces_valid_model(xy: tuple[pd.DataFrame, pd.Series]) -> None:
    x, y = xy
    trainer = ModelTrainer(MODEL_CONFIG, TrainingConfig(test_size=0.2, n_splits=3))
    model = trainer.train(Asset.BTC, x, y)
    assert model.feature_names == list(x.columns)
    assert 0.0 <= model.metrics.accuracy <= 1.0
    assert model.metrics.n_samples > 0


def test_model_store_roundtrip_and_prediction(
    xy: tuple[pd.DataFrame, pd.Series], tmp_path: object
) -> None:
    x, y = xy
    trainer = ModelTrainer(MODEL_CONFIG, TrainingConfig(test_size=0.2, n_splits=3))
    model = trainer.train(Asset.BTC, x, y)

    store = ModelStore(tmp_path)  # type: ignore[arg-type]
    version = store.save(model)
    predictor = store.load(Asset.BTC)
    assert predictor.version == version

    features = x.iloc[-1].to_dict()
    prediction = predictor.predict(Asset.BTC, features)
    assert prediction.label in {"UP", "DOWN"}
    assert 0.0 <= prediction.probability_up <= 1.0


def test_predictor_rejects_missing_features(
    xy: tuple[pd.DataFrame, pd.Series], tmp_path: object
) -> None:
    x, y = xy
    trainer = ModelTrainer(MODEL_CONFIG, TrainingConfig(test_size=0.2, n_splits=3))
    model = trainer.train(Asset.BTC, x, y)
    store = ModelStore(tmp_path)  # type: ignore[arg-type]
    store.save(model)
    predictor = store.load(Asset.BTC)
    with pytest.raises(FeatureEngineeringError, match="Missing features"):
        predictor.predict(Asset.BTC, {"rsi": 50.0})
