# Architecture

AutoPredict-Pipeline is organised around **Clean Architecture** (a.k.a. Ports &
Adapters / Hexagonal). The goal: business rules that are independent of
frameworks, databases and external services, so they stay testable and durable
while the outside world churns.

## The layers

```
┌──────────────────────────────────────────────────────────────┐
│  Adapters / Frameworks                                        │
│  Binance · CoinGecko · Fear&Greed · MLflow · FastAPI · joblib │
│                          │  implement                         │
│                          ▼                                    │
│  Use-cases (application)                                      │
│  DataCollector · ModelTrainer · PromotionGate · Predictors    │
│                          │  depend on                         │
│                          ▼                                    │
│  Domain core                                                  │
│  entities · enums · ports (Protocols) · exceptions            │
└──────────────────────────────────────────────────────────────┘
        dependencies point INWARD only (the Dependency Rule)
```

### 1. Domain core (`core/`)
Pure Python + Pydantic. Defines the vocabulary (`Asset`, `Direction`,
`OHLCV`, `Prediction`, `ModelMetrics`, `PromotionDecision`) and the **ports** —
`Protocol` interfaces such as `MarketDataProvider`, `ModelRegistry`,
`Predictor`. It imports nothing from the rest of the app.

### 2. Use-cases
Application logic that orchestrates ports:
- `DataCollector` — fetch → validate → persist.
- `FeaturePipeline` — raw candles → leakage-safe `(X, y)`.
- `ModelTrainer` — chronological split, fit, evaluate.
- `PromotionGate` — the production guardrail.

They depend on **abstractions**, never concrete adapters, so tests inject fakes.

### 3. Adapters
Concrete implementations bound to real technology: HTTP providers, the Parquet
repository, the MLflow registry, the FastAPI app, the joblib model store.

### 4. Composition root (`orchestration/`)
The *only* place that knows how to wire adapters to use-cases. `factory.py`
builds objects; `pipeline.py` sequences the stages; `flows.py` wraps them for
Prefect; `cli.py` exposes them to operators.

## Design patterns in use
- **Ports & Adapters** — every boundary is a `Protocol`.
- **Factory** — `training/model_factory.py` builds estimators from config.
- **Dependency Injection** — use-cases receive their collaborators.
- **Strategy** — swappable estimators (LightGBM/XGBoost/LogReg) behind one interface.
- **Repository** — `DatasetRepository` abstracts persistence.

## Why it matters
- Swap Binance for another exchange → write one adapter, touch nothing else.
- Unit-test the gate, trainer and collector with zero network calls.
- Replace MLflow with a different registry → implement the `ModelRegistry` port.
