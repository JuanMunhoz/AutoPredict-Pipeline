# MLOps Design

This document explains the automated lifecycle and the decisions that make the
system production-realistic.

## The retraining loop

```
schedule (every 4h) ──► ingest ──► validate ──► features ──► train
                                                               │
                                          ┌────────────────────┘
                                          ▼
                                   evaluate metrics
                                          │
                                   ┌──────┴───────┐
                                   ▼              ▼
                             PROMOTION GATE   (reject)
                                   │           keep prod
                             (promote)
                                   ▼
                         register + version (MLflow + DVC)
                                   ▼
                         build image ──► push GHCR ──► rolling deploy
                                   ▼
                              monitor + drift
                                   └──────── drift ⇒ next retrain
```

## The promotion gate

The single most important piece of governance. A candidate becomes `Production`
**iff**:

1. `roc_auc ≥ min_roc_auc` (absolute floor), **and**
2. `accuracy ≥ min_accuracy`, **and**
3. if an incumbent exists: `candidate_auc ≥ incumbent_auc + improvement_margin`.

No incumbent (cold start) ⇒ clearing the absolute floors is enough. This
prevents silent regressions: production only ever moves *forward*.

Thresholds live in `configs/config.yaml` and can be overridden by env vars, so
the risk appetite is tunable without code changes.

## Experiment tracking & registry (MLflow)

Every training run logs parameters, hyperparameters, metrics, the dataset shape
and the serialised model. Models are registered under
`autopredict-<asset>` and transitioned through `Staging → Production → Archived`.
The gate reads the current Production metrics to compute the comparison.

MLflow is **optional at runtime**: if the tracking server is unreachable,
training still completes and persists locally, so developer laptops and CI don't
require infrastructure.

## Data & model versioning (DVC)

`data/` and `models/` are DVC-tracked and pushed to Supabase Storage
(S3-compatible). Git stores only lightweight `.dvc` pointers, so:
- reproducibility: any commit can `dvc pull` the exact data/model it used;
- lean repo: no large binaries in git history.

## Leakage prevention

- Labels use a **future** close (`shift(-horizon)`); the last `horizon` rows are
  dropped because their future is unknown.
- Splits are **chronological**; CV is **walk-forward** (`TimeSeriesSplit`).
- Features only use past/current information.

## Zero-downtime deploys

The image (API + model) is pushed to GHCR and deployed to Fly.io with a
`rolling` strategy and `max_unavailable = 0`, so machines are replaced one at a
time behind health checks — users never hit a cold instance.

## Observability

- **Logs**: structured JSON in prod, request-scoped correlation ids.
- **Metrics**: Prometheus counters/histograms at `/metrics`.
- **Drift**: Evidently HTML reports comparing training reference vs recent
  production features; a high drift share signals the next retrain.
