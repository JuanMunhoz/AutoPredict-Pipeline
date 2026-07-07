# Contributing

Thanks for your interest! This project follows a strict quality bar.

## Setup

```bash
uv sync --all-extras --dev
uv run pre-commit install
```

## Before you push

```bash
make format   # ruff format + autofix
make lint     # ruff check + mypy --strict
make test     # pytest with coverage
```

CI (`.github/workflows/ci.yml`) runs the same checks on Python 3.11 and 3.12.
Both must pass before merge.

## Conventions

- **Type hints everywhere**; `mypy --strict` must be clean.
- **Clean Architecture**: `core/` must not import from adapter layers. New
  external integrations go behind a `Protocol` in `core/ports.py`.
- **Tests**: every use-case is tested against in-memory fakes — no network in
  the unit suite. Mark network tests with `@pytest.mark.integration`.
- **Commits**: conventional-commit style (`feat:`, `fix:`, `docs:`, `refactor:`).

## Adding a new data provider

1. Implement `MarketDataProvider` / `SentimentProvider` in `ingestion/providers/`.
2. Wire it in `orchestration/factory.py`.
3. Add a unit test with a fake HTTP layer (`respx`).
