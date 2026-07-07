# AutoPredict-Pipeline — developer entrypoints.
# Uses `uv` for env & dependency management.

.DEFAULT_GOAL := help
SHELL := /bin/bash

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Create venv and install all extras + dev deps
	uv sync --all-extras --dev

.PHONY: lint
lint: ## Run ruff (lint) and mypy (types)
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy src

.PHONY: format
format: ## Auto-format with ruff
	uv run ruff format src tests
	uv run ruff check --fix src tests

.PHONY: test
test: ## Run the unit test suite with coverage
	uv run pytest

.PHONY: ingest
ingest: ## Run the ingestion pipeline once
	uv run autopredict ingest

.PHONY: features
features: ## Build the feature dataset
	uv run autopredict build-features

.PHONY: train
train: ## Train models and evaluate the promotion gate
	uv run autopredict train

.PHONY: drift
drift: ## Generate Evidently drift reports
	uv run autopredict drift-report

.PHONY: api
api: ## Run the FastAPI service locally
	uv run uvicorn autopredict.api.main:app --reload --port 8000

.PHONY: mlflow
mlflow: ## Start a local MLflow tracking server
	uv run mlflow server --host 0.0.0.0 --port 5000 \
		--backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts

.PHONY: compose-up
compose-up: ## Start the full local stack (api + mlflow + minio)
	docker compose -f docker/docker-compose.yml up -d --build

.PHONY: compose-down
compose-down: ## Stop the local stack
	docker compose -f docker/docker-compose.yml down

.PHONY: docker-build
docker-build: ## Build the production API image
	docker build -f docker/Dockerfile -t autopredict-api:local .
