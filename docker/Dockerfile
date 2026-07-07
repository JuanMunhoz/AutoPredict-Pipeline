# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Multi-stage build for the AutoPredict FastAPI service.
# Stage 1 resolves dependencies with uv into a venv; stage 2 is a slim runtime
# that carries only the venv, the package and the trained model artifacts.
# ---------------------------------------------------------------------------

FROM python:3.11-slim AS builder

# uv: fast, reproducible dependency resolution.
COPY --from=ghcr.io/astral-sh/uv:0.5.0 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Install deps first (cached layer) using only the lockfile + manifest.
COPY pyproject.toml uv.lock* README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra api --extra monitoring || \
    uv sync --no-install-project --extra api --extra monitoring

# Now install the project itself.
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-editable --extra api --extra monitoring || \
    uv pip install --no-deps .

# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Non-root user for the running container.
RUN groupadd --system app && useradd --system --gid app --create-home app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_ENV=production \
    APP_LOG_JSON=true \
    APP_MODEL_DIR=/app/models

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY configs ./configs
# Trained model artifacts (pulled via `dvc pull` in CI before build).
COPY models ./models

RUN chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8000/health').status_code==200 else 1)"

CMD ["uvicorn", "autopredict.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
