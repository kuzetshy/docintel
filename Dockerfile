# syntax=docker/dockerfile:1

# ---- Build stage: install dependencies and the project into a venv ----
FROM python:3.12.16-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Dependencies first: this layer is cached until uv.lock or pyproject.toml change.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Then the project itself, installed as a regular (non-editable) package.
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# ---- Runtime stage: only Python and the venv, no uv, no sources ----
FROM python:3.12.16-slim AS runtime

RUN groupadd --system app && useradd --system --gid app --no-create-home app \
    && mkdir -p /app/.cache/fastembed && chown -R app:app /app/.cache

COPY --from=builder --chown=app:app /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    EMBEDDING_CACHE_DIR=/app/.cache/fastembed \
    # The user has no home dir; keep Hugging Face state inside the cache volume.
    HF_HOME=/app/.cache/fastembed/huggingface

WORKDIR /app
USER app
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]

CMD ["uvicorn", "docintel.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
