# Only needed if you deploy somewhere that wants a container.
# Render's native Python runtime uses render.yaml instead.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependency layer first so code edits do not invalidate the install.
COPY pyproject.toml uv.lock* README.md ./
RUN uv sync --no-dev --no-install-project

COPY src ./src
RUN uv sync --no-dev

EXPOSE 8000
CMD ["sh", "-c", "uv run uvicorn bike_selector.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
