# syntax=docker/dockerfile:1

# ---- build: resolve dependencies with uv into a self-contained virtualenv ----
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /bin/uv
ARG WITH_ASR=true
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$WITH_ASR" = "true" ]; then EXTRA="--extra asr"; else EXTRA=""; fi; \
    uv sync --frozen --no-dev --no-install-project $EXTRA
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$WITH_ASR" = "true" ]; then EXTRA="--extra asr"; else EXTRA=""; fi; \
    uv sync --frozen --no-dev --no-editable $EXTRA

# ---- runtime: ffmpeg (with libass for the burned-in captions) and the app ----
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface
COPY --from=builder /app/.venv /app/.venv
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --chown=app:app assets ./assets
COPY --chown=app:app data/samples ./data/samples
COPY --chown=app:app data/testset/manifest.json ./data/testset/manifest.json
COPY --chown=app:app results ./results
ENV RECUT_DATA_DIR=/app/data/projects \
    RECUT_SAMPLES_DIR=/app/data/samples \
    RECUT_FACE_MODEL=/app/assets/models/face_detection_yunet_2023mar.onnx \
    RECUT_FONTS_DIR=/app/assets/fonts \
    RECUT_LLM_CACHE_DIR=/home/app/.cache/llm \
    RECUT_LLM_LEDGER=/home/app/.cache/calls.jsonl \
    RECUT_TRANSCRIPT_CACHE_DIR=/home/app/.cache/transcripts \
    LOG_FORMAT=json
RUN mkdir -p /app/data/projects /home/app/.cache && chown -R app:app /app/data /home/app/.cache
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=6 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]
CMD ["uvicorn", "recut.api.app:create_default_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
