# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=python:3.12.13-slim-bookworm

FROM ${BASE_IMAGE} AS native-runtime-base
RUN apt-get update && \
    apt-get install --yes --no-install-recommends libgomp1 libssl3 && \
    rm -rf /var/lib/apt/lists/*

FROM native-runtime-base AS llama-runtime
ARG TARGETARCH
WORKDIR /runtime-build
COPY vendor/llama.cpp.lock.json vendor/LLAMA_CPP_LICENSE.txt ./vendor/
COPY scripts/download_llama_runtime.py ./scripts/download_llama_runtime.py
RUN case "$TARGETARCH" in \
      amd64) runtime_key=linux-x64 ;; \
      arm64) runtime_key=linux-arm64 ;; \
      *) echo "Unsupported container architecture: $TARGETARCH" >&2; exit 1 ;; \
    esac && \
    python scripts/download_llama_runtime.py \
      --key "$runtime_key" \
      --destination /opt/llama.cpp

FROM ${BASE_IMAGE} AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /build
COPY requirements.lock pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --upgrade "pip==26.2.1" && \
    python -m pip wheel --wheel-dir /wheels -r requirements.lock .

FROM native-runtime-base AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    MODEL_CACHE_DIR=/var/cache/inference-models \
    LLAMA_SERVER_PATH=/opt/llama.cpp/llama-server \
    PORT=8000

RUN groupadd --system --gid 10001 inference && \
    useradd --system --uid 10001 --gid inference --home-dir /nonexistent --shell /usr/sbin/nologin inference && \
    mkdir -p /app /var/cache/inference-models && \
    chown -R inference:inference /app /var/cache/inference-models
WORKDIR /app
COPY --from=builder /wheels /wheels
COPY --from=llama-runtime /opt/llama.cpp /opt/llama.cpp
RUN python -m pip install --no-index --find-links=/wheels /wheels/*.whl && rm -rf /wheels

USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=120s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2)"]
ENTRYPOINT ["python", "-m", "inference_service.main"]
