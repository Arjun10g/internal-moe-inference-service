# Internal MoE Inference Service

A secure, offline-capable HTTP/SSE inference service for locally mounted or private-S3 decoder-only checkpoints. It supports Transformers safetensors and a pinned `llama.cpp` GGUF runtime. The process resolves and verifies artifacts once, loads one model into RAM/VRAM, warms it, and reuses that resident instance for every request.

This repository deliberately contains **no model weights** and **no AWS deployment infrastructure**. It provides the containerized inference workload that a separate ECS/GPU deployment can run.

## What is implemented

- Local-directory and `s3://bucket/prefix` model sources.
- Versioned JSON manifest, SHA-256, byte-size, safe-path, allowed-file, and symlink validation.
- Format-locked engines: safetensors use offline Transformers; GGUF uses the pinned official `llama-server` build.
- Supervised `llama.cpp` process bound only to loopback with an ephemeral internal API key, no web UI, no agent mode, and offline loading.
- Background startup lifecycle: resolve → validate → preflight → load → warmup → ready.
- OpenAI-shaped non-streaming and SSE chat completions.
- Bounded inference concurrency, queue timeout/backpressure, generation timeout, and cancellation signaling.
- Bearer authentication, request-size limits, no-store responses, request IDs, and redacted JSON logs.
- Liveness, readiness, current-model, and Prometheus metrics endpoints.
- Non-root, read-only-compatible Docker image with offline/telemetry-disabled defaults.
- CPU CI test that creates and executes a real tiny Qwen2-MoE safetensors checkpoint entirely offline.
- Separate scripts for validation, manifests, S3 upload, smoke testing, benchmarking, and memory estimation.

The included tiny random model verifies the Transformers serving pipeline—not language quality or compatibility/performance of a supplied ~30B checkpoint. The GGUF integration test exercises runtime supervision and both completion modes against a process-compatible test server. Run the external-checkpoint qualification in `docs/MODEL_ARTIFACTS.md` on the actual target hardware before making performance claims.

## Quick start with the offline tiny MoE

```bash
uv sync --extra test
make test-tiny-model
make smoke-tiny-model
```

The smoke command creates a temporary Qwen2-MoE checkpoint, starts the real Transformers backend on CPU, makes non-streaming and streaming requests, verifies generated-token counts, and proves `model_load_total` remains `1`.

## Approved Qwen GCP download

The companion local-coder project selects **Qwen3-Coder 30B-A3B UD-Q4_K_XL** and
publishes its approved GGUF through Google Cloud Storage. This repository keeps
the exact URL, expected byte count, and SHA-256 in a resumable verification
helper:

```bash
make qwen-url       # print the direct GCP URL without downloading
make qwen-check     # verify that the object is reachable and has the approved size
make qwen-download  # resume/download to models/ and verify SHA-256 before publication
```

The downloaded file is about 16.48 GiB. The helper verifies the full SHA-256 and
creates the required `model-manifest.json` beside the GGUF. Model files and
partial downloads are excluded from Git and Docker build contexts.

The artifact is GGUF, so it must be run with `MODEL_FORMAT=gguf` and
`INFERENCE_ENGINE=llama_cpp`. The service refuses mismatched engine/format pairs.
The model source is the containing directory, not the `.gguf` file itself.

See [`docs/QWEN_GCP_MODEL.md`](docs/QWEN_GCP_MODEL.md) for the direct link,
approved digest, output override, runtime pin, and recovery behavior.

## Run the approved Qwen GGUF

Native execution is the practical path on Apple Silicon because the downloaded
Darwin runtime can use Metal. The runtime downloader selects the current
platform and verifies the release archive's exact byte count and SHA-256:

```bash
uv sync --extra test
make qwen-download
make llama-runtime
cp .env.qwen.example .env

# Replace API_KEY in .env, then point these at the downloaded directories.
export MODEL_SOURCE="$PWD/models/qwen3-coder-30b-a3b-q4xl"
export LLAMA_SERVER_PATH="$PWD/runtime/darwin-arm64/llama-server" # Apple Silicon
.venv/bin/inference-service
```

On Linux, use `runtime/linux-x64/llama-server` or
`runtime/linux-arm64/llama-server` as printed by `make llama-runtime`.

The Docker image embeds the verified upstream Linux CPU `llama.cpp` runtime and
its native library dependencies. After the model download, set
`LOCAL_MODEL_PATH` to its containing directory, copy
`.env.qwen.example` to `.env`, replace the API key, and run:

```bash
export LOCAL_MODEL_PATH="$PWD/models/qwen3-coder-30b-a3b-q4xl"
docker compose up --build
```

Apple Docker Desktop does not expose Metal to Linux containers, and the checked
Linux archive is CPU-only; use native execution for Metal acceleration on a Mac.

## Run with supplied safetensors weights

The directory must satisfy the safetensors contract in `docs/MODEL_ARTIFACTS.md` and contain `model-manifest.json`.

```bash
export API_KEY="$(openssl rand -hex 32)"
docker build --target runtime -t internal-llm .
docker run --rm --gpus all \
  --read-only --tmpfs /tmp:rw,noexec,nosuid,size=256m \
  --cap-drop ALL --security-opt no-new-privileges:true \
  -p 127.0.0.1:8000:8000 \
  -v "$PWD/model:/models/model:ro" \
  -e MODEL_SOURCE=/models/model \
  -e API_KEY="$API_KEY" \
  -e DEVICE=cuda -e MODEL_DTYPE=bfloat16 \
  internal-llm
```

In production, inject the API key through the task secret mechanism rather than shell history.

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Explain paged attention."}],"max_tokens":64}'
```

Set `"stream":true` for SSE.

## S3 model source

```bash
docker run --rm --gpus all -p 127.0.0.1:8000:8000 \
  -e MODEL_SOURCE=s3://private-model-bucket/models/qwen/revision-001 \
  -e API_KEY="$API_KEY" -e DEVICE=cuda -e MODEL_DTYPE=bfloat16 \
  internal-llm
```

The SDK uses the normal credential provider chain. On ECS, use the task IAM role. No static AWS key configuration exists in this codebase.

## Endpoints

| Endpoint | Auth | Purpose |
| --- | --- | --- |
| `GET /health/live` | No | Process liveness and model state |
| `GET /health/ready` | No | `200` only after load and warmup |
| `GET /v1/models` | Bearer | OpenAI-shaped list containing the resident model |
| `GET /v1/models/current` | Bearer | Loaded revision and non-sensitive runtime metadata |
| `POST /v1/chat/completions` | Bearer | Non-streaming or SSE generation |
| `GET /metrics` | Bearer | Prometheus metrics |

## Verification commands

```bash
make lint
make typecheck
make test
make security
make audit
make docker-build
make docker-smoke-tiny-model
```

Docker commands require a local Docker daemon. Full GPU/model tests use the `gpu`, `external_checkpoint`, and `slow` pytest markers and are intentionally separate from ordinary CPU CI.

The checked verification result for this release is recorded in `artifacts/verification/verification.json`.

## Design boundaries

- ECR stores the image; S3 stores model artifacts; the GPU task performs computation.
- No model download, reconstruction, or reload occurs in a normal request.
- No Hugging Face Hub or public network access is used at runtime.
- No pickle checkpoints, arbitrary remote URLs, or model-provided Python code are accepted; GGUF and safetensors cannot be mixed.
- This project does not provision AWS resources or implement a deployment pipeline.

See `docs/ARCHITECTURE.md`, `docs/SECURITY.md`, `docs/CONFIGURATION.md`, and `docs/AWS_RUNTIME_CONTRACT.md` for operational details.
