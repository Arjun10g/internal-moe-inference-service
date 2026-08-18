# Approved Qwen artifact on Google Cloud Storage

The companion
[`Arjun10g/restricted-local-coder`](https://github.com/Arjun10g/restricted-local-coder)
repository selects this profile as its default local coding model:

| Field | Approved value |
| --- | --- |
| Profile | `qwen3-coder-30b-a3b-q4xl` |
| Base model | `Qwen/Qwen3-Coder-30B-A3B-Instruct` |
| Quantization and format | `UD-Q4_K_XL`, GGUF |
| File | `Qwen3-Coder-30B-A3B-Instruct-1M-UD-Q4_K_XL.gguf` |
| Bytes | `17690500448` (16.48 GiB) |
| SHA-256 | `e71c9271166ad64865767022e86f45ea4f03a8258389460cc55c8d95e18833db` |
| GCP object | [Download from Google Cloud Storage](https://storage.googleapis.com/restricted-local-coder-dazzling-howl-491904/Qwen3-Coder-30B-A3B-Instruct-1M-UD-Q4_K_XL.gguf) |

## Verify or download

Use the repository helper instead of an unverified browser download:

```bash
uv sync --extra test
make qwen-check
make qwen-download
```

`qwen-check` makes a HEAD request and requires the exact approved object size.
`qwen-download` writes to a `.part` file, resumes with an HTTP Range request,
enforces the maximum byte count while streaming, hashes the completed file, and
atomically publishes it only after both size and SHA-256 match.

The default destination is
`models/qwen3-coder-30b-a3b-q4xl/Qwen3-Coder-30B-A3B-Instruct-1M-UD-Q4_K_XL.gguf`.
After verification, the helper writes `model-manifest.json` into that profile
directory. Set `MODEL_SOURCE` to the directory, not the file.
Choose another path with:

```bash
.venv/bin/python scripts/download_qwen_gcp.py \
  --output /approved/model-store/Qwen3-Coder-30B-A3B-Instruct-1M-UD-Q4_K_XL.gguf
```

An interrupted transfer is retained as `<file>.part`; rerun the same command to
resume. If a completed or partial local file fails verification, inspect why it
changed and pass `--force` only when you intentionally want to discard it and
restart.

## Runtime compatibility

The linked object is a single GGUF built for llama.cpp. This service runs it only
through `INFERENCE_ENGINE=llama_cpp`; the Transformers engine remains
safetensors-only. Configuration validation, manifest validation, and model
manager validation all reject cross-format combinations.

The runtime is pinned to upstream llama.cpp build `b10355`, commit
`dd1ea524333b1e697489067d7a4c39c60d32beee`, matching the companion repository.
`vendor/llama.cpp.lock.json` records exact release archive names, byte counts,
and SHA-256 values for Darwin arm64, Linux x64/arm64, and Windows CPU x64.

```bash
make llama-runtime
cp .env.qwen.example .env
export MODEL_SOURCE="$PWD/models/qwen3-coder-30b-a3b-q4xl"
export LLAMA_SERVER_PATH="$PWD/runtime/darwin-arm64/llama-server" # adjust for platform
.venv/bin/inference-service
```

At startup the service checks `llama-server --version`, starts it on a random
loopback port, authenticates the internal connection with an ephemeral key,
loads in offline mode, waits for health, and performs a warmup before reporting
ready. Set `LLAMA_CPP_REQUIRE_PINNED_RUNTIME=false` only for an intentionally
qualified replacement runtime.
