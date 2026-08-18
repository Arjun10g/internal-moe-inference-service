# Architecture

## Startup lifecycle

1. Parse and validate all configuration before the server is created.
2. Expose liveness immediately while readiness remains `503`.
3. Resolve a read-only local directory or download a manifest-declared S3 revision into an atomic cache directory.
4. Reject unsafe paths, symlinks, unexpected files, pickle-style formats, byte-size mismatches, and SHA-256 mismatches.
5. Inspect runtime/GPU capacity and record non-sensitive diagnostics.
6. Select the engine required by the manifest format:
   - `safetensors` loads through offline Transformers with remote code disabled.
   - `gguf` starts the pinned `llama-server` on a random loopback port with an ephemeral internal API key.
7. Execute a short warmup generation.
8. Change state to `ready`; the same engine and model objects remain resident until shutdown.

An S3 cache directory is addressed by bucket, prefix, and canonical manifest digest. A `.complete` marker is written only after every artifact has been streamed, hashed, and revalidated. Corrupt or partial directories are never reused.

## Request lifecycle

`HTTP → schema/byte/auth limits → capacity permit → selected engine → prefill/KV cache/decode → JSON or SSE → metrics`

The request handler never calls storage resolution or model loading. A bounded semaphore limits active generations; a short bounded queue timeout returns `429` rather than allowing unbounded memory growth. The API enforces configured prompt and completion limits before/while tokenizing.

## Components

- `config.py`: fail-closed settings validation.
- `storage/`: local and S3 artifact resolution.
- `model/manifest.py` and `validation.py`: supply-chain contract.
- `model/manager.py`: one-way lifecycle and resident ownership.
- `engines/transformers.py`: offline reference inference and SSE streaming.
- `engines/llama_cpp.py`: pinned-process supervision and loopback OpenAI API proxy for GGUF.
- `api/`: health, model, metrics, and chat endpoints.
- `security/`: bearer verification, paths, and redaction.
- `observability/`: isolated Prometheus registry and JSON logging.

## MoE memory

Active parameters per token determine compute, not checkpoint residency. The full set of expert weights must normally reside in CPU/GPU memory. `scripts/estimate_memory.py` therefore uses total parameters for weight memory and separately estimates KV-cache and runtime overhead.

## Deliberate exclusions

The service does not include Terraform/CDK/CloudFormation, ECS/ECR/VPC resources, model training/quantization, automatic model reloading, multi-model routing, or a vLLM backend. Transformers is the safetensors reference engine; pinned `llama.cpp` is the GGUF engine. Each engine has load, generation, streaming, and shutdown coverage.
