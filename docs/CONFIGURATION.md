# Configuration

All values are environment variables. Startup fails on invalid or unsafe combinations.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL_SOURCE` | `/models/model` | Local directory or private S3 prefix |
| `MODEL_ID`, `MODEL_REVISION` | descriptive fallback | Manifest is authoritative after validation |
| `MODEL_FORMAT` | `safetensors` | `safetensors` or `gguf`; must match the manifest and engine |
| `MODEL_DTYPE` | `bfloat16` | Transformers only: `float32`, `float16`, or `bfloat16` |
| `MODEL_MAX_CONTEXT` | `8192` | Maximum prompt + completion budget |
| `MODEL_MAX_NEW_TOKENS` | `1024` | Per-request completion ceiling |
| `MAX_PROMPT_TOKENS` | `7168` | Prompt ceiling after tokenization |
| `INFERENCE_ENGINE` | `transformers` | `transformers` for safetensors, `llama_cpp` for GGUF; `mock` is test-only |
| `DEVICE` | `cuda` | `cuda`, `cpu`, or `auto` |
| `TENSOR_PARALLEL_SIZE` | `1` | Reserved/diagnostic; Transformers reference path is single-process |
| `GPU_MEMORY_UTILIZATION` | `0.90` | Preflight/runtime policy input; capped at `0.98` |
| `MAX_CONCURRENT_REQUESTS` | `1` | Active generation permits; the single-slot llama.cpp backend requires `1` |
| `REQUEST_QUEUE_TIMEOUT_SECONDS` | `0.25` | Bounded wait before `429` |
| `MAX_BATCH_TOKENS` | `32768` | Capacity policy; must cover max context |
| `MAX_REQUEST_BYTES` | `1048576` | Header and chunked-body byte ceiling |
| `GENERATION_TIMEOUT_SECONDS` | `300` | End-to-end generation limit |
| `MODEL_CACHE_DIR` | `/var/cache/inference-models` | Verified S3 artifact cache |
| `STRICT_MANIFEST` | `true` | Reject unlisted local files |
| `API_KEY` | none | Required unless explicitly unauthenticated |
| `ALLOW_UNAUTHENTICATED` | `false` | Development/test escape hatch |
| `ENVIRONMENT` | `production` | Controls docs and mock-engine permission |
| `LOG_LEVEL` | `INFO` | Structured log level |
| `LOG_PROMPTS` | `false` | Reserved; prompt logging remains off by design |
| `LLAMA_SERVER_PATH` | `/opt/llama.cpp/llama-server` | Executable used by the GGUF backend |
| `LLAMA_CPP_REQUIRE_PINNED_RUNTIME` | `true` | Require build `b10355`/the pinned commit at startup |
| `LLAMA_CPP_STARTUP_TIMEOUT_SECONDS` | `1800` | Maximum GGUF load/readiness time |
| `LLAMA_CPP_THREADS` | `0` | CPU threads; `0` selects a bounded host-based default |
| `LLAMA_CPP_BATCH_SIZE` | `512` | Logical llama.cpp prompt batch size |
| `LLAMA_CPP_UBATCH_SIZE` | `128` | Physical batch size; cannot exceed batch size |
| `LLAMA_CPP_GPU_LAYERS` | `-1` | Layers offloaded when `DEVICE` is not `cpu`; `-1` means all possible |
| `LLAMA_CPP_KV_CACHE_TYPE` | `f16` | `f16` or `q8_0` KV cache |
| `LLAMA_CPP_REPACK` | `false` | Enable runtime tensor repacking only after qualification |

`MAX_PROMPT_TOKENS + MODEL_MAX_NEW_TOKENS` must not exceed `MODEL_MAX_CONTEXT`. `MAX_BATCH_TOKENS` must be at least `MODEL_MAX_CONTEXT`. `transformers` requires `safetensors`; `llama_cpp` requires `gguf`. The checked `.env.example` and `.env.qwen.example` provide coherent profiles for each path.
