# Model artifact contract

Every revision is an immutable directory/prefix containing exactly one supported
checkpoint format plus `model-manifest.json`. The manifest's `model_format` is
authoritative and startup also requires it to match `MODEL_FORMAT` and the
selected engine.

Safetensors example (`MODEL_FORMAT=safetensors`, `INFERENCE_ENGINE=transformers`):

```text
config.json
generation_config.json
tokenizer.json
tokenizer_config.json
special_tokens_map.json
model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
model-00003-of-00004.safetensors
model-00004-of-00004.safetensors
model.safetensors.index.json
model-manifest.json
```

Build and validate:

```bash
.venv/bin/python scripts/build_model_manifest.py ./model \
  --model-id internal-qwen-moe --revision version-001 --dtype bfloat16 \
  --format safetensors
.venv/bin/python scripts/validate_model.py ./model
```

GGUF revisions contain exactly one `.gguf` file and no safetensors. They do not
need separate tokenizer/config files because llama.cpp reads that metadata from
GGUF. Example (`MODEL_FORMAT=gguf`, `INFERENCE_ENGINE=llama_cpp`):

```text
Qwen3-Coder-30B-A3B-Instruct-1M-UD-Q4_K_XL.gguf
model-manifest.json
```

```bash
.venv/bin/python scripts/build_model_manifest.py ./model \
  --model-id qwen3-coder-30b-a3b-q4xl \
  --revision sha256-e71c9271166ad648 --architecture qwen3moe \
  --dtype UD-Q4_K_XL --format gguf
.venv/bin/python scripts/validate_model.py ./model
```

`make qwen-download` performs both the verified download and GGUF manifest
creation automatically.

Manifest paths are relative POSIX paths. Absolute paths, `..`, backslashes, duplicate entries, symlinks, unknown extensions, `.bin`, `.pt`, `.pth`, `.pkl`, `.pickle`, `.ckpt`, and `.py` are rejected. Strict mode rejects files omitted from the manifest.

## Target checkpoint qualification

Before claiming a supplied ~30B MoE works:

1. Preserve an immutable copy and record the manifest digest.
2. Inspect Transformers config/tokenizer/index metadata or GGUF metadata, including MoE expert/top-k fields and dtype/quantization.
3. Verify native support in the matching pinned engine without remote code.
4. Run preflight against actual free VRAM; total parameters—not active parameters—drive weight residency.
5. Load, warm, and execute deterministic non-streaming and streaming completions.
6. Execute cancellation and at least two concurrent requests.
7. Confirm `model_load_total` remains `1`.
8. Record startup time, TTFT, decode tokens/s, peak CPU RAM and GPU VRAM, context, and concurrency.
9. Keep model revision, engine/package versions, GPU/CUDA, dtype, and all warnings in the evidence report.

Until this phase passes, the correct status is: “Inference lifecycle and runtime integration verified; target ~30B MoE quality and hardware-specific performance not yet qualified.”

## Companion Qwen GGUF

The approved GCP pointer for the companion local-coder project's Qwen3-Coder
30B-A3B UD-Q4_K_XL artifact is documented in
[`QWEN_GCP_MODEL.md`](QWEN_GCP_MODEL.md). It is the checked GGUF profile for the
`llama_cpp` engine.
