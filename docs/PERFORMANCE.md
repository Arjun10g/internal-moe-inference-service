# Performance and capacity

Benchmark the exact model format, engine, GPU, driver, runtime build, dtype/quantization, context distribution, and concurrency before choosing production task sizing. The Transformers backend is the safetensors reference; the approved Qwen GGUF uses pinned llama.cpp.

Measure at minimum startup/load/warmup time, prompt tokens, output tokens, TTFT, decode tokens/s, request P50/P95/P99, peak CPU RAM, peak VRAM, saturation/429 rate, and cancellation latency.

```bash
.venv/bin/python scripts/benchmark.py \
  --base-url http://127.0.0.1:8000 --api-key "$API_KEY" \
  --requests 20 --concurrency 2 --max-tokens 64
```

MoE active parameters can reduce per-token matrix work but do not normally reduce resident checkpoint bytes. KV cache scales with layers, hidden dimensions, context, dtype, and concurrent sequences. Leave capacity for CUDA graphs/workspaces, allocator fragmentation, tokenizer/process RAM, and transient prefill activations.

The service does not silently quantize, shard, or truncate context. The GGUF backend explicitly controls GPU-layer offload and KV-cache type; changes from the checked Qwen profile must be revalidated against memory, quality, and latency.
