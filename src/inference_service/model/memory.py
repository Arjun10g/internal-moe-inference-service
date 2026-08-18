from __future__ import annotations

from dataclasses import dataclass

DTYPE_BYTES = {"float32": 4.0, "float16": 2.0, "bfloat16": 2.0, "int8": 1.0, "int4": 0.5}


@dataclass(frozen=True)
class MemoryEstimate:
    weights_gib: float
    kv_cache_gib: float
    runtime_overhead_gib: float
    total_gib: float


def estimate_memory(
    *,
    total_parameters: int,
    dtype: str,
    hidden_size: int,
    num_layers: int,
    context_tokens: int,
    concurrent_sequences: int,
) -> MemoryEstimate:
    bytes_per_value = DTYPE_BYTES.get(dtype)
    if bytes_per_value is None:
        raise ValueError(f"unsupported dtype for estimation: {dtype}")
    weights = total_parameters * bytes_per_value
    # K and V are both retained per layer; this intentionally uses total model layers.
    kv = 2 * num_layers * hidden_size * context_tokens * concurrent_sequences * bytes_per_value
    overhead = weights * 0.15
    gib = 1024**3
    return MemoryEstimate(
        weights_gib=weights / gib,
        kv_cache_gib=kv / gib,
        runtime_overhead_gib=overhead / gib,
        total_gib=(weights + kv + overhead) / gib,
    )
