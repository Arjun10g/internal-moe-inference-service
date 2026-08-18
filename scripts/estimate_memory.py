#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from inference_service.model.memory import estimate_memory


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate checkpoint and KV-cache memory")
    parser.add_argument("--total-parameters", type=int, required=True)
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16", "int8", "int4"), required=True
    )
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    result = estimate_memory(
        total_parameters=args.total_parameters,
        dtype=args.dtype,
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        context_tokens=args.context,
        concurrent_sequences=args.concurrency,
    )
    print(json.dumps(result.__dict__, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
