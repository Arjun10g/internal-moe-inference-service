#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description="Small HTTP inference benchmark")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    payload = {
        "messages": [{"role": "user", "content": "Explain paged attention briefly."}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
    }

    def one(_: int) -> tuple[float, int]:
        started = time.perf_counter()
        with httpx.Client(trust_env=False, timeout=600) as client:
            response = client.post(
                f"{args.base_url}/v1/chat/completions", json=payload, headers=headers
            )
        response.raise_for_status()
        return time.perf_counter() - started, int(response.json()["usage"]["completion_tokens"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(one, range(args.requests)))
    latencies = sorted(item[0] for item in results)
    tokens = sum(item[1] for item in results)
    print(
        json.dumps(
            {
                "requests": args.requests,
                "concurrency": args.concurrency,
                "latency_p50_seconds": statistics.median(latencies),
                "latency_p99_seconds": latencies[max(0, int(len(latencies) * 0.99) - 1)],
                "generated_tokens": tokens,
                "aggregate_tokens_per_second": tokens / sum(latencies),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
