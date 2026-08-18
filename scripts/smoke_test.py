#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test a running inference service")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    payload = {
        "messages": [{"role": "user", "content": "The sky is"}],
        "max_tokens": 4,
        "temperature": 0,
    }
    with httpx.Client(
        base_url=args.base_url, headers=headers, timeout=args.timeout, trust_env=False
    ) as client:
        live = client.get("/health/live")
        live.raise_for_status()
        ready = client.get("/health/ready")
        ready.raise_for_status()
        listing = client.get("/v1/models")
        listing.raise_for_status()
        model = client.get("/v1/models/current")
        model.raise_for_status()
        first = client.post("/v1/chat/completions", json=payload)
        first.raise_for_status()
        first_body = first.json()
        if first_body["usage"]["completion_tokens"] < 1:
            raise RuntimeError("non-streaming completion generated zero tokens")
        second = client.post("/v1/chat/completions", json=payload)
        second.raise_for_status()
        with client.stream(
            "POST", "/v1/chat/completions", json={**payload, "stream": True}
        ) as response:
            response.raise_for_status()
            stream_text = "\n".join(response.iter_lines())
        if (
            "data: [DONE]" not in stream_text
            or '"completion_tokens":' not in stream_text
            or '"content":' not in stream_text
        ):
            raise RuntimeError("stream did not contain terminal usage and [DONE]")
        metrics = client.get("/metrics")
        metrics.raise_for_status()
        match = re.search(r"^model_load_total\s+([0-9.]+)$", metrics.text, re.MULTILINE)
        if match is None or float(match.group(1)) != 1.0:
            raise RuntimeError("model_load_total was not exactly 1")
    print(
        json.dumps(
            {
                "passed": True,
                "model": model.json(),
                "completion_tokens": first_body["usage"]["completion_tokens"],
                "requests": 3,
                "model_load_total": 1,
                "stream_done": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
