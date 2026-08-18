#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import socket
import subprocess  # nosec B404
import sys
import tempfile
import time
from pathlib import Path

import httpx


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="tiny-real-model-") as temp:
        model_dir = Path(temp) / "model"
        subprocess.run(  # nosec B603
            [sys.executable, str(repo / "scripts/create_tiny_test_model.py"), str(model_dir)],
            check=True,
            cwd=repo,
        )
        port = free_port()
        api_key = "local-smoke-key"
        env = {
            **os.environ,
            "MODEL_SOURCE": str(model_dir),
            "MODEL_DTYPE": "float32",
            "MODEL_MAX_CONTEXT": "128",
            "MAX_PROMPT_TOKENS": "96",
            "MODEL_MAX_NEW_TOKENS": "16",
            "MAX_BATCH_TOKENS": "128",
            "DEVICE": "cpu",
            "API_KEY": api_key,
            "ENVIRONMENT": "test",
            "PORT": str(port),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            # Large CPU pools make this deliberately tiny model slower and flaky.
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "1"),
        }
        process = subprocess.Popen(  # nosec B603
            [sys.executable, "-m", "inference_service.main"], cwd=repo, env=env
        )
        try:
            deadline = time.monotonic() + 90
            with httpx.Client(trust_env=False) as client:
                while time.monotonic() < deadline:
                    try:
                        response = client.get(f"http://127.0.0.1:{port}/health/ready", timeout=1)
                        if response.status_code == 200:
                            break
                    except httpx.HTTPError:
                        time.sleep(0.05)
                    if process.poll() is not None:
                        raise RuntimeError(f"service exited early with status {process.returncode}")
                    time.sleep(0.25)
                else:
                    raise TimeoutError("service did not become ready")
            result = subprocess.run(  # nosec B603
                [
                    sys.executable,
                    str(repo / "scripts/smoke_test.py"),
                    "--base-url",
                    f"http://127.0.0.1:{port}",
                    "--api-key",
                    api_key,
                ],
                check=True,
                cwd=repo,
                capture_output=True,
                text=True,
            )
            report = json.loads(result.stdout)
            report.update({"outbound_network_disabled": False, "offline_flags_enforced": True})
            print(json.dumps(report, indent=2))
            return 0
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
