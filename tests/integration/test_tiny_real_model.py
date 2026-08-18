from __future__ import annotations

import concurrent.futures
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from inference_service.config import Settings
from inference_service.main import create_app
from tests.conftest import wait_until_ready


@pytest.fixture(scope="module")
def tiny_moe_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("tiny-moe") / "model"
    repo = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }
    subprocess.run(
        [sys.executable, str(repo / "scripts/create_tiny_test_model.py"), str(output)],
        check=True,
        cwd=repo,
        env=env,
        timeout=120,
    )
    return output


def real_settings(model_dir: Path) -> Settings:
    return Settings(
        model_source=str(model_dir),
        model_dtype="float32",
        device="cpu",
        inference_engine="transformers",
        environment="test",
        allow_unauthenticated=True,
        model_max_context=128,
        model_max_new_tokens=8,
        max_prompt_tokens=96,
        max_batch_tokens=128,
        max_concurrent_requests=2,
        request_queue_timeout_seconds=20,
        generation_timeout_seconds=60,
        warmup_max_new_tokens=1,
    )


def payload(*, stream: bool = False) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": "The sky is"}],
        "max_tokens": 4,
        "temperature": 0,
        "stream": stream,
    }


@pytest.mark.tiny_model
def test_real_tiny_moe_end_to_end_without_network(
    tiny_moe_model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Thread-pool setup dominates this 41K-parameter model and can make the test
    # flaky on busy shared runners. One CPU thread is both faster and deterministic.
    import torch

    torch.set_num_threads(1)
    original_connect = socket.socket.connect

    def block_external(self: socket.socket, address: Any) -> Any:
        if isinstance(address, tuple) and address[0] not in {"127.0.0.1", "::1"}:
            raise AssertionError(f"unexpected outbound network connection: {address}")
        return original_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", block_external)
    app = create_app(real_settings(tiny_moe_model))
    with TestClient(app) as client:
        wait_until_ready(client, timeout=60)
        manager = app.state.manager
        engine_identity = id(manager.engine)
        model_identity = id(manager.engine.model)
        response = client.post("/v1/chat/completions", json=payload())
        assert response.status_code == 200, response.text
        first = response.json()
        assert first["usage"]["prompt_tokens"] > 0
        assert first["usage"]["completion_tokens"] == 4
        assert first["choices"][0]["finish_reason"] == "length"

        stream_payload = {**payload(stream=True), "stop": ["Hello blue model works"]}
        with client.stream("POST", "/v1/chat/completions", json=stream_payload) as stream:
            stream_text = "\n".join(stream.iter_lines())
        assert stream.status_code == 200
        assert '"content":' in stream_text
        assert "data: [DONE]" in stream_text
        assert '"completion_tokens":4' in stream_text

        def call_model(_: int) -> int:
            result = client.post("/v1/chat/completions", json=payload())
            assert result.status_code == 200, result.text
            return int(result.json()["usage"]["completion_tokens"])

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            assert list(pool.map(call_model, range(2))) == [4, 4]

        assert id(manager.engine) == engine_identity
        assert id(manager.engine.model) == model_identity
        model_info = client.get("/v1/models/current").json()
        assert model_info["metadata"]["model_type"] == "qwen2_moe"
        assert model_info["metadata"]["parameter_count"] > 0
        metrics = client.get("/metrics").text
        assert "model_load_total 1.0" in metrics
        assert not tiny_moe_model.joinpath("pytorch_model.bin").exists()
