from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from inference_service.config import Settings
from inference_service.engines.mock import MockInferenceEngine
from inference_service.main import create_app
from inference_service.model.manifest import build_manifest, write_manifest


@pytest.fixture
def mock_model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "mock-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"mock"}\n', encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"safe-test-placeholder")
    manifest = build_manifest(
        model_dir,
        model_id="mock-test-model",
        revision="test-001",
        architecture="MockForCausalLM",
        dtype="float32",
    )
    write_manifest(model_dir, manifest)
    return model_dir


def make_settings(model_dir: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "model_source": str(model_dir),
        "model_dtype": "float32",
        "device": "cpu",
        "inference_engine": "mock",
        "environment": "test",
        "allow_unauthenticated": True,
        "model_max_context": 128,
        "model_max_new_tokens": 16,
        "max_prompt_tokens": 96,
        "max_batch_tokens": 128,
        "warmup_max_new_tokens": 1,
        "request_queue_timeout_seconds": 0.05,
    }
    values.update(overrides)
    return Settings(**values)


def wait_until_ready(client: TestClient, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get("/health/ready")
        if response.status_code == 200:
            return
        if response.json().get("model_state") == "failed":
            manager = client.app.state.manager
            raise AssertionError(f"model initialization failed: {manager.error}")
        time.sleep(0.02)
    raise AssertionError(f"model did not become ready: {client.get('/health/ready').json()}")


@pytest.fixture
def client(mock_model_dir: Path) -> Iterator[TestClient]:
    settings = make_settings(mock_model_dir)
    app = create_app(settings, engine_factory=lambda _: MockInferenceEngine("service works"))
    with TestClient(app) as test_client:
        wait_until_ready(test_client)
        yield test_client
