from __future__ import annotations

import re
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from inference_service.engines.mock import MockInferenceEngine
from inference_service.main import create_app
from tests.conftest import make_settings, wait_until_ready


def request_body(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 4,
        "temperature": 0,
    }
    value.update(overrides)
    return value


def test_liveness_and_readiness(client: TestClient) -> None:
    assert client.get("/health/live").json()["status"] == "live"
    assert client.get("/health/ready").json() == {"status": "ready", "model_state": "ready"}


def test_non_streaming_completion(client: TestClient) -> None:
    response = client.post("/v1/chat/completions", json=request_body())
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "service works"
    assert body["usage"]["completion_tokens"] == 2
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"
    assert re.fullmatch(r"[0-9a-f]{32}", response.headers["x-request-id"])


def test_streaming_completion(client: TestClient) -> None:
    with client.stream("POST", "/v1/chat/completions", json=request_body(stream=True)) as response:
        text = "\n".join(response.iter_lines())
    assert response.status_code == 200
    assert "service " in text
    assert '"finish_reason":"stop"' in text
    assert "data: [DONE]" in text


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({}, 422),
        ({"messages": []}, 422),
        ({"messages": [{"role": "invalid", "content": "x"}]}, 422),
        ({"messages": [{"role": "user", "content": " "}]}, 422),
        (request_body(max_tokens=17), 422),
        (request_body(top_p=0), 422),
    ],
)
def test_invalid_requests(client: TestClient, body: dict[str, object], status: int) -> None:
    assert client.post("/v1/chat/completions", json=body).status_code == status


def test_invalid_json(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions", content=b"{broken", headers={"content-type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_model_endpoint_and_load_count(client: TestClient) -> None:
    model = client.get("/v1/models/current").json()
    assert model["model_id"] == "mock-test-model"
    for _ in range(3):
        assert client.post("/v1/chat/completions", json=request_body()).status_code == 200
    metrics = client.get("/metrics").text
    assert "model_load_total 1.0" in metrics
    listing = client.get("/v1/models").json()
    assert listing["object"] == "list"
    assert listing["data"][0]["id"] == "mock-test-model"


def test_authentication_required(mock_model_dir: Path) -> None:
    settings = make_settings(
        mock_model_dir,
        allow_unauthenticated=False,
        api_key="correct-secret",
    )
    app = create_app(settings, engine_factory=lambda _: MockInferenceEngine())
    with TestClient(app) as auth_client:
        wait_until_ready(auth_client)
        assert auth_client.post("/v1/chat/completions", json=request_body()).status_code == 401
        assert (
            auth_client.post(
                "/v1/chat/completions",
                json=request_body(),
                headers={"Authorization": "Bearer wrong"},
            ).status_code
            == 401
        )
        assert (
            auth_client.post(
                "/v1/chat/completions",
                json=request_body(),
                headers={"Authorization": "Bearer correct-secret"},
            ).status_code
            == 200
        )


def test_oversized_content_length(mock_model_dir: Path) -> None:
    settings = make_settings(mock_model_dir, max_request_bytes=1024)
    app = create_app(settings, engine_factory=lambda _: MockInferenceEngine())
    with TestClient(app) as size_client:
        wait_until_ready(size_client)
        response = size_client.post(
            "/v1/chat/completions",
            content=b"x" * 1025,
            headers={"content-type": "application/json", "content-length": "1025"},
        )
        assert response.status_code == 413


class SlowMockEngine(MockInferenceEngine):
    async def generate(self, request):  # type: ignore[no-untyped-def]
        import asyncio

        await asyncio.sleep(0.2)
        return await super().generate(request)


class FailAfterWarmupEngine(MockInferenceEngine):
    async def generate(self, request):  # type: ignore[no-untyped-def]
        if self.load_calls and getattr(self, "generation_calls", 0) >= 1:
            raise RuntimeError("synthetic engine failure")
        self.generation_calls = getattr(self, "generation_calls", 0) + 1
        return await super().generate(request)


def test_capacity_limit_returns_429(mock_model_dir: Path) -> None:
    import concurrent.futures

    settings = make_settings(
        mock_model_dir,
        max_concurrent_requests=1,
        request_queue_timeout_seconds=0.01,
    )
    app = create_app(settings, engine_factory=lambda _: SlowMockEngine())
    with TestClient(app) as limited_client:
        wait_until_ready(limited_client)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(limited_client.post, "/v1/chat/completions", json=request_body())
                for _ in range(2)
            ]
            statuses = sorted(future.result().status_code for future in futures)
        assert statuses == [200, 429]


def test_generation_timeout_returns_504(mock_model_dir: Path) -> None:
    settings = make_settings(mock_model_dir, generation_timeout_seconds=0.01)
    app = create_app(settings, engine_factory=lambda _: SlowMockEngine())
    with TestClient(app) as timeout_client:
        wait_until_ready(timeout_client)
        response = timeout_client.post("/v1/chat/completions", json=request_body())
        assert response.status_code == 504


def test_generation_failure_returns_503(mock_model_dir: Path) -> None:
    settings = make_settings(mock_model_dir)
    app = create_app(settings, engine_factory=lambda _: FailAfterWarmupEngine())
    with TestClient(app) as failure_client:
        wait_until_ready(failure_client)
        response = failure_client.post("/v1/chat/completions", json=request_body())
        assert response.status_code == 503


def test_graceful_shutdown_unloads_engine(mock_model_dir: Path) -> None:
    engine = MockInferenceEngine()
    app = create_app(make_settings(mock_model_dir), engine_factory=lambda _: engine)
    with TestClient(app) as shutdown_client:
        wait_until_ready(shutdown_client)
        assert engine.loaded
    assert not engine.loaded
    assert app.state.manager.state.value == "stopped"


def test_failed_initialization_is_live_but_not_ready(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    settings = make_settings(missing)
    app = create_app(settings, engine_factory=lambda _: MockInferenceEngine())
    with TestClient(app) as failed_client:
        deadline = time.monotonic() + 3
        while failed_client.get("/health/live").json()["model_state"] != "failed":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert failed_client.get("/health/live").status_code == 200
        assert failed_client.get("/health/ready").status_code == 503
