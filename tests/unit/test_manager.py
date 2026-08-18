from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from inference_service.engines.base import GenerationRequest, GenerationResult, StreamChunk
from inference_service.engines.mock import MockInferenceEngine
from inference_service.model.manager import ModelManager
from inference_service.observability.metrics import Metrics
from tests.conftest import make_settings


class CancellationAwareEngine(MockInferenceEngine):
    def __init__(self) -> None:
        super().__init__()
        self.stream_finalized = False

    async def stream(self, request: GenerationRequest) -> AsyncIterator[StreamChunk]:
        try:
            yield StreamChunk(text="first")
            yield StreamChunk(text="second")
        finally:
            self.stream_finalized = True


class WarmupFailureEngine(MockInferenceEngine):
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        raise RuntimeError("warmup failed")


@pytest.mark.asyncio
async def test_stream_cancellation_releases_capacity(mock_model_dir: Path) -> None:
    engine = CancellationAwareEngine()
    manager = ModelManager(
        make_settings(mock_model_dir, max_concurrent_requests=1),
        Metrics(),
        engine_factory=lambda _: engine,
    )
    await manager.initialize()
    request = GenerationRequest(
        messages=({"role": "user", "content": "hello"},),
        max_new_tokens=4,
        temperature=0,
    )
    stream = manager.stream(request)
    assert (await anext(stream)).text == "first"
    await stream.aclose()
    assert engine.stream_finalized
    # Capacity must be reusable immediately after cancellation.
    result = await manager.generate(request)
    assert result.completion_tokens > 0
    await manager.shutdown()


@pytest.mark.asyncio
async def test_initialize_is_idempotent_and_loads_once(mock_model_dir: Path) -> None:
    engine = MockInferenceEngine()
    manager = ModelManager(
        make_settings(mock_model_dir), Metrics(), engine_factory=lambda _: engine
    )
    await manager.initialize()
    await manager.initialize()
    assert engine.load_calls == 1
    await manager.shutdown()


@pytest.mark.asyncio
async def test_initialization_failure_unloads_engine(mock_model_dir: Path) -> None:
    engine = WarmupFailureEngine()
    manager = ModelManager(
        make_settings(mock_model_dir), Metrics(), engine_factory=lambda _: engine
    )
    with pytest.raises(RuntimeError, match="warmup failed"):
        await manager.initialize()
    assert not engine.loaded
    assert manager.error == "RuntimeError: warmup failed"
