from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from inference_service.engines.base import (
    GenerationRequest,
    GenerationResult,
    InferenceEngine,
    StreamChunk,
)


class MockInferenceEngine(InferenceEngine):
    """Deterministic test-only engine; production configuration rejects it."""

    def __init__(self, text: str = "mock completion") -> None:
        self.text = text
        self.loaded = False
        self.load_calls = 0

    def load(self, model_path: Path) -> dict[str, Any]:
        self.loaded = True
        self.load_calls += 1
        return self.metadata

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        if not self.loaded:
            raise RuntimeError("mock engine not loaded")
        await asyncio.sleep(0)
        return GenerationResult(
            text=self.text,
            prompt_tokens=sum(len(message["content"].split()) for message in request.messages),
            completion_tokens=len(self.text.split()),
            finish_reason="stop",
            elapsed_seconds=0.001,
        )

    async def stream(self, request: GenerationRequest) -> AsyncIterator[StreamChunk]:
        result = await self.generate(request)
        for part in self.text.split(" "):
            yield StreamChunk(text=part + " ", generated_tokens=1)
        yield StreamChunk(
            done=True,
            prompt_tokens=result.prompt_tokens,
            generated_tokens=result.completion_tokens,
            finish_reason=result.finish_reason,
            ttft_seconds=0.001,
        )

    def unload(self) -> None:
        self.loaded = False

    @property
    def metadata(self) -> dict[str, Any]:
        return {"engine": "mock", "architecture": "mock", "device": "cpu", "dtype": "none"}
