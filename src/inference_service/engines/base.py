from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class GenerationRequest:
    messages: Sequence[dict[str, str]]
    max_new_tokens: int
    temperature: float
    top_p: float = 1.0
    stop: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: Literal["stop", "length", "cancelled", "error"]
    elapsed_seconds: float


@dataclass(frozen=True)
class StreamChunk:
    text: str = ""
    generated_tokens: int = 0
    prompt_tokens: int = 0
    finish_reason: Literal["stop", "length", "cancelled", "error"] | None = None
    done: bool = False
    ttft_seconds: float | None = None


class InferenceEngine(ABC):
    @abstractmethod
    def load(self, model_path: Path) -> dict[str, Any]:
        """Load the model exactly once."""

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate a complete response."""

    @abstractmethod
    def stream(self, request: GenerationRequest) -> AsyncIterator[StreamChunk]:
        """Generate a streamed response."""

    @abstractmethod
    def unload(self) -> None:
        """Release model resources during shutdown."""

    @property
    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return non-sensitive loaded model metadata."""
