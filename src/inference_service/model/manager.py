from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from enum import StrEnum
from pathlib import Path
from typing import Any

from inference_service.config import EngineName, Settings
from inference_service.diagnostics import runtime_diagnostics
from inference_service.engines import (
    InferenceEngine,
    LlamaCppInferenceEngine,
    MockInferenceEngine,
    TransformersInferenceEngine,
)
from inference_service.engines.base import GenerationRequest, GenerationResult, StreamChunk
from inference_service.model.validation import ValidationSummary, validate_model_directory
from inference_service.observability.metrics import Metrics
from inference_service.storage import storage_for_source

logger = logging.getLogger(__name__)


class ModelState(StrEnum):
    NEW = "new"
    RESOLVING = "resolving"
    VALIDATING = "validating"
    LOADING = "loading"
    WARMING = "warming"
    READY = "ready"
    FAILED = "failed"
    STOPPING = "stopping"
    STOPPED = "stopped"


EngineFactory = Callable[[Settings], InferenceEngine]


class CapacityError(RuntimeError):
    """Raised when bounded inference capacity cannot be acquired promptly."""


def default_engine_factory(settings: Settings) -> InferenceEngine:
    if settings.inference_engine == EngineName.MOCK:
        return MockInferenceEngine()
    if settings.inference_engine == EngineName.LLAMA_CPP:
        return LlamaCppInferenceEngine(settings)
    return TransformersInferenceEngine(settings)


class ModelManager:
    def __init__(
        self,
        settings: Settings,
        metrics: Metrics,
        *,
        engine_factory: EngineFactory = default_engine_factory,
    ) -> None:
        self.settings = settings
        self.metrics = metrics
        self.engine = engine_factory(settings)
        self.state = ModelState.NEW
        self.error: str | None = None
        self.model_path: Path | None = None
        self.validation: ValidationSummary | None = None
        self.loaded_at: float | None = None
        self.startup_seconds: float | None = None
        self.diagnostics: dict[str, Any] = {}
        self._initialization_lock = asyncio.Lock()
        self._capacity = asyncio.Semaphore(settings.max_concurrent_requests)

    async def initialize(self) -> None:
        async with self._initialization_lock:
            if self.state == ModelState.READY:
                return
            if self.state not in {ModelState.NEW, ModelState.FAILED}:
                raise RuntimeError(f"cannot initialize model from state {self.state}")
            started = time.perf_counter()
            self.error = None
            try:
                self.state = ModelState.RESOLVING
                storage = storage_for_source(
                    self.settings.model_source,
                    self.settings.model_cache_dir,
                    strict=self.settings.strict_manifest,
                )
                self.model_path = await asyncio.to_thread(storage.resolve)
                self.state = ModelState.VALIDATING
                self.validation = await asyncio.to_thread(
                    validate_model_directory,
                    self.model_path,
                    strict=self.settings.strict_manifest,
                )
                if self.validation.manifest.model_format != self.settings.model_format:
                    raise ValueError(
                        "manifest model_format does not match configured MODEL_FORMAT: "
                        f"{self.validation.manifest.model_format} != {self.settings.model_format}"
                    )
                self.diagnostics = runtime_diagnostics(
                    self.settings, artifact_bytes=self.validation.total_bytes
                )
                logger.info("runtime_preflight", extra={"diagnostics": self.diagnostics})
                self.state = ModelState.LOADING
                await asyncio.to_thread(self.engine.load, self.model_path)
                self.metrics.model_loads.inc()
                self.state = ModelState.WARMING
                await self.engine.generate(
                    GenerationRequest(
                        messages=({"role": "user", "content": self.settings.warmup_prompt},),
                        max_new_tokens=self.settings.warmup_max_new_tokens,
                        temperature=0.0,
                    )
                )
                self.loaded_at = time.time()
                self.startup_seconds = time.perf_counter() - started
                self.state = ModelState.READY
                self.metrics.model_ready.set(1)
                self.metrics.model_startup.observe(self.startup_seconds)
                logger.info(
                    "model_ready",
                    extra={
                        "model_id": self.validation.manifest.model_id,
                        "revision": self.validation.manifest.revision,
                        "startup_seconds": self.startup_seconds,
                        "engine": self.engine.metadata.get("engine"),
                    },
                )
            except BaseException as exc:
                self.state = ModelState.FAILED
                self.error = f"{type(exc).__name__}: {exc}"
                self.metrics.model_ready.set(0)
                self.metrics.model_load_failures.inc()
                logger.exception("model_initialization_failed")
                try:
                    await asyncio.to_thread(self.engine.unload)
                except Exception:
                    logger.exception("model_cleanup_after_initialization_failure_failed")
                raise

    def _require_ready(self) -> None:
        if self.state != ModelState.READY:
            raise RuntimeError(f"model is not ready (state={self.state})")

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self._require_ready()
        try:
            await asyncio.wait_for(
                self._capacity.acquire(), timeout=self.settings.request_queue_timeout_seconds
            )
        except TimeoutError as exc:
            raise CapacityError("inference capacity is exhausted") from exc
        try:
            self.metrics.in_progress.inc()
            try:
                return await self.engine.generate(request)
            finally:
                self.metrics.in_progress.dec()
        finally:
            self._capacity.release()

    async def stream(self, request: GenerationRequest) -> AsyncIterator[StreamChunk]:
        self._require_ready()
        try:
            await asyncio.wait_for(
                self._capacity.acquire(), timeout=self.settings.request_queue_timeout_seconds
            )
        except TimeoutError as exc:
            raise CapacityError("inference capacity is exhausted") from exc
        try:
            self.metrics.in_progress.inc()
            engine_stream = self.engine.stream(request)
            try:
                async for chunk in engine_stream:
                    yield chunk
            finally:
                close = getattr(engine_stream, "aclose", None)
                if close is not None:
                    await close()
                self.metrics.in_progress.dec()
        finally:
            self._capacity.release()

    async def shutdown(self) -> None:
        if self.state in {ModelState.STOPPING, ModelState.STOPPED}:
            return
        self.state = ModelState.STOPPING
        self.metrics.model_ready.set(0)
        await asyncio.to_thread(self.engine.unload)
        self.state = ModelState.STOPPED

    @property
    def ready(self) -> bool:
        return self.state == ModelState.READY

    def info(self) -> dict[str, Any]:
        manifest = self.validation.manifest if self.validation else None
        return {
            "model_id": manifest.model_id if manifest else self.settings.model_id,
            "revision": manifest.revision if manifest else self.settings.model_revision,
            "architecture": manifest.architecture if manifest else None,
            "engine": self.engine.metadata.get("engine", self.settings.inference_engine.value),
            "device": self.engine.metadata.get("device", self.settings.device),
            "dtype": self.engine.metadata.get("dtype", self.settings.model_dtype),
            "state": self.state.value,
            "loaded_at": self.loaded_at,
            "metadata": {
                **self.engine.metadata,
                "startup_seconds": self.startup_seconds,
                "file_count": self.validation.file_count if self.validation else None,
                "artifact_bytes": self.validation.total_bytes if self.validation else None,
            },
        }
