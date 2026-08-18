from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any, Literal

from inference_service.config import Settings
from inference_service.engines.base import (
    GenerationRequest,
    GenerationResult,
    InferenceEngine,
    StreamChunk,
)

logger = logging.getLogger(__name__)


def _truncate_at_stop(text: str, stops: Sequence[str]) -> tuple[str, bool]:
    positions = [position for stop in stops if (position := text.find(stop)) >= 0]
    if not positions:
        return text, False
    return text[: min(positions)], True


class _BufferedStopFilter:
    """Remove stop strings from streamed text, including across chunk boundaries."""

    def __init__(self, stops: Sequence[str]) -> None:
        self.stops = tuple(stops)
        self.buffer = ""
        self.stopped = False
        self._tail_chars = max((len(stop) - 1 for stop in self.stops), default=0)

    def push(self, text: str) -> str:
        if self.stopped or not text:
            return ""
        self.buffer += text
        truncated, stopped = _truncate_at_stop(self.buffer, self.stops)
        if stopped:
            self.buffer = ""
            self.stopped = True
            return truncated
        if self._tail_chars == 0:
            emitted, self.buffer = self.buffer, ""
            return emitted
        if len(self.buffer) <= self._tail_chars:
            return ""
        emitted = self.buffer[: -self._tail_chars]
        self.buffer = self.buffer[-self._tail_chars :]
        return emitted

    def finish(self) -> str:
        if self.stopped:
            return ""
        emitted, self.buffer = self.buffer, ""
        return emitted


class TransformersInferenceEngine(InferenceEngine):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self._metadata: dict[str, Any] = {"engine": "transformers"}

    def _resolved_device(self, torch: Any) -> str:
        if self.settings.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.settings.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("DEVICE=cuda was requested but CUDA is not available")
        return self.settings.device

    def load(self, model_path: Path) -> dict[str, Any]:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        if self.model is not None:
            raise RuntimeError("model is already loaded")
        device = self._resolved_device(torch)
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[self.settings.model_dtype]
        if device == "cpu" and dtype == torch.float16:
            raise RuntimeError("float16 CPU inference is unsupported; use float32 or bfloat16")

        config = AutoConfig.from_pretrained(  # nosec B615
            model_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        tokenizer: Any = AutoTokenizer.from_pretrained(  # nosec B615
            model_path,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        model: Any = AutoModelForCausalLM.from_pretrained(  # nosec B615
            model_path,
            config=config,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            dtype=dtype,
            low_cpu_mem_usage=True,
            device_map={"": device},
        )
        model.eval()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.model = model
        self.tokenizer = tokenizer
        self._metadata = {
            "engine": "transformers",
            "architecture": (config.architectures or [config.model_type])[0],
            "model_type": config.model_type,
            "device": device,
            "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "transformers_version": __import__("transformers").__version__,
            "torch_version": torch.__version__,
        }
        return self.metadata

    def _prompt(self, messages: Sequence[dict[str, str]]) -> str:
        if self.tokenizer is None:
            raise RuntimeError("tokenizer is not loaded")
        try:
            return str(
                self.tokenizer.apply_chat_template(
                    list(messages), tokenize=False, add_generation_prompt=True
                )
            )
        except (ValueError, TypeError, AttributeError):
            rendered = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
            return rendered + "\nassistant:"

    def _inputs(self, request: GenerationRequest) -> tuple[dict[str, Any], int]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("model is not loaded")
        prompt = self._prompt(request.messages)
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        # Fast tokenizers can emit segment IDs that decoder-only models do not accept.
        encoded.pop("token_type_ids", None)
        prompt_tokens = int(encoded["input_ids"].shape[-1])
        if prompt_tokens > self.settings.max_prompt_tokens:
            raise ValueError(
                f"prompt contains {prompt_tokens} tokens; "
                f"maximum is {self.settings.max_prompt_tokens}"
            )
        device = next(self.model.parameters()).device
        return ({name: tensor.to(device) for name, tensor in encoded.items()}, prompt_tokens)

    def _generation_kwargs(self, request: GenerationRequest) -> dict[str, Any]:
        if self.tokenizer is None:
            raise RuntimeError("tokenizer is not loaded")
        kwargs: dict[str, Any] = {
            "max_new_tokens": request.max_new_tokens,
            "do_sample": request.temperature > 0,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if request.temperature > 0:
            kwargs["temperature"] = request.temperature
            kwargs["top_p"] = request.top_p
        return kwargs

    def _generate_sync(self, request: GenerationRequest) -> GenerationResult:
        import torch

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("model is not loaded")
        model = self.model
        inputs, prompt_tokens = self._inputs(request)
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, **self._generation_kwargs(request))
        generated = output[0, prompt_tokens:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        text, matched_stop = _truncate_at_stop(text, request.stop)
        completion_tokens = int(generated.shape[-1])
        finish_reason: Literal["length", "stop"] = (
            "stop" if matched_stop or completion_tokens < request.max_new_tokens else "length"
        )
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            elapsed_seconds=time.perf_counter() - start,
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        return await asyncio.to_thread(self._generate_sync, request)

    async def stream(self, request: GenerationRequest) -> AsyncIterator[StreamChunk]:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("model is not loaded")
        model = self.model
        inputs, prompt_tokens = self._inputs(request)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        cancelled = threading.Event()

        class CancellationCriteria(StoppingCriteria):
            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                return cancelled.is_set()

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=None,
        )
        kwargs = self._generation_kwargs(request)
        kwargs["streamer"] = streamer
        kwargs["stopping_criteria"] = StoppingCriteriaList([CancellationCriteria()])
        started = time.perf_counter()

        def worker() -> None:
            try:
                with torch.inference_mode():
                    output = model.generate(**inputs, **kwargs)
                generated_tokens = int(output.shape[-1] - prompt_tokens)
                loop.call_soon_threadsafe(queue.put_nowait, ("generated", generated_tokens))
            except BaseException as exc:  # propagated to async request task
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))

        def consume() -> None:
            try:
                for text in streamer:
                    loop.call_soon_threadsafe(queue.put_nowait, ("text", text))
                loop.call_soon_threadsafe(queue.put_nowait, ("stream_done", None))
            except BaseException as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))

        generation_thread = threading.Thread(target=worker, name="model-generate", daemon=True)
        stream_thread = threading.Thread(target=consume, name="model-stream", daemon=True)
        generation_thread.start()
        stream_thread.start()
        first_at: float | None = None
        generated_tokens: int | None = None
        stream_finished = False
        stop_filter = _BufferedStopFilter(request.stop)
        try:
            while True:
                kind, value = await queue.get()
                if kind == "error":
                    raise value
                if kind == "text":
                    filtered = stop_filter.push(str(value))
                    if stop_filter.stopped:
                        cancelled.set()
                    if filtered and first_at is None:
                        first_at = time.perf_counter()
                    if filtered:
                        yield StreamChunk(text=filtered)
                    continue
                if kind == "generated":
                    generated_tokens = int(value)
                elif kind == "stream_done":
                    stream_finished = True
                if generated_tokens is None or not stream_finished:
                    continue
                tail = stop_filter.finish()
                if tail:
                    if first_at is None:
                        first_at = time.perf_counter()
                    yield StreamChunk(text=tail)
                yield StreamChunk(
                    done=True,
                    prompt_tokens=prompt_tokens,
                    generated_tokens=generated_tokens,
                    finish_reason=(
                        "stop"
                        if stop_filter.stopped or generated_tokens < request.max_new_tokens
                        else "length"
                    ),
                    ttft_seconds=(first_at or time.perf_counter()) - started,
                )
                break
        finally:
            cancelled.set()
            await asyncio.to_thread(generation_thread.join, 5.0)
            await asyncio.to_thread(stream_thread.join, 5.0)

    def unload(self) -> None:
        model = self.model
        self.model = None
        self.tokenizer = None
        del model
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            logger.debug("torch_unavailable_during_shutdown")

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)
