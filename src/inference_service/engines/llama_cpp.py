from __future__ import annotations

import json
import os
import secrets
import shutil
import socket
import subprocess  # nosec B404
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import httpx

from inference_service.config import Settings
from inference_service.engines.base import (
    GenerationRequest,
    GenerationResult,
    InferenceEngine,
    StreamChunk,
)
from inference_service.model.manifest import load_manifest

PINNED_LLAMA_CPP_BUILD = "b10355"
PINNED_LLAMA_CPP_COMMIT = "dd1ea524333b1e697489067d7a4c39c60d32beee"


def _free_loopback_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _content_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            str(part["text"])
            for part in value
            if isinstance(part, Mapping) and isinstance(part.get("text"), str)
        )
    return ""


def _finish_reason(
    value: object, completion_tokens: int, maximum: int
) -> Literal["stop", "length", "cancelled", "error"]:
    if value in {"stop", "length", "cancelled", "error"}:
        return value
    return "length" if completion_tokens >= maximum else "stop"


class LlamaCppInferenceEngine(InferenceEngine):
    """Supervise a pinned llama-server and proxy its loopback OpenAI API."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.process: subprocess.Popen[str] | None = None
        self._binary: Path | None = None
        self._base_url: str | None = None
        self._api_key: str | None = None
        self._metadata: dict[str, Any] = {"engine": "llama_cpp", "format": "gguf"}
        self._model_id = settings.model_id
        self._logs: deque[str] = deque(maxlen=80)
        self._log_threads: list[threading.Thread] = []

    def _resolve_binary(self) -> Path:
        configured = self.settings.llama_server_path.expanduser()
        raw = str(configured)
        resolved = shutil.which(raw) if configured.parent == Path(".") else raw
        if resolved is None:
            raise RuntimeError(f"llama-server was not found: {configured}")
        binary = Path(resolved).resolve(strict=True)
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise RuntimeError(f"llama-server is not executable: {binary}")
        return binary

    def _runtime_environment(self, binary: Path) -> dict[str, str]:
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.upper().startswith(("LLAMA_", "GGML_", "HF_", "HUGGINGFACE_"))
            and key not in {"LD_PRELOAD", "LD_LIBRARY_PATH"}
            and not key.startswith("DYLD_")
        }
        if self._api_key is None:
            raise RuntimeError("llama.cpp API key was not initialized")
        environment.update(
            {
                "LLAMA_API_KEY": self._api_key,
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
            }
        )
        if sys_platform() == "linux":
            environment["LD_LIBRARY_PATH"] = str(binary.parent)
        elif sys_platform() == "darwin":
            environment["DYLD_LIBRARY_PATH"] = str(binary.parent)
        return environment

    def _version(self, binary: Path, environment: dict[str, str]) -> str:
        result = subprocess.run(  # noqa: S603  # nosec B603
            [str(binary), "--version"],
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        output = (result.stdout + result.stderr).strip()
        if result.returncode != 0 or not output:
            raise RuntimeError("llama-server --version failed")
        if self.settings.llama_cpp_require_pinned_runtime and not (
            "10355" in output or PINNED_LLAMA_CPP_COMMIT[:12] in output
        ):
            raise RuntimeError(
                f"llama-server must be {PINNED_LLAMA_CPP_BUILD} "
                f"({PINNED_LLAMA_CPP_COMMIT[:12]}); got {output.splitlines()[0]}"
            )
        return output.splitlines()[0]

    def _threads(self) -> int:
        if self.settings.llama_cpp_threads:
            return self.settings.llama_cpp_threads
        available = os.cpu_count() or 2
        return max(2, min(16, int(available * 0.75)))

    def _arguments(self, model_file: Path, port: int) -> list[str]:
        threads = self._threads()
        arguments = [
            "--model",
            str(model_file),
            "--alias",
            self._model_id,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--ctx-size",
            str(self.settings.model_max_context),
            "--threads",
            str(threads),
            "--threads-batch",
            str(threads),
            "--batch-size",
            str(self.settings.llama_cpp_batch_size),
            "--ubatch-size",
            str(self.settings.llama_cpp_ubatch_size),
            "--parallel",
            "1",
            "--cache-type-k",
            self.settings.llama_cpp_kv_cache_type,
            "--cache-type-v",
            self.settings.llama_cpp_kv_cache_type,
            "--load-mode",
            "mmap",
            "--flash-attn",
            "auto",
            "--jinja",
            "--no-webui",
            "--no-agent",
            "--offline",
            "--cors-origins",
            "localhost",
            "--no-cors-credentials",
            "--no-slots",
            "--log-colors",
            "off",
            "--log-timestamps",
        ]
        if not self.settings.llama_cpp_repack:
            arguments.append("--no-repack")
        gpu_layers = 0 if self.settings.device == "cpu" else self.settings.llama_cpp_gpu_layers
        arguments.extend(("--n-gpu-layers", str(gpu_layers)))
        return arguments

    def _drain(self, stream: Any, source: str) -> None:
        try:
            for line in stream:
                stripped = str(line).strip()
                if stripped:
                    self._logs.append(f"{source}: {stripped}")
        finally:
            stream.close()

    def _start_log_threads(self, process: subprocess.Popen[str]) -> None:
        self._log_threads = []
        for source, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
            if stream is None:
                continue
            thread = threading.Thread(
                target=self._drain,
                args=(stream, source),
                name=f"llama-server-{source}",
                daemon=True,
            )
            thread.start()
            self._log_threads.append(thread)

    def _wait_until_ready(self) -> None:
        if self.process is None or self._base_url is None or self._api_key is None:
            raise RuntimeError("llama-server process was not initialized")
        deadline = time.monotonic() + self.settings.llama_cpp_startup_timeout_seconds
        headers = {"Authorization": f"Bearer {self._api_key}"}
        with httpx.Client(
            base_url=self._base_url, headers=headers, trust_env=False, timeout=3
        ) as client:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    detail = "\n".join(self._logs)[-4000:]
                    raise RuntimeError(f"llama-server exited during startup\n{detail}")
                try:
                    response = client.get("/health")
                    if response.status_code == 200 and response.json().get("status") == "ok":
                        return
                except (httpx.HTTPError, json.JSONDecodeError, ValueError):
                    pass
                time.sleep(0.25)
        raise RuntimeError("llama-server did not become ready before the startup timeout")

    def load(self, model_path: Path) -> dict[str, Any]:
        if self.process is not None:
            raise RuntimeError("llama-server is already loaded")
        model_files = sorted(model_path.rglob("*.gguf"))
        if len(model_files) != 1:
            raise ValueError("GGUF model directory must contain exactly one .gguf file")
        manifest = load_manifest(model_path)
        if manifest.model_format != "gguf":
            raise ValueError("llama.cpp requires a GGUF model manifest")
        self._model_id = manifest.model_id

        binary = self._resolve_binary()
        self._api_key = secrets.token_urlsafe(32)
        environment = self._runtime_environment(binary)
        runtime_version = self._version(binary, environment)
        port = _free_loopback_port()
        self._base_url = f"http://127.0.0.1:{port}"
        arguments = self._arguments(model_files[0], port)
        self._logs.clear()
        try:
            self.process = subprocess.Popen(  # noqa: S603  # nosec B603
                [str(binary), *arguments],
                cwd=model_path,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self._start_log_threads(self.process)
            self._wait_until_ready()
        except BaseException:
            self.unload()
            raise
        self._binary = binary
        self._metadata = {
            "engine": "llama_cpp",
            "format": "gguf",
            "device": self.settings.device,
            "dtype": "quantized",
            "runtime_build": PINNED_LLAMA_CPP_BUILD,
            "runtime_commit": PINNED_LLAMA_CPP_COMMIT,
            "runtime_version": runtime_version,
            "threads": self._threads(),
            "context_size": self.settings.model_max_context,
            "kv_cache_type": self.settings.llama_cpp_kv_cache_type,
            "gpu_layers": 0
            if self.settings.device == "cpu"
            else self.settings.llama_cpp_gpu_layers,
        }
        return self.metadata

    def _client(self) -> httpx.AsyncClient:
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("llama-server is not running")
        if self._base_url is None or self._api_key is None:
            raise RuntimeError("llama-server connection is not initialized")
        return httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            trust_env=False,
            timeout=httpx.Timeout(self.settings.generation_timeout_seconds, connect=10.0),
        )

    def _payload(self, request: GenerationRequest, *, stream: bool) -> dict[str, object]:
        return {
            "model": self._model_id,
            "messages": list(request.messages),
            "max_tokens": request.max_new_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stop": list(request.stop),
            "stream": stream,
            **({"stream_options": {"include_usage": True}} if stream else {}),
        }

    @staticmethod
    async def _raise_for_status(response: httpx.Response) -> None:
        if response.is_success:
            return
        await response.aread()
        if response.status_code in {400, 404, 413, 422}:
            raise ValueError("llama.cpp rejected the generation request")
        raise RuntimeError(f"llama.cpp generation failed with HTTP {response.status_code}")

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        started = time.perf_counter()
        try:
            async with self._client() as client:
                response = await client.post(
                    "/v1/chat/completions", json=self._payload(request, stream=False)
                )
                await self._raise_for_status(response)
                payload = response.json()
        except httpx.HTTPError as exc:
            raise RuntimeError("llama.cpp generation request failed") from exc
        try:
            choice = payload["choices"][0]
            message = choice["message"]
            text = _content_text(message.get("content"))
            usage = payload.get("usage") or {}
            timings = payload.get("timings") or {}
            prompt_tokens = int(usage.get("prompt_tokens", timings.get("prompt_n", 0)))
            completion_tokens = int(usage.get("completion_tokens", timings.get("predicted_n", 0)))
            reason = _finish_reason(
                choice.get("finish_reason"), completion_tokens, request.max_new_tokens
            )
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("llama.cpp returned an invalid completion response") from exc
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=reason,
            elapsed_seconds=time.perf_counter() - started,
        )

    async def stream(self, request: GenerationRequest) -> AsyncIterator[StreamChunk]:
        started = time.perf_counter()
        first_at: float | None = None
        prompt_tokens = 0
        completion_tokens = 0
        reason: object = None
        saw_done = False
        try:
            async with (
                self._client() as client,
                client.stream(
                    "POST",
                    "/v1/chat/completions",
                    json=self._payload(request, stream=True),
                ) as response,
            ):
                await self._raise_for_status(response)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        saw_done = True
                        break
                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError("llama.cpp returned invalid SSE JSON") from exc
                    if payload.get("error"):
                        raise RuntimeError("llama.cpp returned a streaming generation error")
                    usage = payload.get("usage") or {}
                    timings = payload.get("timings") or {}
                    prompt_tokens = int(
                        usage.get("prompt_tokens", timings.get("prompt_n", prompt_tokens))
                    )
                    completion_tokens = int(
                        usage.get(
                            "completion_tokens",
                            timings.get("predicted_n", completion_tokens),
                        )
                    )
                    choices: Sequence[Mapping[str, object]] = payload.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    reason = choice.get("finish_reason") or reason
                    delta = choice.get("delta") or {}
                    text = _content_text(
                        delta.get("content") if isinstance(delta, Mapping) else None
                    )
                    if text:
                        if first_at is None:
                            first_at = time.perf_counter()
                        yield StreamChunk(text=text)
        except httpx.HTTPError as exc:
            raise RuntimeError("llama.cpp streaming request failed") from exc
        if not saw_done:
            raise RuntimeError("llama.cpp stream ended without a [DONE] sentinel")
        yield StreamChunk(
            done=True,
            prompt_tokens=prompt_tokens,
            generated_tokens=completion_tokens,
            finish_reason=_finish_reason(reason, completion_tokens, request.max_new_tokens),
            ttft_seconds=(first_at or time.perf_counter()) - started,
        )

    def unload(self) -> None:
        process = self.process
        self.process = None
        self._base_url = None
        self._api_key = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for thread in self._log_threads:
            thread.join(timeout=1)
        self._log_threads = []

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)


def sys_platform() -> str:
    # Kept as a seam for platform-specific environment tests.
    import sys

    return sys.platform
