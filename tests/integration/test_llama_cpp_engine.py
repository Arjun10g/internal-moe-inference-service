from __future__ import annotations

from pathlib import Path

import pytest

from inference_service.config import Settings
from inference_service.engines.base import GenerationRequest
from inference_service.model.manager import ModelManager
from inference_service.model.manifest import build_manifest, write_manifest
from inference_service.observability.metrics import Metrics

FAKE_LLAMA_SERVER = r"""#!/usr/bin/env python3
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

if "--version" in sys.argv:
    print("llama.cpp version 10355 (dd1ea524333b)")
    raise SystemExit(0)

port = int(sys.argv[sys.argv.index("--port") + 1])
api_key = os.environ["LLAMA_API_KEY"]

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def authorized(self):
        return self.headers.get("Authorization") == f"Bearer {{api_key}}"

    def send_json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health" and self.authorized():
            self.send_json(200, {{"status": "ok"}})
        else:
            self.send_json(401, {{"error": "unauthorized"}})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        if not self.authorized():
            self.send_json(401, {{"error": "unauthorized"}})
            return
        if payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            events = [
                {{"choices": [{{"delta": {{"content": "fake "}}, "finish_reason": None}}]}},
                {{
                    "choices": [{{"delta": {{"content": "response"}}, "finish_reason": "stop"}}],
                    "usage": {{"prompt_tokens": 3, "completion_tokens": 2}},
                }},
            ]
            for event in events:
                self.wfile.write(f"data: {{json.dumps(event)}}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        self.send_json(200, {{
            "choices": [{{
                "message": {{"role": "assistant", "content": "fake response"}},
                "finish_reason": "stop",
            }}],
            "usage": {{"prompt_tokens": 3, "completion_tokens": 2}},
        }})

ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""


def create_fake_runtime(path: Path) -> Path:
    path.write_text(FAKE_LLAMA_SERVER.format(), encoding="utf-8")
    path.chmod(0o755)
    return path


def create_gguf_model(path: Path) -> Path:
    path.mkdir()
    (path / "model.gguf").write_bytes(b"GGUF-test-model")
    manifest = build_manifest(
        path,
        model_id="qwen-gguf-test",
        revision="test-001",
        architecture="qwen3moe",
        dtype="UD-Q4_K_XL",
        model_format="gguf",
    )
    write_manifest(path, manifest)
    return path


@pytest.mark.asyncio
async def test_llama_cpp_process_generation_and_streaming(tmp_path: Path) -> None:
    runtime = create_fake_runtime(tmp_path / "llama-server")
    model_dir = create_gguf_model(tmp_path / "model")
    settings = Settings(
        model_source=str(model_dir),
        model_id="fallback-id",
        model_format="gguf",
        inference_engine="llama_cpp",
        device="cpu",
        environment="test",
        allow_unauthenticated=True,
        model_max_context=128,
        model_max_new_tokens=16,
        max_prompt_tokens=96,
        max_batch_tokens=128,
        max_concurrent_requests=1,
        llama_server_path=runtime,
        llama_cpp_batch_size=64,
        llama_cpp_ubatch_size=32,
        warmup_max_new_tokens=1,
    )
    manager = ModelManager(settings, Metrics())
    await manager.initialize()
    request = GenerationRequest(
        messages=({"role": "user", "content": "hello"},),
        max_new_tokens=4,
        temperature=0,
    )

    result = await manager.generate(request)
    assert result.text == "fake response"
    assert result.prompt_tokens == 3
    assert result.completion_tokens == 2

    chunks = [chunk async for chunk in manager.stream(request)]
    assert "".join(chunk.text for chunk in chunks) == "fake response"
    assert chunks[-1].done
    assert chunks[-1].generated_tokens == 2
    assert manager.info()["engine"] == "llama_cpp"
    assert manager.info()["model_id"] == "qwen-gguf-test"

    engine = manager.engine
    await manager.shutdown()
    assert engine.process is None
