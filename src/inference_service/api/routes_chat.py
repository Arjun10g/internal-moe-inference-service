from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from inference_service.engines.base import GenerationRequest
from inference_service.model.manager import CapacityError
from inference_service.schemas.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    Usage,
)
from inference_service.security.auth import require_api_key

router = APIRouter(tags=["chat"])


def _engine_request(body: ChatCompletionRequest, max_new_tokens: int) -> GenerationRequest:
    return GenerationRequest(
        messages=tuple(message.model_dump() for message in body.messages),
        max_new_tokens=max_new_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        stop=tuple(body.stop),
    )


@router.post("/v1/chat/completions", dependencies=[Depends(require_api_key)], response_model=None)
async def chat_completion(
    request: Request, body: ChatCompletionRequest
) -> ChatCompletionResponse | StreamingResponse:
    settings = request.app.state.settings
    manager = request.app.state.manager
    metrics = request.app.state.metrics
    model_info = manager.info()
    if body.model is not None and body.model != model_info["model_id"]:
        raise HTTPException(status_code=404, detail="requested model is not loaded")
    max_new_tokens = body.max_tokens or settings.model_max_new_tokens
    if max_new_tokens > settings.model_max_new_tokens:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"max_tokens exceeds configured limit {settings.model_max_new_tokens}",
        )
    generation = _engine_request(body, max_new_tokens)
    if body.stream:
        if not manager.ready:
            raise HTTPException(status_code=503, detail="model unavailable")
        return StreamingResponse(
            _stream_response(request, generation, str(model_info["model_id"])),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )

    started = time.perf_counter()
    try:
        async with asyncio.timeout(settings.generation_timeout_seconds):
            result = await manager.generate(generation)
    except CapacityError as exc:
        metrics.requests.labels("chat", "saturated").inc()
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except TimeoutError as exc:
        metrics.requests.labels("chat", "timeout").inc()
        raise HTTPException(status_code=504, detail="generation timed out") from exc
    except ValueError as exc:
        metrics.requests.labels("chat", "invalid").inc()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        metrics.requests.labels("chat", "unavailable").inc()
        raise HTTPException(status_code=503, detail="model unavailable") from exc
    metrics.requests.labels("chat", "success").inc()
    metrics.request_latency.observe(time.perf_counter() - started)
    metrics.generated_tokens.inc(result.completion_tokens)
    return ChatCompletionResponse(
        model=str(model_info["model_id"]),
        choices=[
            Choice(
                message=ChoiceMessage(content=result.text),
                finish_reason=result.finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        ),
    )


async def _stream_response(
    request: Request, generation: GenerationRequest, model_id: str
) -> AsyncIterator[str]:
    manager = request.app.state.manager
    metrics = request.app.state.metrics
    settings = request.app.state.settings
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    started = time.perf_counter()
    first = True
    try:
        async with asyncio.timeout(settings.generation_timeout_seconds):
            async for chunk in manager.stream(generation):
                if await request.is_disconnected():
                    metrics.requests.labels("chat_stream", "cancelled").inc()
                    return
                if chunk.text:
                    payload = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": chunk.text}
                                if first
                                else {"content": chunk.text},
                                "finish_reason": None,
                            }
                        ],
                    }
                    first = False
                    yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
                if chunk.done:
                    metrics.generated_tokens.inc(chunk.generated_tokens)
                    if chunk.ttft_seconds is not None:
                        metrics.ttft.observe(chunk.ttft_seconds)
                    payload = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [
                            {"index": 0, "delta": {}, "finish_reason": chunk.finish_reason}
                        ],
                        "usage": {
                            "prompt_tokens": chunk.prompt_tokens,
                            "completion_tokens": chunk.generated_tokens,
                            "total_tokens": chunk.prompt_tokens + chunk.generated_tokens,
                        },
                    }
                    yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
                    yield "data: [DONE]\n\n"
        metrics.requests.labels("chat_stream", "success").inc()
        metrics.request_latency.observe(time.perf_counter() - started)
    except CapacityError:
        metrics.requests.labels("chat_stream", "saturated").inc()
        payload = {"error": {"code": "capacity_exhausted", "message": "capacity exhausted"}}
        yield f"data: {json.dumps(payload)}\n\n"
    except TimeoutError:
        metrics.requests.labels("chat_stream", "timeout").inc()
        payload = {"error": {"code": "timeout", "message": "generation timed out"}}
        yield f"data: {json.dumps(payload)}\n\n"
    except (RuntimeError, ValueError):
        metrics.requests.labels("chat_stream", "error").inc()
        payload = {"error": {"code": "generation_error", "message": "generation failed"}}
        yield f"data: {json.dumps(payload)}\n\n"
