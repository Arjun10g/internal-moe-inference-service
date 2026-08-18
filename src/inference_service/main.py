from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from inference_service.api.middleware import RequestContextMiddleware, RequestSizeLimitMiddleware
from inference_service.api.routes_chat import router as chat_router
from inference_service.api.routes_health import router as health_router
from inference_service.api.routes_metrics import router as metrics_router
from inference_service.api.routes_models import router as models_router
from inference_service.config import Settings
from inference_service.model.manager import EngineFactory, ModelManager, default_engine_factory
from inference_service.observability.logging import configure_logging
from inference_service.observability.metrics import Metrics

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    engine_factory: EngineFactory = default_engine_factory,
) -> FastAPI:
    resolved_settings = settings or Settings()
    configure_logging(resolved_settings.log_level)
    metrics = Metrics()
    manager = ModelManager(resolved_settings, metrics, engine_factory=engine_factory)

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        load_task = asyncio.create_task(manager.initialize(), name="model-initialization")

        def consume_failure(task: asyncio.Task[None]) -> None:
            if task.cancelled():
                return
            # Retrieving the exception prevents an unhandled-task warning. ModelManager logged it.
            _ = task.exception()

        load_task.add_done_callback(consume_failure)
        app.state.load_task = load_task
        yield
        if not load_task.done():
            try:
                await asyncio.wait_for(
                    asyncio.shield(load_task), timeout=resolved_settings.shutdown_grace_seconds
                )
            except TimeoutError:
                load_task.cancel()
        await manager.shutdown()

    app = FastAPI(
        title="Internal MoE Inference Service",
        version="0.1.0",
        docs_url=None if resolved_settings.environment == "production" else "/docs",
        redoc_url=None,
        openapi_url=None if resolved_settings.environment == "production" else "/openapi.json",
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.state.metrics = metrics
    app.state.manager = manager
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=resolved_settings.max_request_bytes)
    app.include_router(health_router)
    app.include_router(models_router)
    app.include_router(metrics_router)
    app.include_router(chat_router)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {"type": item.get("type"), "loc": item.get("loc"), "msg": item.get("msg")}
            for item in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "request validation failed",
                    "request_id": getattr(request.state, "request_id", None),
                    "details": details,
                }
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "unhandled_request_error",
            extra={"request_id": getattr(request.state, "request_id", None)},
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "internal server error",
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )

    return app


def run() -> None:
    settings = Settings()
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_config=None,
        access_log=False,
        timeout_graceful_shutdown=int(settings.shutdown_grace_seconds),
    )


if __name__ == "__main__":
    run()
