from __future__ import annotations

from fastapi import APIRouter, Request, Response, status

from inference_service.schemas.model import HealthStatus

router = APIRouter(tags=["health"])


@router.get("/health/live", response_model=HealthStatus)
async def live(request: Request) -> HealthStatus:
    return HealthStatus(status="live", model_state=request.app.state.manager.state.value)


@router.get("/health/ready", response_model=HealthStatus)
async def ready(request: Request, response: Response) -> HealthStatus:
    manager = request.app.state.manager
    if not manager.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthStatus(status="not_ready", model_state=manager.state.value)
    return HealthStatus(status="ready", model_state=manager.state.value)
