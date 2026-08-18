from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response

from inference_service.security.auth import require_api_key

router = APIRouter(tags=["metrics"])


@router.get("/metrics", dependencies=[Depends(require_api_key)])
async def metrics(request: Request) -> Response:
    return Response(
        content=request.app.state.metrics.render(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
