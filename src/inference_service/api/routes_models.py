from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from inference_service.schemas.model import ModelInfo
from inference_service.security.auth import require_api_key

router = APIRouter(tags=["models"])


@router.get("/v1/models/current", response_model=ModelInfo, dependencies=[Depends(require_api_key)])
async def current_model(request: Request) -> ModelInfo:
    return ModelInfo.model_validate(request.app.state.manager.info())


@router.get("/v1/models", dependencies=[Depends(require_api_key)])
async def list_models(request: Request) -> dict[str, object]:
    info = request.app.state.manager.info()
    return {
        "object": "list",
        "data": [
            {
                "id": info["model_id"],
                "object": "model",
                "created": int(info["loaded_at"] or 0),
                "owned_by": "internal",
            }
        ],
    }
