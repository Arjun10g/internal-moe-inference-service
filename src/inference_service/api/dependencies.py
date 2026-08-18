from __future__ import annotations

from typing import cast

from fastapi import Request

from inference_service.model.manager import ModelManager


def get_manager(request: Request) -> ModelManager:
    return cast(ModelManager, request.app.state.manager)
