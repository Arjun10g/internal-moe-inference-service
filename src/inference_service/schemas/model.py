from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ModelInfo(BaseModel):
    model_id: str
    revision: str
    architecture: str | None
    engine: str
    device: str
    dtype: str
    state: str
    loaded_at: float | None
    model_path: str | None = None
    metadata: dict[str, Any]


class HealthStatus(BaseModel):
    status: str
    model_state: str
