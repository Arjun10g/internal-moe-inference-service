from __future__ import annotations

from pathlib import Path

from inference_service.storage.base import ModelStorage
from inference_service.storage.local import LocalModelStorage
from inference_service.storage.s3 import S3ModelStorage


def storage_for_source(source: str, cache_root: Path, *, strict: bool = True) -> ModelStorage:
    if source.startswith("s3://"):
        return S3ModelStorage(source, cache_root, strict=strict)
    return LocalModelStorage(source)


__all__ = ["LocalModelStorage", "ModelStorage", "S3ModelStorage", "storage_for_source"]
