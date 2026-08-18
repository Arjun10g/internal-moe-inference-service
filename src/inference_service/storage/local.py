from __future__ import annotations

from pathlib import Path

from inference_service.storage.base import ModelStorage


class LocalModelStorage(ModelStorage):
    def __init__(self, source: str) -> None:
        self._source = Path(source)

    def resolve(self) -> Path:
        if not self._source.exists() or not self._source.is_dir():
            raise ValueError(f"local MODEL_SOURCE is not a directory: {self._source}")
        if self._source.is_symlink():
            raise ValueError("local MODEL_SOURCE must not be a symbolic link")
        return self._source.resolve(strict=True)
