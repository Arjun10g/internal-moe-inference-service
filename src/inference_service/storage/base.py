from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class ModelStorage(ABC):
    @abstractmethod
    def resolve(self) -> Path:
        """Resolve artifacts once during startup and return a local directory."""
