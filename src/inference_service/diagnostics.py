from __future__ import annotations

import platform
from typing import Any

from inference_service.config import Settings


def runtime_diagnostics(settings: Settings, artifact_bytes: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python_version": platform.python_version(),
        "selected_device": settings.device,
        "model_format": settings.model_format,
        "selected_dtype": (
            settings.model_dtype if settings.model_format == "safetensors" else "defined_by_gguf"
        ),
        "inference_engine": settings.inference_engine.value,
        "model_max_context": settings.model_max_context,
        "max_concurrent_requests": settings.max_concurrent_requests,
        "artifact_size_bytes": artifact_bytes,
        "gpu_detected": False,
    }
    try:
        import torch

        result["torch_version"] = torch.__version__
        result["torch_cuda_version"] = torch.version.cuda
        result["gpu_detected"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            result.update(
                {
                    "gpu_model": torch.cuda.get_device_name(0),
                    "gpu_free_vram_bytes": free,
                    "gpu_total_vram_bytes": total,
                }
            )
            if artifact_bytes is not None and artifact_bytes * 1.2 > free:
                result["memory_warning"] = (
                    "checkpoint bytes plus a minimal 20% runtime allowance exceed free VRAM"
                )
    except ImportError:
        result["torch_version"] = None
    return result
