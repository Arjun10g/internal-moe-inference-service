from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from inference_service.config import Settings


def base_values(tmp_path: Path) -> dict[str, object]:
    return {
        "model_source": str(tmp_path),
        "model_dtype": "float32",
        "device": "cpu",
        "environment": "test",
        "inference_engine": "mock",
        "allow_unauthenticated": True,
        "model_max_context": 128,
        "model_max_new_tokens": 16,
        "max_prompt_tokens": 96,
        "max_batch_tokens": 128,
    }


def test_rejects_remote_http_source(tmp_path: Path) -> None:
    values = base_values(tmp_path)
    values["model_source"] = "https://huggingface.co/unsafe/model"
    with pytest.raises(ValidationError, match="local path or an s3"):
        Settings(**values)


def test_rejects_mock_engine_outside_test(tmp_path: Path) -> None:
    values = base_values(tmp_path)
    values["environment"] = "production"
    with pytest.raises(ValidationError, match="permitted only"):
        Settings(**values)


def test_requires_authentication_by_default(tmp_path: Path) -> None:
    values = base_values(tmp_path)
    values["allow_unauthenticated"] = False
    with pytest.raises(ValidationError, match="API_KEY"):
        Settings(**values)


def test_rejects_empty_api_key(tmp_path: Path) -> None:
    values = base_values(tmp_path)
    values.update({"allow_unauthenticated": False, "api_key": ""})
    with pytest.raises(ValidationError, match="API_KEY"):
        Settings(**values)


def test_rejects_impossible_context_budget(tmp_path: Path) -> None:
    values = base_values(tmp_path)
    values["max_prompt_tokens"] = 120
    with pytest.raises(ValidationError, match="exceeds MODEL_MAX_CONTEXT"):
        Settings(**values)


def test_valid_private_s3_source(tmp_path: Path) -> None:
    values = base_values(tmp_path)
    values["model_source"] = "s3://private-models/qwen/revision-001"
    settings = Settings(**values)
    assert settings.model_source.startswith("s3://")


def test_rejects_unimplemented_transformers_tensor_parallelism(tmp_path: Path) -> None:
    values = base_values(tmp_path)
    values.update({"inference_engine": "transformers", "tensor_parallel_size": 2})
    with pytest.raises(ValidationError, match="TENSOR_PARALLEL_SIZE=1"):
        Settings(**values)


def test_rejects_zero_queue_timeout(tmp_path: Path) -> None:
    values = base_values(tmp_path)
    values["request_queue_timeout_seconds"] = 0
    with pytest.raises(ValidationError, match="greater than 0"):
        Settings(**values)


@pytest.mark.parametrize(
    ("engine", "model_format", "message"),
    [
        ("transformers", "gguf", "MODEL_FORMAT=safetensors"),
        ("llama_cpp", "safetensors", "MODEL_FORMAT=gguf"),
    ],
)
def test_rejects_engine_format_mismatch(
    tmp_path: Path, engine: str, model_format: str, message: str
) -> None:
    values = base_values(tmp_path)
    values.update({"inference_engine": engine, "model_format": model_format})
    with pytest.raises(ValidationError, match=message):
        Settings(**values)


def test_valid_llama_cpp_gguf_configuration(tmp_path: Path) -> None:
    values = base_values(tmp_path)
    values.update({"inference_engine": "llama_cpp", "model_format": "gguf"})
    settings = Settings(**values)
    assert settings.inference_engine.value == "llama_cpp"
    assert settings.model_format == "gguf"


def test_rejects_multiple_llama_cpp_inference_slots(tmp_path: Path) -> None:
    values = base_values(tmp_path)
    values.update(
        {
            "inference_engine": "llama_cpp",
            "model_format": "gguf",
            "max_concurrent_requests": 2,
        }
    )
    with pytest.raises(ValidationError, match="MAX_CONCURRENT_REQUESTS=1"):
        Settings(**values)
