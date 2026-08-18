from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EngineName(StrEnum):
    TRANSFORMERS = "transformers"
    LLAMA_CPP = "llama_cpp"
    MOCK = "mock"


class Settings(BaseSettings):
    """Validated configuration loaded once at process startup."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    model_source: str = "/models/model"
    model_id: str = "internal-model"
    model_revision: str = "unversioned"
    model_format: Literal["safetensors", "gguf"] = "safetensors"
    model_dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    model_max_context: int = Field(default=8192, ge=32, le=1_048_576)
    model_max_new_tokens: int = Field(default=1024, ge=1, le=65_536)
    inference_engine: EngineName = EngineName.TRANSFORMERS
    device: Literal["auto", "cpu", "cuda"] = "cuda"
    tensor_parallel_size: int = Field(default=1, ge=1, le=64)
    gpu_memory_utilization: float = Field(default=0.90, gt=0.0, le=0.98)
    max_concurrent_requests: int = Field(default=1, ge=1, le=1024)
    request_queue_timeout_seconds: float = Field(default=0.25, gt=0.0, le=60.0)
    max_batch_tokens: int = Field(default=32768, ge=32)
    max_prompt_tokens: int = Field(default=7168, ge=1)
    max_request_bytes: int = Field(default=1_048_576, ge=1024, le=16_777_216)
    generation_timeout_seconds: float = Field(default=300.0, gt=0.0, le=3600.0)
    model_cache_dir: Path = Path("/var/cache/inference-models")
    strict_manifest: bool = True
    api_key: SecretStr | None = None
    allow_unauthenticated: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_prompts: bool = False
    host: str = "0.0.0.0"  # noqa: S104  # nosec B104
    port: int = Field(default=8000, ge=1, le=65535)
    shutdown_grace_seconds: float = Field(default=30.0, ge=1.0, le=300.0)
    environment: Literal["development", "test", "production"] = "production"
    warmup_prompt: str = "Hello"
    warmup_max_new_tokens: int = Field(default=1, ge=1, le=8)
    llama_server_path: Path = Path("/opt/llama.cpp/llama-server")
    llama_cpp_require_pinned_runtime: bool = True
    llama_cpp_startup_timeout_seconds: float = Field(default=1800.0, ge=30.0, le=3600.0)
    llama_cpp_threads: int = Field(default=0, ge=0, le=256)
    llama_cpp_batch_size: int = Field(default=512, ge=32, le=4096)
    llama_cpp_ubatch_size: int = Field(default=128, ge=32, le=4096)
    llama_cpp_gpu_layers: int = Field(default=-1, ge=-1, le=999)
    llama_cpp_kv_cache_type: Literal["f16", "q8_0"] = "f16"
    llama_cpp_repack: bool = False

    @model_validator(mode="after")
    def validate_security_and_capacity(self) -> Settings:
        source = self.model_source.strip()
        if not source:
            raise ValueError("MODEL_SOURCE must not be empty")
        parsed = urlparse(source)
        if parsed.scheme and parsed.scheme != "s3":
            raise ValueError("MODEL_SOURCE must be a local path or an s3:// URI")
        if parsed.scheme == "s3" and (not parsed.netloc or not parsed.path.strip("/")):
            raise ValueError("S3 MODEL_SOURCE requires a bucket and non-empty prefix")
        if self.inference_engine == EngineName.MOCK and self.environment != "test":
            raise ValueError("MockInferenceEngine is permitted only when ENVIRONMENT=test")
        if self.inference_engine == EngineName.TRANSFORMERS and self.model_format != "safetensors":
            raise ValueError("TransformersInferenceEngine requires MODEL_FORMAT=safetensors")
        if self.inference_engine == EngineName.LLAMA_CPP and self.model_format != "gguf":
            raise ValueError("LlamaCppInferenceEngine requires MODEL_FORMAT=gguf")
        if self.inference_engine == EngineName.LLAMA_CPP and self.max_concurrent_requests != 1:
            raise ValueError(
                "LlamaCppInferenceEngine requires MAX_CONCURRENT_REQUESTS=1 "
                "because its supervised server uses one inference slot"
            )
        if self.inference_engine == EngineName.TRANSFORMERS and self.tensor_parallel_size != 1:
            raise ValueError(
                "TransformersInferenceEngine currently requires TENSOR_PARALLEL_SIZE=1"
            )
        if not self.allow_unauthenticated and (
            self.api_key is None or not self.api_key.get_secret_value()
        ):
            raise ValueError("API_KEY is required unless ALLOW_UNAUTHENTICATED=true")
        if self.max_prompt_tokens + self.model_max_new_tokens > self.model_max_context:
            raise ValueError("MAX_PROMPT_TOKENS + MODEL_MAX_NEW_TOKENS exceeds MODEL_MAX_CONTEXT")
        if self.max_batch_tokens < self.model_max_context:
            raise ValueError("MAX_BATCH_TOKENS must be at least MODEL_MAX_CONTEXT")
        if self.llama_cpp_ubatch_size > self.llama_cpp_batch_size:
            raise ValueError("LLAMA_CPP_UBATCH_SIZE must not exceed LLAMA_CPP_BATCH_SIZE")
        return self

    @property
    def api_key_value(self) -> str | None:
        return self.api_key.get_secret_value() if self.api_key else None
