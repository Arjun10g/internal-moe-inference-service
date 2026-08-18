from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MANIFEST_NAME = "model-manifest.json"
UNSAFE_SUFFIXES = {".bin", ".ckpt", ".pkl", ".pickle", ".pt", ".pth", ".py"}
ALLOWED_SUFFIXES = {
    ".json",
    ".gguf",
    ".jinja",
    ".model",
    ".safetensors",
    ".tiktoken",
    ".txt",
}


def validate_relative_artifact_path(raw: str) -> str:
    if not raw or "\\" in raw or "\x00" in raw:
        raise ValueError("artifact path is empty or contains forbidden characters")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe artifact path: {raw!r}")
    if path.as_posix() != raw:
        raise ValueError(f"unsafe non-canonical POSIX artifact path: {raw!r}")
    if path.suffix.lower() in UNSAFE_SUFFIXES:
        raise ValueError(f"unsafe checkpoint/file type: {path.suffix}")
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError(f"unsupported artifact file type: {path.suffix or '<none>'}")
    return path.as_posix()


def reject_symlinks(root: Path) -> None:
    """Reject file and directory symlinks without following either kind."""
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in (*directories, *files):
            path = current_path / name
            if path.is_symlink():
                relative = path.relative_to(root).as_posix()
                raise ValueError(f"symbolic links are forbidden: {relative}")


class ManifestFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    size: int = Field(ge=0)
    sha256: str

    @field_validator("path")
    @classmethod
    def secure_path(cls, value: str) -> str:
        return validate_relative_artifact_path(value)

    @field_validator("sha256")
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        lowered = value.lower()
        if len(lowered) != 64 or any(ch not in "0123456789abcdef" for ch in lowered):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
        return lowered


class ModelManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1, ge=1, le=1)
    model_id: str = Field(min_length=1, max_length=256)
    revision: str = Field(min_length=1, max_length=256)
    architecture: str | None = Field(default=None, max_length=256)
    dtype: str | None = Field(default=None, max_length=64)
    model_format: Literal["safetensors", "gguf"] = "safetensors"
    files: list[ManifestFile] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_paths_and_required_files(self) -> ModelManifest:
        paths = [entry.path for entry in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("manifest contains duplicate file paths")
        safetensors = [path for path in paths if path.endswith(".safetensors")]
        gguf = [path for path in paths if path.endswith(".gguf")]
        if self.model_format == "safetensors":
            if "config.json" not in paths:
                raise ValueError("safetensors manifest must include config.json")
            if not safetensors:
                raise ValueError("safetensors manifest must include safetensors weights")
            if not any(path.startswith("tokenizer") for path in paths):
                raise ValueError("safetensors manifest must include tokenizer artifacts")
            if gguf:
                raise ValueError("safetensors manifest must not include GGUF weights")
        elif len(gguf) != 1 or safetensors:
            raise ValueError("GGUF manifest must include exactly one GGUF and no safetensors")
        return self

    @classmethod
    def from_bytes(cls, raw: bytes) -> ModelManifest:
        if len(raw) > 4 * 1024 * 1024:
            raise ValueError("manifest exceeds 4 MiB")
        data: Any = json.loads(raw.decode("utf-8"))
        return cls.model_validate(data)

    @property
    def digest(self) -> str:
        canonical = self.model_dump_json(exclude_none=True)
        return hashlib.sha256(canonical.encode()).hexdigest()


def load_manifest(model_dir: Path) -> ModelManifest:
    path = model_dir / MANIFEST_NAME
    if path.is_symlink():
        raise ValueError("manifest must not be a symbolic link")
    return ModelManifest.from_bytes(path.read_bytes())


def build_manifest(
    model_dir: Path,
    *,
    model_id: str,
    revision: str,
    architecture: str | None = None,
    dtype: str | None = None,
    model_format: Literal["safetensors", "gguf"] | None = None,
) -> ModelManifest:
    reject_symlinks(model_dir)
    entries: list[ManifestFile] = []
    for path in sorted(model_dir.rglob("*")):
        if not path.is_file() or path.name == MANIFEST_NAME:
            continue
        if path.is_symlink():
            raise ValueError(f"symbolic links are forbidden: {path}")
        relative = path.relative_to(model_dir).as_posix()
        validate_relative_artifact_path(relative)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        entries.append(
            ManifestFile(path=relative, size=path.stat().st_size, sha256=digest.hexdigest())
        )
    formats = {
        "gguf" if entry.path.endswith(".gguf") else "safetensors"
        for entry in entries
        if entry.path.endswith((".gguf", ".safetensors"))
    }
    if model_format is None:
        if len(formats) != 1:
            raise ValueError("could not infer one model format from checkpoint files")
        model_format = "gguf" if formats.pop() == "gguf" else "safetensors"
    return ModelManifest(
        model_id=model_id,
        revision=revision,
        architecture=architecture,
        dtype=dtype,
        model_format=model_format,
        files=entries,
    )


def write_manifest(model_dir: Path, manifest: ModelManifest) -> Path:
    target = model_dir / MANIFEST_NAME
    if target.is_symlink():
        raise ValueError("refusing to write a manifest through a symbolic link")
    payload = manifest.model_dump_json(indent=2, exclude_none=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{MANIFEST_NAME}.", dir=model_dir)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o644)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
