from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from inference_service.model.manifest import (
    MANIFEST_NAME,
    UNSAFE_SUFFIXES,
    ModelManifest,
    load_manifest,
    reject_symlinks,
)


@dataclass(frozen=True)
class ValidationSummary:
    manifest: ModelManifest
    total_bytes: int
    file_count: int


def _secure_resolve(root: Path, relative: str) -> Path:
    root_real = root.resolve(strict=True)
    target = root / relative
    if target.is_symlink():
        raise ValueError(f"symbolic links are forbidden: {relative}")
    target_real = target.resolve(strict=True)
    if os.path.commonpath((root_real, target_real)) != str(root_real):
        raise ValueError(f"artifact escapes model directory: {relative}")
    return target_real


def validate_model_directory(model_dir: Path, *, strict: bool = True) -> ValidationSummary:
    if not model_dir.exists() or not model_dir.is_dir():
        raise ValueError(f"model directory does not exist: {model_dir}")
    if model_dir.is_symlink():
        raise ValueError("model directory must not be a symbolic link")

    reject_symlinks(model_dir)
    manifest = load_manifest(model_dir)
    expected = {entry.path for entry in manifest.files}
    actual_files = [
        path
        for path in model_dir.rglob("*")
        if path.is_file() and path.name not in {MANIFEST_NAME, ".complete"}
    ]
    for path in actual_files:
        relative = path.relative_to(model_dir).as_posix()
        suffix = path.suffix.lower()
        if suffix in UNSAFE_SUFFIXES:
            raise ValueError(f"unsafe checkpoint/file type: {relative}")
        if manifest.model_format == "gguf" and suffix == ".safetensors":
            raise ValueError("GGUF model directory must not contain safetensors weights")
        if manifest.model_format == "safetensors" and suffix == ".gguf":
            raise ValueError("safetensors model directory must not contain GGUF weights")
    total = 0
    for entry in manifest.files:
        path = _secure_resolve(model_dir, entry.path)
        if not path.is_file():
            raise ValueError(f"artifact is not a regular file: {entry.path}")
        stat = path.stat()
        if stat.st_size != entry.size:
            raise ValueError(
                f"size mismatch for {entry.path}: expected {entry.size}, got {stat.st_size}"
            )
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != entry.sha256:
            raise ValueError(f"SHA-256 mismatch for {entry.path}")
        total += stat.st_size

    if strict:
        actual = {path.relative_to(model_dir).as_posix() for path in actual_files}
        unexpected = sorted(actual - expected)
        if unexpected:
            raise ValueError(f"unexpected model artifacts: {unexpected}")
    return ValidationSummary(manifest=manifest, total_bytes=total, file_count=len(expected))
