from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from inference_service.model.manifest import (
    MANIFEST_NAME,
    ManifestFile,
    ModelManifest,
    build_manifest,
    load_manifest,
    write_manifest,
)
from inference_service.model.validation import validate_model_directory


def entry(path: str, data: bytes = b"x") -> ManifestFile:
    return ManifestFile(path=path, size=len(data), sha256=hashlib.sha256(data).hexdigest())


@pytest.mark.parametrize(
    "path",
    [
        "../escape.json",
        "/absolute.json",
        "nested/../../escape.json",
        "bad\\path.json",
        "nested//file.json",
        "nested/./file.json",
    ],
)
def test_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(ValidationError, match="unsafe|forbidden"):
        ManifestFile(path=path, size=1, sha256="0" * 64)


@pytest.mark.parametrize("path", ["model.bin", "model.pkl", "configuration.py", "weights.pt"])
def test_rejects_unsafe_file_types(path: str) -> None:
    with pytest.raises(ValidationError, match="unsafe"):
        ManifestFile(path=path, size=1, sha256="0" * 64)


def test_rejects_duplicate_entries() -> None:
    files = [
        entry("config.json"),
        entry("tokenizer.json"),
        entry("model.safetensors"),
        entry("model.safetensors"),
    ]
    with pytest.raises(ValidationError, match="duplicate"):
        ModelManifest(model_id="x", revision="1", files=files)


def test_hash_mismatch_is_detected(mock_model_dir: Path) -> None:
    (mock_model_dir / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="mismatch"):
        validate_model_directory(mock_model_dir)


def test_unexpected_file_is_rejected(mock_model_dir: Path) -> None:
    (mock_model_dir / "extra.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected"):
        validate_model_directory(mock_model_dir)


def test_symlink_escape_is_rejected(mock_model_dir: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    link = mock_model_dir / "linked.json"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")
    # build_manifest must stop before a malicious symlink can enter a manifest.
    with pytest.raises(ValueError, match="symbolic links"):
        build_manifest(
            mock_model_dir,
            model_id="x",
            revision="1",
            architecture="x",
            dtype="float32",
        )


def test_symlink_directory_is_rejected(mock_model_dir: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    link = mock_model_dir / "linked-directory"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="symbolic links"):
        validate_model_directory(mock_model_dir)


def test_valid_directory_summary(mock_model_dir: Path) -> None:
    summary = validate_model_directory(mock_model_dir)
    assert summary.file_count == 3
    assert summary.total_bytes > 0


def test_builds_and_validates_gguf_manifest(tmp_path: Path) -> None:
    model_dir = tmp_path / "gguf-model"
    model_dir.mkdir()
    (model_dir / "model.gguf").write_bytes(b"GGUF-test")
    manifest = build_manifest(
        model_dir,
        model_id="qwen-test",
        revision="sha256-test",
        architecture="qwen3moe",
        dtype="UD-Q4_K_XL",
        model_format="gguf",
    )
    from inference_service.model.manifest import write_manifest

    write_manifest(model_dir, manifest)
    summary = validate_model_directory(model_dir)
    assert summary.manifest.model_format == "gguf"
    assert summary.file_count == 1


def test_rejects_mixed_checkpoint_formats() -> None:
    files = [entry("model.safetensors"), entry("model.gguf")]
    with pytest.raises(ValidationError, match="GGUF manifest"):
        ModelManifest(model_id="x", revision="1", model_format="gguf", files=files)


@pytest.mark.parametrize("extra_name", ["unlisted.safetensors", "unlisted.pkl"])
def test_non_strict_gguf_still_rejects_unsafe_or_mixed_weights(
    tmp_path: Path, extra_name: str
) -> None:
    model_dir = tmp_path / "gguf-model"
    model_dir.mkdir()
    (model_dir / "model.gguf").write_bytes(b"GGUF-test")
    manifest = build_manifest(
        model_dir,
        model_id="qwen-test",
        revision="test",
        model_format="gguf",
    )
    from inference_service.model.manifest import write_manifest

    write_manifest(model_dir, manifest)
    (model_dir / extra_name).write_bytes(b"unsafe")
    with pytest.raises(ValueError, match="safetensors|unsafe"):
        validate_model_directory(model_dir, strict=False)


def test_manifest_writer_refuses_symlink_target(mock_model_dir: Path, tmp_path: Path) -> None:
    manifest = load_manifest(mock_model_dir)
    outside = tmp_path / "outside.json"
    outside.write_text("preserve me", encoding="utf-8")
    target = mock_model_dir / MANIFEST_NAME
    target.unlink()
    try:
        target.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="symbolic link"):
        write_manifest(mock_model_dir, manifest)
    assert outside.read_text(encoding="utf-8") == "preserve me"
