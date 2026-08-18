from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from urllib.request import Request

import pytest

from inference_service.model.manifest import load_manifest
from scripts.download_qwen_gcp import Artifact, download, verify_file


class FakeResponse(BytesIO):
    def __init__(self, data: bytes, *, status: int, headers: dict[str, str]) -> None:
        super().__init__(data)
        self.status = status
        self.headers = headers

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def tiny_artifact(data: bytes) -> Artifact:
    return Artifact(
        profile_id="test",
        base_model="test/model",
        quantization="test",
        file_name="model.gguf",
        url="https://storage.googleapis.com/example/model.gguf",
        expected_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def test_downloads_and_verifies_approved_artifact(tmp_path: Path) -> None:
    data = b"approved model bytes"
    artifact = tiny_artifact(data)

    def opener(request: Request, *, timeout: float) -> FakeResponse:
        assert request.full_url == artifact.url
        assert timeout == 60.0
        return FakeResponse(data, status=200, headers={"Content-Length": str(len(data))})

    destination = tmp_path / artifact.file_name
    assert download(destination, artifact=artifact, opener=opener) == destination
    assert destination.read_bytes() == data
    assert not destination.with_name(destination.name + ".part").exists()
    manifest = load_manifest(tmp_path)
    assert manifest.model_format == "gguf"
    assert manifest.files[0].path == artifact.file_name


def test_resumes_partial_download(tmp_path: Path) -> None:
    data = b"0123456789"
    artifact = tiny_artifact(data)
    destination = tmp_path / artifact.file_name
    partial = destination.with_name(destination.name + ".part")
    partial.write_bytes(data[:4])

    def opener(request: Request, *, timeout: float) -> FakeResponse:
        assert request.get_header("Range") == "bytes=4-"
        return FakeResponse(
            data[4:],
            status=206,
            headers={"Content-Range": f"bytes 4-{len(data) - 1}/{len(data)}"},
        )

    download(destination, artifact=artifact, opener=opener)
    assert destination.read_bytes() == data


def test_rejects_wrong_digest(tmp_path: Path) -> None:
    artifact = tiny_artifact(b"approved")
    path = tmp_path / artifact.file_name
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_file(path, artifact)


def test_refuses_symlink_destination(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"target")
    link = tmp_path / "model.gguf"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable")

    def unreachable_opener(request: Request, *, timeout: float) -> FakeResponse:
        raise AssertionError("a symlink destination must be rejected before opening the network")

    with pytest.raises(ValueError, match="symbolic link"):
        download(link, artifact=tiny_artifact(b"approved"), opener=unreachable_opener)
