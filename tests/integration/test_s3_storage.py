from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from botocore.exceptions import ClientError

from inference_service.model.manifest import MANIFEST_NAME
from inference_service.storage.s3 import S3ModelStorage, parse_s3_source


class FakeS3:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.calls: list[str] = []
        self.denied: set[str] = set()

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        assert Bucket == "private-bucket"
        self.calls.append(Key)
        if Key in self.denied:
            raise ClientError({"Error": {"Code": "AccessDenied", "Message": "denied"}}, "GetObject")
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key])}


def s3_objects(model_dir: Path) -> dict[str, bytes]:
    return {
        f"models/rev1/{path.relative_to(model_dir).as_posix()}": path.read_bytes()
        for path in model_dir.rglob("*")
        if path.is_file()
    }


def test_parse_s3_source_rejects_unsafe_prefix() -> None:
    with pytest.raises(ValueError, match="unsafe"):
        parse_s3_source("s3://private-bucket/models/../secret")


def test_download_and_cache_reuse(mock_model_dir: Path, tmp_path: Path) -> None:
    fake = FakeS3(s3_objects(mock_model_dir))
    storage = S3ModelStorage("s3://private-bucket/models/rev1", tmp_path / "cache", client=fake)
    first = storage.resolve()
    assert (first / ".complete").is_file()
    assert (first / "model.safetensors").read_bytes() == b"safe-test-placeholder"
    second = storage.resolve()
    assert second == first
    # Manifest is checked at startup; weight artifacts are not downloaded again.
    assert fake.calls.count("models/rev1/model.safetensors") == 1


def test_invalid_completion_marker_rebuilds_cache(mock_model_dir: Path, tmp_path: Path) -> None:
    fake = FakeS3(s3_objects(mock_model_dir))
    storage = S3ModelStorage("s3://private-bucket/models/rev1", tmp_path / "cache", client=fake)
    first = storage.resolve()
    (first / ".complete").write_text("wrong-digest\n", encoding="ascii")

    rebuilt = storage.resolve()

    assert rebuilt == first
    assert (rebuilt / ".complete").read_text(encoding="ascii").strip() != "wrong-digest"
    assert fake.calls.count("models/rev1/model.safetensors") == 2


def test_missing_object_fails_without_complete_cache(mock_model_dir: Path, tmp_path: Path) -> None:
    objects = s3_objects(mock_model_dir)
    del objects["models/rev1/model.safetensors"]
    fake = FakeS3(objects)
    storage = S3ModelStorage("s3://private-bucket/models/rev1", tmp_path / "cache", client=fake)
    with pytest.raises(ClientError):
        storage.resolve()
    assert not list((tmp_path / "cache").glob("*/.complete"))


def test_access_denied_fails_closed(mock_model_dir: Path, tmp_path: Path) -> None:
    fake = FakeS3(s3_objects(mock_model_dir))
    fake.denied.add("models/rev1/model.safetensors")
    storage = S3ModelStorage("s3://private-bucket/models/rev1", tmp_path / "cache", client=fake)
    with pytest.raises(ClientError, match="AccessDenied"):
        storage.resolve()


def test_truncated_or_hash_mismatched_object_fails(mock_model_dir: Path, tmp_path: Path) -> None:
    objects = s3_objects(mock_model_dir)
    objects["models/rev1/model.safetensors"] = b"short"
    fake = FakeS3(objects)
    storage = S3ModelStorage("s3://private-bucket/models/rev1", tmp_path / "cache", client=fake)
    with pytest.raises(ValueError, match="integrity"):
        storage.resolve()


def test_malicious_manifest_is_rejected_before_artifacts(tmp_path: Path) -> None:
    raw = (
        b'{"schema_version":1,"model_id":"x","revision":"1","files":['
        b'{"path":"../escape.json","size":1,"sha256":"' + b"0" * 64 + b'"}]}'
    )
    fake = FakeS3({f"models/rev1/{MANIFEST_NAME}": raw})
    storage = S3ModelStorage("s3://private-bucket/models/rev1", tmp_path / "cache", client=fake)
    with pytest.raises(ValueError, match="unsafe"):
        storage.resolve()
    assert fake.calls == [f"models/rev1/{MANIFEST_NAME}"]
