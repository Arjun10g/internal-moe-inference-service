from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3  # type: ignore[import-untyped]
from filelock import FileLock

from inference_service.model.manifest import MANIFEST_NAME, ModelManifest
from inference_service.model.validation import validate_model_directory
from inference_service.storage.base import ModelStorage

logger = logging.getLogger(__name__)


def parse_s3_source(source: str) -> tuple[str, str]:
    parsed = urlparse(source)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError("model source is not a valid s3:// URI")
    prefix = parsed.path.strip("/")
    if not prefix or any(part in {"", ".", ".."} for part in prefix.split("/")):
        raise ValueError("S3 model prefix is empty or unsafe")
    return parsed.netloc, prefix


class S3ModelStorage(ModelStorage):
    """Download a manifest-declared checkpoint exactly once into a verified cache."""

    def __init__(
        self,
        source: str,
        cache_root: Path,
        *,
        strict: bool = True,
        client: Any | None = None,
    ) -> None:
        self.bucket, self.prefix = parse_s3_source(source)
        self.cache_root = cache_root
        self.strict = strict
        self.client = client

    def _client(self) -> Any:
        # The default provider chain obtains ECS task credentials; no static key is accepted here.
        if self.client is None:
            self.client = boto3.client("s3")
        return self.client

    def _get_bytes(self, key: str, *, limit: int | None = None) -> bytes:
        response = self._client().get_object(Bucket=self.bucket, Key=key)
        body = response["Body"]
        try:
            data = body.read((limit + 1) if limit is not None else -1)
        finally:
            body.close()
        if limit is not None and len(data) > limit:
            raise ValueError(f"S3 object exceeds allowed size: {key}")
        return bytes(data)

    def _cache_path(self, manifest: ModelManifest) -> Path:
        identity = f"{self.bucket}/{self.prefix}/{manifest.digest}"
        return self.cache_root / hashlib.sha256(identity.encode()).hexdigest()

    def resolve(self) -> Path:
        manifest_key = f"{self.prefix}/{MANIFEST_NAME}"
        raw_manifest = self._get_bytes(manifest_key, limit=4 * 1024 * 1024)
        manifest = ModelManifest.from_bytes(raw_manifest)
        target = self._cache_path(manifest)
        self.cache_root.mkdir(parents=True, exist_ok=True, mode=0o750)
        lock = FileLock(str(target) + ".lock", timeout=900)
        with lock:
            marker = target / ".complete"
            if marker.is_file():
                try:
                    if marker.read_text(encoding="ascii").strip() != manifest.digest:
                        raise ValueError("model cache completion marker does not match manifest")
                    validate_model_directory(target, strict=self.strict)
                    logger.info("model_cache_hit", extra={"cache_key": target.name})
                    return target
                except (OSError, ValueError, json.JSONDecodeError):
                    # Never reuse a partial/corrupted cache. The target is a single validated key.
                    if target.is_symlink() or not target.is_dir():
                        target.unlink(missing_ok=True)
                    else:
                        shutil.rmtree(target)
            elif target.exists():
                # A cache key without a valid completion marker is never publishable.
                if target.is_symlink() or not target.is_dir():
                    target.unlink()
                else:
                    shutil.rmtree(target)

            temp_dir = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=self.cache_root))
            try:
                (temp_dir / MANIFEST_NAME).write_bytes(raw_manifest)
                for entry in manifest.files:
                    destination = temp_dir / entry.path
                    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
                    key = f"{self.prefix}/{entry.path}"
                    response = self._client().get_object(Bucket=self.bucket, Key=key)
                    digest = hashlib.sha256()
                    written = 0
                    body = response["Body"]
                    try:
                        with destination.open("xb") as handle:
                            while chunk := body.read(1024 * 1024):
                                written += len(chunk)
                                if written > entry.size:
                                    raise ValueError(
                                        f"S3 object is larger than manifest: {entry.path}"
                                    )
                                digest.update(chunk)
                                handle.write(chunk)
                            handle.flush()
                            os.fsync(handle.fileno())
                    finally:
                        body.close()
                    if written != entry.size or digest.hexdigest() != entry.sha256:
                        raise ValueError(f"S3 object integrity check failed: {entry.path}")

                validate_model_directory(temp_dir, strict=self.strict)
                (temp_dir / ".complete").write_text(manifest.digest + "\n", encoding="ascii")
                os.replace(temp_dir, target)
                logger.info("model_cache_populated", extra={"cache_key": target.name})
                return target
            except BaseException:
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise
