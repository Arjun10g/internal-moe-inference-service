#!/usr/bin/env python3
"""Download the approved companion Qwen GGUF from its Google Cloud mirror."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from inference_service.model.manifest import ManifestFile, ModelManifest, write_manifest


@dataclass(frozen=True)
class Artifact:
    profile_id: str
    base_model: str
    quantization: str
    file_name: str
    url: str
    expected_bytes: int
    sha256: str


QWEN_ARTIFACT = Artifact(
    profile_id="qwen3-coder-30b-a3b-q4xl",
    base_model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
    quantization="UD-Q4_K_XL (GGUF)",
    file_name="Qwen3-Coder-30B-A3B-Instruct-1M-UD-Q4_K_XL.gguf",
    url=(
        "https://storage.googleapis.com/restricted-local-coder-dazzling-howl-491904/"
        "Qwen3-Coder-30B-A3B-Instruct-1M-UD-Q4_K_XL.gguf"
    ),
    expected_bytes=17_690_500_448,
    sha256="e71c9271166ad64865767022e86f45ea4f03a8258389460cc55c8d95e18833db",
)

CHUNK_BYTES = 8 * 1024 * 1024
PROGRESS_BYTES = 512 * 1024 * 1024


class Response(Protocol):
    status: int
    headers: Mapping[str, str]

    def read(self, size: int = -1) -> bytes: ...

    def __enter__(self) -> Response: ...

    def __exit__(self, *args: object) -> None: ...


class Opener(Protocol):
    def __call__(self, request: Request, *, timeout: float) -> Response: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, artifact: Artifact = QWEN_ARTIFACT) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"artifact is not a regular file: {path}")
    actual_bytes = path.stat().st_size
    if actual_bytes != artifact.expected_bytes:
        raise ValueError(
            f"size mismatch for {path}: expected {artifact.expected_bytes}, got {actual_bytes}"
        )
    actual_sha256 = _sha256(path)
    if actual_sha256 != artifact.sha256:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {artifact.sha256}, got {actual_sha256}"
        )


def write_runtime_manifest(path: Path, artifact: Artifact = QWEN_ARTIFACT) -> Path:
    manifest = ModelManifest(
        model_id=artifact.profile_id,
        revision=f"sha256-{artifact.sha256[:16]}",
        architecture="qwen3moe",
        dtype=artifact.quantization,
        model_format="gguf",
        files=[
            ManifestFile(
                path=path.name,
                size=artifact.expected_bytes,
                sha256=artifact.sha256,
            )
        ],
    )
    return write_manifest(path.parent, manifest)


def _content_range_start(response: Response) -> int | None:
    raw = response.headers.get("Content-Range")
    if not raw or not str(raw).startswith("bytes "):
        return None
    first = str(raw).removeprefix("bytes ").split("-", 1)[0]
    try:
        return int(first)
    except ValueError:
        return None


def _stream_response(
    response: Response,
    partial: Path,
    *,
    offset: int,
    artifact: Artifact,
) -> None:
    status = int(response.status)
    if offset and status == 206:
        if _content_range_start(response) != offset:
            raise RuntimeError("mirror returned an invalid Content-Range for the resumed download")
        mode = "ab"
        written = offset
    elif status == 200:
        mode = "wb"
        written = 0
    else:
        raise RuntimeError(f"mirror returned unexpected HTTP status {status}")

    next_progress = ((written // PROGRESS_BYTES) + 1) * PROGRESS_BYTES
    with partial.open(mode) as handle:
        output: BinaryIO = handle
        while chunk := response.read(CHUNK_BYTES):
            written += len(chunk)
            if written > artifact.expected_bytes:
                raise ValueError("download exceeded the approved artifact size")
            output.write(chunk)
            if written >= next_progress:
                print(
                    f"downloaded {written / (1024**3):.2f} / "
                    f"{artifact.expected_bytes / (1024**3):.2f} GiB",
                    file=sys.stderr,
                )
                next_progress += PROGRESS_BYTES
        output.flush()
        os.fsync(output.fileno())


def download(
    destination: Path,
    *,
    artifact: Artifact = QWEN_ARTIFACT,
    opener: Opener = urlopen,  # type: ignore[assignment]
    force: bool = False,
) -> Path:
    destination = destination.expanduser()
    partial = destination.with_name(destination.name + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_symlink() or partial.is_symlink():
        raise ValueError("refusing to write through a symbolic link")
    if force:
        destination.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
    if destination.exists():
        verify_file(destination, artifact)
        write_runtime_manifest(destination, artifact)
        return destination

    offset = partial.stat().st_size if partial.exists() else 0
    if offset > artifact.expected_bytes:
        raise ValueError(
            "partial download is larger than the approved artifact; rerun with --force"
        )
    parsed_url = urlparse(artifact.url)
    if parsed_url.scheme != "https" or parsed_url.hostname != "storage.googleapis.com":
        raise ValueError("artifact URL must use the approved Google Cloud Storage HTTPS origin")
    request = Request(  # noqa: S310 - scheme and exact host are validated above
        artifact.url, headers={"User-Agent": "model-inference-artifact-fetch/1"}
    )
    if offset:
        request.add_header("Range", f"bytes={offset}-")

    try:
        with opener(request, timeout=60.0) as response:  # nosec B310
            _stream_response(response, partial, offset=offset, artifact=artifact)
    except HTTPError as exc:
        if not (exc.code == 416 and offset == artifact.expected_bytes):
            raise RuntimeError(f"model download failed with HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"model download failed: {exc.reason}") from exc

    verify_file(partial, artifact)
    os.replace(partial, destination)
    write_runtime_manifest(destination, artifact)
    return destination


def check_mirror(
    *,
    artifact: Artifact = QWEN_ARTIFACT,
    opener: Opener = urlopen,  # type: ignore[assignment]
) -> dict[str, object]:
    parsed_url = urlparse(artifact.url)
    if parsed_url.scheme != "https" or parsed_url.hostname != "storage.googleapis.com":
        raise ValueError("artifact URL must use the approved Google Cloud Storage HTTPS origin")
    request = Request(  # noqa: S310 - scheme and exact host are validated above
        artifact.url,
        method="HEAD",
        headers={"User-Agent": "model-inference-artifact-fetch/1"},
    )
    try:
        with opener(request, timeout=30.0) as response:  # nosec B310
            status = int(response.status)
            raw_length = response.headers.get("Content-Length")
    except HTTPError as exc:
        raise RuntimeError(f"model mirror check failed with HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"model mirror check failed: {exc.reason}") from exc
    if status != 200:
        raise RuntimeError(f"model mirror returned unexpected HTTP status {status}")
    if raw_length is None or int(raw_length) != artifact.expected_bytes:
        raise ValueError(
            f"mirror size mismatch: expected {artifact.expected_bytes}, "
            f"got {raw_length or 'missing'}"
        )
    return {"reachable": True, "content_length": int(raw_length), **asdict(artifact)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch the approved Qwen3-Coder GGUF from Google Cloud Storage"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models") / QWEN_ARTIFACT.profile_id / QWEN_ARTIFACT.file_name,
        help=(
            "destination file "
            f"(default: models/{QWEN_ARTIFACT.profile_id}/{QWEN_ARTIFACT.file_name})"
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--print-url", action="store_true", help="print the approved GCP URL")
    mode.add_argument("--check", action="store_true", help="verify mirror reachability and size")
    parser.add_argument("--force", action="store_true", help="discard local copies and restart")
    args = parser.parse_args()

    try:
        if args.print_url:
            print(QWEN_ARTIFACT.url)
            return 0
        if args.check:
            print(json.dumps(check_mirror(), indent=2))
            return 0
        path = download(args.output, force=args.force)
        print(
            json.dumps(
                {
                    "downloaded": str(path.resolve()),
                    "bytes": QWEN_ARTIFACT.expected_bytes,
                    "sha256": QWEN_ARTIFACT.sha256,
                    "profile_id": QWEN_ARTIFACT.profile_id,
                },
                indent=2,
            )
        )
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
