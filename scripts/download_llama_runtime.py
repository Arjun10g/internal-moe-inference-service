#!/usr/bin/env python3
"""Fetch and verify the pinned llama.cpp server runtime for this platform."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess  # nosec B404
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

CHUNK_BYTES = 1024 * 1024
RELEASE_URL = "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{name}"


def platform_key() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin" and machine in {"arm64", "aarch64"}:
        return "darwin-arm64"
    if system == "linux" and machine in {"x86_64", "amd64"}:
        return "linux-x64"
    if system == "linux" and machine in {"arm64", "aarch64"}:
        return "linux-arm64"
    if system == "windows" and machine in {"x86_64", "amd64"}:
        return "win32-x64"
    raise ValueError(f"no pinned llama.cpp runtime for {system}/{machine}")


def executable_name(key: str) -> str:
    return "llama-server.exe" if key.startswith("win32") else "llama-server"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path, *, expected_size: int) -> None:
    request = urllib.request.Request(  # noqa: S310 - fixed HTTPS GitHub release origin
        url, headers={"User-Agent": "model-inference-runtime-fetch/1"}
    )
    try:
        with (
            urllib.request.urlopen(  # noqa: S310  # nosec B310
                request, timeout=60
            ) as response,
            target.open("wb") as output,
        ):
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) != expected_size:
                raise ValueError("runtime response size does not match the lock")
            written = 0
            while chunk := response.read(CHUNK_BYTES):
                written += len(chunk)
                if written > expected_size:
                    raise ValueError("runtime download exceeded the locked size")
                output.write(chunk)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"llama.cpp runtime download failed: {exc.reason}") from exc


def _is_runtime_file(name: str, key: str) -> bool:
    lowered = name.lower()
    if name == executable_name(key) or lowered in {"license", "license.txt"}:
        return True
    is_library = lowered.endswith((".dll", ".dylib")) or ".so" in lowered
    return is_library and not ("-impl" in lowered and "server-impl" not in lowered)


def extract_runtime(archive: Path, destination: Path, key: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    links: dict[str, str] = {}
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                base = PurePosixPath(member.filename).name
                if member.is_dir() or not base or not _is_runtime_file(base, key):
                    continue
                with bundle.open(member) as source, (destination / base).open("wb") as output:
                    shutil.copyfileobj(source, output, CHUNK_BYTES)
    else:
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle.getmembers():
                base = PurePosixPath(member.name).name
                if not base or not _is_runtime_file(base, key):
                    continue
                if member.issym() or member.islnk():
                    links[base] = PurePosixPath(member.linkname).name
                    continue
                if not member.isfile():
                    continue
                source = bundle.extractfile(member)
                if source is None:
                    continue
                with source, (destination / base).open("wb") as output:
                    shutil.copyfileobj(source, output, CHUNK_BYTES)
        for link, target in links.items():
            resolved = target
            seen = {link}
            while resolved in links and resolved not in seen:
                seen.add(resolved)
                resolved = links[resolved]
            source = destination / resolved
            if source.is_file() and not (destination / link).exists():
                shutil.copy2(source, destination / link)

    server = destination / executable_name(key)
    if not server.is_file():
        raise ValueError("verified archive did not contain llama-server")
    if os.name != "nt":
        server.chmod(server.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def runtime_environment(destination: Path, key: str) -> dict[str, str]:
    environment = os.environ.copy()
    if key.startswith("linux"):
        environment["LD_LIBRARY_PATH"] = str(destination)
    elif key.startswith("darwin"):
        environment["DYLD_LIBRARY_PATH"] = str(destination)
    return environment


def verify_runtime(destination: Path, key: str, expected_build: str) -> str:
    server = destination / executable_name(key)
    result = subprocess.run(  # noqa: S603  # nosec B603
        [str(server), "--version"],
        env=runtime_environment(destination, key),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0 or not output:
        raise RuntimeError("downloaded llama-server failed its version check")
    if expected_build.removeprefix("b") not in output:
        raise RuntimeError(f"downloaded runtime is not {expected_build}: {output.splitlines()[0]}")
    return output.splitlines()[0]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", help="runtime key; defaults to the current platform")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--lock", type=Path, default=root / "vendor/llama.cpp.lock.json")
    args = parser.parse_args()

    try:
        lock = json.loads(args.lock.read_text(encoding="utf-8"))
        if lock.get("schemaVersion") != 1:
            raise ValueError("unsupported llama.cpp lock schema")
        key = args.key or platform_key()
        entry = lock.get("assets", {}).get(key)
        if not isinstance(entry, dict):
            raise ValueError(f"runtime key is not pinned: {key}")
        name = entry.get("name")
        expected_size = entry.get("sizeBytes")
        expected_sha256 = entry.get("sha256")
        if not isinstance(name, str) or PurePosixPath(name).name != name:
            raise ValueError("runtime lock contains an unsafe archive name")
        if not isinstance(expected_size, int) or expected_size <= 0:
            raise ValueError("runtime lock contains an invalid archive size")
        if not isinstance(expected_sha256, str) or (
            len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise ValueError("runtime lock contains an invalid SHA-256")
        requested_destination = (args.destination or root / "runtime" / key).expanduser().absolute()
        if requested_destination.is_symlink():
            raise ValueError("runtime destination must not be a symbolic link")
        destination = requested_destination.parent.resolve() / requested_destination.name
        if destination.parent == destination:
            raise ValueError("runtime destination cannot be a filesystem root")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not destination.is_dir():
            raise ValueError("runtime destination must be a directory, not a file or symlink")
        if (
            destination.is_dir()
            and any(destination.iterdir())
            and not (destination / executable_name(key)).is_file()
        ):
            raise ValueError("refusing to replace a non-runtime directory")
        with tempfile.TemporaryDirectory(
            prefix=".llama-runtime-", dir=destination.parent
        ) as temporary:
            temporary_path = Path(temporary)
            archive = temporary_path / name
            staging = temporary_path / "staging"
            url = RELEASE_URL.format(tag=lock["tag"], name=name)
            print(f"downloading {url}", file=sys.stderr)
            download(url, archive, expected_size=expected_size)
            actual_size = archive.stat().st_size
            if actual_size != expected_size:
                raise ValueError(
                    f"runtime size mismatch: expected {expected_size}, got {actual_size}"
                )
            actual_sha256 = sha256_file(archive)
            if actual_sha256 != expected_sha256:
                raise ValueError("runtime SHA-256 mismatch")
            extract_runtime(archive, staging, key)
            license_file = staging / "LLAMA_CPP_LICENSE.txt"
            if not license_file.exists():
                shutil.copy2(root / "vendor/LLAMA_CPP_LICENSE.txt", license_file)
            version = verify_runtime(staging, key, lock["tag"])
            previous = temporary_path / "previous"
            if destination.exists():
                os.replace(destination, previous)
            try:
                os.replace(staging, destination)
            except BaseException:
                if previous.exists():
                    os.replace(previous, destination)
                raise
        print(
            json.dumps(
                {
                    "runtime": str(destination),
                    "executable": str(destination / executable_name(key)),
                    "key": key,
                    "tag": lock["tag"],
                    "commit": lock["commit"],
                    "version": version,
                },
                indent=2,
            )
        )
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
