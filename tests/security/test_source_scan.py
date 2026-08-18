from __future__ import annotations

import json
from pathlib import Path

import yaml

from inference_service.engines.llama_cpp import (
    PINNED_LLAMA_CPP_BUILD,
    PINNED_LLAMA_CPP_COMMIT,
)


def test_no_remote_code_or_hardcoded_credentials() -> None:
    root = Path(__file__).resolve().parents[2]
    source = "\n".join(path.read_text(encoding="utf-8") for path in (root / "src").rglob("*.py"))
    assert "trust_remote_code=True" not in source
    assert "aws_access_key_id=" not in source.lower()
    assert "aws_secret_access_key=" not in source.lower()
    assert "huggingface.co" not in source.lower()


def test_production_image_excludes_models() -> None:
    root = Path(__file__).resolve().parents[2]
    dockerignore = (root / ".dockerignore").read_text(encoding="utf-8")
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    assert "artifacts" in dockerignore
    assert "COPY ." not in dockerfile
    assert "USER 10001:10001" in dockerfile
    assert "HF_HUB_OFFLINE=1" in dockerfile


def test_compose_file_parses_with_hardened_defaults() -> None:
    root = Path(__file__).resolve().parents[2]
    compose = yaml.safe_load((root / "docker-compose.yml").read_text(encoding="utf-8"))
    service = compose["services"]["inference"]
    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in service["security_opt"]
    assert service["ports"] == ["127.0.0.1:8000:8000"]
    assert all("AWS_SECRET" not in item for item in service.get("environment", []))


def test_llama_cpp_code_and_supply_chain_lock_match() -> None:
    root = Path(__file__).resolve().parents[2]
    lock = json.loads((root / "vendor/llama.cpp.lock.json").read_text(encoding="utf-8"))
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    assert lock["tag"] == PINNED_LLAMA_CPP_BUILD
    assert lock["commit"] == PINNED_LLAMA_CPP_COMMIT
    assert "download_llama_runtime.py" in dockerfile
    assert "COPY --from=llama-runtime /opt/llama.cpp" in dockerfile
    assert "libgomp1" in dockerfile
