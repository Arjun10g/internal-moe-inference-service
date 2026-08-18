#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference_service.model.manifest import build_manifest, write_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a SHA-256 model artifact manifest")
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--revision", default="unversioned")
    parser.add_argument("--dtype")
    parser.add_argument("--format", choices=("safetensors", "gguf"))
    args = parser.parse_args()
    model_dir = args.model_dir.resolve(strict=True)
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    architecture = (config.get("architectures") or [config.get("model_type")])[0]
    manifest = build_manifest(
        model_dir,
        model_id=args.model_id or model_dir.name,
        revision=args.revision,
        architecture=architecture,
        dtype=args.dtype or config.get("torch_dtype"),
        model_format=args.format,
    )
    path = write_manifest(model_dir, manifest)
    print(
        json.dumps({"manifest": str(path), "digest": manifest.digest, "files": len(manifest.files)})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
