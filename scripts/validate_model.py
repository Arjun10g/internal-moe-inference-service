#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference_service.model.validation import validate_model_directory


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a local model artifact directory")
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--allow-unexpected", action="store_true")
    args = parser.parse_args()
    try:
        summary = validate_model_directory(
            args.model_dir.resolve(strict=True), strict=not args.allow_unexpected
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"valid": False, "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 1
    print(
        json.dumps(
            {
                "valid": True,
                "model_id": summary.manifest.model_id,
                "revision": summary.manifest.revision,
                "architecture": summary.manifest.architecture,
                "dtype": summary.manifest.dtype,
                "file_count": summary.file_count,
                "total_bytes": summary.total_bytes,
                "manifest_digest": summary.manifest.digest,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
