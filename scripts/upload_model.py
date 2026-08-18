#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import boto3

from inference_service.model.manifest import MANIFEST_NAME
from inference_service.model.validation import validate_model_directory
from inference_service.storage.s3 import parse_s3_source


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload a validated model directory to private S3")
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("destination", help="s3://bucket/prefix")
    parser.add_argument("--sse-kms-key-id")
    args = parser.parse_args()
    model_dir = args.model_dir.resolve(strict=True)
    summary = validate_model_directory(model_dir, strict=True)
    bucket, prefix = parse_s3_source(args.destination)
    client = boto3.client("s3")
    extra = (
        {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": args.sse_kms_key_id}
        if args.sse_kms_key_id
        else {"ServerSideEncryption": "AES256"}
    )
    # Upload the manifest last so readers never observe an incomplete revision as publishable.
    for entry in summary.manifest.files:
        client.upload_file(
            str(model_dir / entry.path), bucket, f"{prefix}/{entry.path}", ExtraArgs=extra
        )
    client.upload_file(
        str(model_dir / MANIFEST_NAME), bucket, f"{prefix}/{MANIFEST_NAME}", ExtraArgs=extra
    )
    print(
        f"uploaded {len(summary.manifest.files)} artifacts and manifest to s3://{bucket}/{prefix}/"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
