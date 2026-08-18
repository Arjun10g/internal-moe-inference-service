# AWS runtime contract (documentation only)

This repository provisions nothing. A separate deployment must provide:

- An ECS-compatible GPU compute target with NVIDIA runtime/driver compatible with the image's PyTorch CUDA build.
- An immutable ECR image digest and a task role—not static credentials.
- `s3:GetObject` on exactly one approved model revision prefix, `s3:ListBucket` only if separately needed, and KMS decrypt for the bucket key.
- Private connectivity to ECR API/DKR, S3, logging/metrics, secrets, and any internal caller; no public egress is required.
- A writable cache/ephemeral volume large enough for the entire checkpoint plus safe headroom.
- A read-only container root, non-root UID 10001, dropped Linux capabilities, `no-new-privileges`, CPU/RAM/GPU reservations, and a long enough startup grace period.
- Secret injection for `API_KEY` or replacement by a trusted identity-aware proxy/mesh.
- Liveness on `/health/live` and readiness on `/health/ready`; never route generation traffic before readiness.
- Graceful stop timeout at least `SHUTDOWN_GRACE_SECONDS` and load/performance alarms from `/metrics` or an approved collector.

ECR stores code/image layers. S3 stores model artifacts. ECS/GPU provides compute. The inference process owns tokenizer/model/KV state. The application is only an authenticated HTTP client.
