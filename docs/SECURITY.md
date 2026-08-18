# Security and threat model

## Protected assets

- Model weights and immutable revision identity.
- Application/API credentials.
- User prompts and generated content.
- GPU/CPU memory and inference availability.

## Controls

- Model sources are local directories or private `s3://` prefixes only.
- S3 uses the default AWS credential chain; no static credential arguments or files.
- Manifest-first downloads, maximum manifest size, safe relative paths, allowed extensions, per-file byte limits, streaming SHA-256, atomic cache publication, and full revalidation.
- Format-locked loading: Transformers accepts only safetensors with remote code disabled; llama.cpp accepts exactly one manifest-verified GGUF.
- The GGUF server is a pinned, SHA-256-verified upstream build, binds to a random loopback port, uses an ephemeral internal bearer key, disables web UI/agent mode, and starts offline.
- The production image runs as UID/GID 10001 and supports read-only root filesystems, dropped capabilities, and `no-new-privileges`.
- Bearer tokens are compared with `hmac.compare_digest`; unauthenticated mode is explicit.
- Body bytes, message counts/content size, prompt tokens, completion tokens, concurrency, queue wait, and generation duration are bounded.
- Prompt logging is disabled. Error responses do not expose paths, stack traces, credentials, prompt contents, or S3 internals.
- Runtime network/telemetry flags are set. The service does not call the Hub.

## Required deployment controls

- Private subnets/security groups and an internal load balancer or service discovery.
- TLS and workload-to-workload authentication at the ingress/service-mesh boundary.
- Least-privilege task role scoped to one immutable S3 prefix and required KMS key.
- S3/ECR VPC endpoints, bucket public-access block, versioning, encryption, and access logging.
- Read-only root filesystem, ephemeral/cache volume sizing, secrets injection, log retention/redaction, image signing/SBOM scanning, and CloudWatch alarms.
- Do not expose `/metrics` or model metadata outside the internal trust boundary.

## Residual risks

- Bearer authentication is not a replacement for TLS or network identity.
- Model/tokenizer and native GGUF parsers remain a supply-chain surface; pin and scan packages/runtime archives and accept only approved artifacts.
- Cancellation is cooperative; a GPU kernel already executing cannot be preempted instantly.
- Prompt injection is an application/model-behavior issue outside this transport service.
