from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry(auto_describe=True)
        self.requests = Counter(
            "inference_requests",
            "Inference API requests",
            ["endpoint", "status"],
            registry=self.registry,
        )
        self.in_progress = Gauge(
            "inference_requests_in_progress",
            "Requests currently holding inference capacity",
            registry=self.registry,
        )
        self.request_latency = Histogram(
            "inference_request_duration_seconds",
            "End-to-end inference latency",
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 300),
            registry=self.registry,
        )
        self.ttft = Histogram(
            "inference_ttft_seconds",
            "Time to first streamed token/text chunk",
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
            registry=self.registry,
        )
        self.generated_tokens = Counter(
            "inference_generated_tokens",
            "Generated completion tokens",
            registry=self.registry,
        )
        self.model_loads = Counter(
            "model_load",
            "Model load attempts that completed successfully",
            registry=self.registry,
        )
        self.model_load_failures = Counter(
            "model_load_failures",
            "Model load attempts that failed",
            registry=self.registry,
        )
        self.model_ready = Gauge(
            "model_ready",
            "1 when the model has completed load and warmup",
            registry=self.registry,
        )
        self.model_startup = Histogram(
            "model_startup_duration_seconds",
            "Artifact resolution, validation, load, and warmup time",
            buckets=(0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600, 1800),
            registry=self.registry,
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)
