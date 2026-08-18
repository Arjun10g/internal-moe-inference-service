from inference_service.observability.logging import redact


def test_redacts_common_secret_shapes() -> None:
    value = "authorization: BearerSecret api_key=abc123 password=hunter2"
    redacted = redact(value)
    assert "abc123" not in redacted
    assert "hunter2" not in redacted
    assert "BearerSecret" not in redacted
    assert redacted.count("[REDACTED]") == 3
