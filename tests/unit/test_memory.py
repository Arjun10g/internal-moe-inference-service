from inference_service.model.memory import estimate_memory


def test_moe_memory_uses_total_parameter_count() -> None:
    estimate = estimate_memory(
        total_parameters=30_000_000_000,
        dtype="int4",
        hidden_size=4096,
        num_layers=48,
        context_tokens=8192,
        concurrent_sequences=1,
    )
    assert 13.9 < estimate.weights_gib < 14.1
    assert estimate.total_gib > estimate.weights_gib
