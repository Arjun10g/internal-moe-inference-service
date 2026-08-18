#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def create_tokenizer(output: Path, vocab_size: int):  # type: ignore[no-untyped-def]
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    pad_token, bos_token, eos_token, unknown_token = "<pad>", "<bos>", "<eos>", "<unk>"
    base_tokens = [
        pad_token,
        bos_token,
        eos_token,
        unknown_token,
        "system",
        "user",
        "assistant",
        ":",
        "Hello",
        "The",
        "sky",
        "is",
        "blue",
        "model",
        "works",
        ".",
    ]
    tokens = base_tokens + [f"token_{index}" for index in range(vocab_size - len(base_tokens))]
    vocab = {token: index for index, token in enumerate(tokens)}
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token=unknown_token))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token=bos_token,
        eos_token=eos_token,
        unk_token=unknown_token,
        pad_token=pad_token,
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\\n"
        "{% endfor %}assistant:"
    )
    tokenizer.model_max_length = 128
    tokenizer.save_pretrained(output)
    return tokenizer


def create_model(output: Path, *, architecture: str, seed: int) -> dict[str, object]:
    import torch

    torch.manual_seed(seed)
    vocab_size = 64
    tokenizer = create_tokenizer(output, vocab_size)
    if architecture == "moe":
        from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

        config = Qwen2MoeConfig(
            vocab_size=vocab_size,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
            shared_expert_intermediate_size=32,
            decoder_sparse_step=1,
            norm_topk_prob=False,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            tie_word_embeddings=False,
            use_cache=True,
        )
        model = Qwen2MoeForCausalLM(config)
    else:
        from transformers import GPT2Config, GPT2LMHeadModel

        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=32,
            n_layer=2,
            n_head=4,
            n_positions=128,
            n_ctx=128,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        model = GPT2LMHeadModel(config)
    model.eval()
    model.save_pretrained(output, safe_serialization=True, max_shard_size="10MB")
    return {
        "architecture": model.__class__.__name__,
        "model_type": config.model_type,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "seed": seed,
        "vocab_size": vocab_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a deterministic tiny offline test checkpoint"
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--architecture", choices=("moe", "dense"), default="moe")
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.force:
        parser.error(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    from inference_service.model.manifest import build_manifest, write_manifest

    details = create_model(output, architecture=args.architecture, seed=args.seed)
    manifest = build_manifest(
        output,
        model_id="tiny-offline-moe-test",
        revision=f"seed-{args.seed}",
        architecture=str(details["architecture"]),
        dtype="float32",
    )
    write_manifest(output, manifest)
    print(
        json.dumps({**details, "path": str(output), "manifest_digest": manifest.digest}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
