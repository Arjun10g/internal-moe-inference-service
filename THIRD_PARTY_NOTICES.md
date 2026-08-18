# Third-party notices

## Qwen3-Coder

- Project: Qwen3-Coder by the Qwen team.
- Base model: `Qwen/Qwen3-Coder-30B-A3B-Instruct`.
- License identified by the companion model manifest: Apache License 2.0.
- The linked `UD-Q4_K_XL` GGUF is identified by that manifest as an Unsloth
  Dynamic GGUF quantization.

This repository links to, but does not redistribute, the model weights. Review
the model card, license, and applicable use restrictions before downloading or
deploying the artifact.

## llama.cpp

- Project: `ggml-org/llama.cpp`.
- Runtime pin: `b10355` (`dd1ea524333b1e697489067d7a4c39c60d32beee`).
- License: MIT; see [`vendor/LLAMA_CPP_LICENSE.txt`](vendor/LLAMA_CPP_LICENSE.txt).

The optional runtime downloader and Docker build verify official release
archives against the byte counts and SHA-256 values in
`vendor/llama.cpp.lock.json` before extraction.
