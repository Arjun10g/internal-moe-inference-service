# Offline deployment

Runtime inference requires only the built image, approved local/S3 artifacts, AWS task credentials for S3 mode, and internal service traffic. It does not require Hugging Face, GitHub, public PyPI, or telemetry. The Docker build fetches and verifies the pinned `llama.cpp` release; the resulting runtime image performs no runtime download.

The checked upstream Linux llama.cpp archive is a CPU build. The image includes
its OpenMP and TLS runtime packages. A CUDA/Vulkan/ROCm llama.cpp build is a
separate artifact and must be independently pinned and qualified before it
replaces the bundled runtime; `--gpus` alone does not add GPU support to a CPU
binary. The Transformers path retains its own PyTorch device behavior.

Build dependencies, base images, and the pinned llama.cpp archive must be mirrored into approved internal registries for an air-gapped build. Override the Docker `BASE_IMAGE`, configure the internal Python index during the controlled build, preserve both Python and native runtime locks, and adapt the fixed runtime URL to the approved mirror without changing its digest. The runtime layer installs from wheels produced in the builder and sets `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `HF_HUB_DISABLE_TELEMETRY`, and `DO_NOT_TRACK`.

Test runtime isolation by running the container on an internal Docker network (`scripts/docker_smoke.sh`) or equivalent denied-egress environment. The smoke must load the mounted model and generate tokens; health-only validation is insufficient.
