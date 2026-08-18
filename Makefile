.PHONY: install lint format typecheck test test-fast test-tiny-model security audit \
	docker-build smoke smoke-tiny-model docker-smoke-tiny-model benchmark manifest preflight \
	qwen-url qwen-check qwen-download llama-runtime clean

PYTHON ?= .venv/bin/python
PYTEST ?= .venv/bin/pytest

install:
	uv sync --extra test

lint:
	.venv/bin/ruff check src tests scripts

format:
	.venv/bin/ruff format src tests scripts

typecheck:
	.venv/bin/mypy src

test:
	$(PYTEST) --cov --cov-report=term-missing

test-fast:
	$(PYTEST) -m "not tiny_model and not gpu and not external_checkpoint and not slow"

test-tiny-model:
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(PYTEST) -m tiny_model -vv

security:
	.venv/bin/bandit -c pyproject.toml -r src scripts

audit:
	.venv/bin/pip-audit

docker-build:
	docker build --target runtime -t llm-inference-service:local .

smoke:
	$(PYTHON) scripts/smoke_test.py --base-url "$${BASE_URL:-http://127.0.0.1:8000}" --api-key "$${API_KEY:-}"

smoke-tiny-model:
	$(PYTHON) scripts/run_tiny_model_smoke.py

docker-smoke-tiny-model:
	./scripts/docker_smoke.sh

benchmark:
	$(PYTHON) scripts/benchmark.py --base-url "$${BASE_URL:-http://127.0.0.1:8000}" --api-key "$${API_KEY:-}"

manifest:
	$(PYTHON) scripts/build_model_manifest.py "$${MODEL_SOURCE:?MODEL_SOURCE is required}"

preflight:
	$(PYTHON) scripts/validate_model.py "$${MODEL_SOURCE:?MODEL_SOURCE is required}"

qwen-url:
	$(PYTHON) scripts/download_qwen_gcp.py --print-url

qwen-check:
	$(PYTHON) scripts/download_qwen_gcp.py --check

qwen-download:
	$(PYTHON) scripts/download_qwen_gcp.py

llama-runtime:
	$(PYTHON) scripts/download_llama_runtime.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -r {} +
