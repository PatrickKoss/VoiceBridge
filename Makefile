.PHONY: prepare prepare-cuda prepare-tray sync lint test test-fast test-e2e test-e2e-smoke test-e2e-stt test-e2e-audio test-e2e-gpu test-e2e-api test-e2e-tts run clean help check-uv check-venv

help:
	@echo "Available commands:"
	@echo "  prepare      - Initialize .venv with uv and install dependencies (CPU)"
	@echo "  prepare-cuda - Initialize .venv with CUDA support using uv"
	@echo "  prepare-tray - Initialize .venv with system tray support using uv"
	@echo "  sync         - Sync dependencies with pyproject.toml"
	@echo "  lint         - Run ruff linting and fix issues"
	@echo "  test         - Run all tests with coverage"
	@echo "  test-fast    - Run tests without coverage"
	@echo "  test-e2e     - Run comprehensive end-to-end CLI tests"
	@echo "  test-e2e-smoke - Run quick E2E smoke tests"
	@echo "  test-e2e-stt - Run STT command E2E tests only"
	@echo "  test-e2e-audio - Run audio command E2E tests only"
	@echo "  test-e2e-gpu - Run GPU command E2E tests only"
	@echo "  test-e2e-api - Run API command E2E tests only"
	@echo "  test-e2e-tts - Run TTS command E2E tests only"
	@echo "  run          - Run the VoiceBridge CLI"
	@echo "  clean        - Clean up cache files and virtual environment"

check-uv:
	@which uv > /dev/null || (echo "Error: uv is not installed. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh" && exit 1)

check-venv: check-uv
	@if [ ! -d ".venv" ]; then \
		echo "Virtual environment .venv not found. Creating it..."; \
		uv venv .venv; \
		echo "Installing basic dependencies..."; \
		uv pip install --editable ".[dev]"; \
	fi

prepare: check-uv
	@echo "Setting up environment with uv (CPU version)..."
	uv venv .venv
	uv pip install --editable ".[dev]"
	@echo "Environment ready! Use 'make run' to start VoiceBridge"
	@echo "To activate: source .venv/bin/activate"

prepare-cuda: check-uv
	@echo "Setting up environment with uv and CUDA support..."
	uv venv .venv
	uv pip install --editable ".[dev,cuda]" --index-url https://download.pytorch.org/whl/cu121
	@echo "Environment ready with CUDA! Use 'make run' to start VoiceBridge"
	@echo "To activate: source .venv/bin/activate"

prepare-tray: check-uv
	@echo "Setting up environment with uv and system tray support..."
	uv venv .venv
	uv pip install --editable ".[dev,tray]"
	@echo "Environment ready with system tray! Use 'make run' to start VoiceBridge"
	@echo "To activate: source .venv/bin/activate"

sync: check-venv
	@echo "Syncing dependencies with pyproject.toml..."
	uv pip sync

lint: check-venv
	@echo "Running linting with ruff..."
	.venv/bin/ruff check --fix .
	.venv/bin/ruff format .

test: check-venv
	@echo "Running tests with coverage..."
	.venv/bin/pytest voicebridge/tests/ --cov=voicebridge --cov-report=term-missing

test-fast: check-venv
	@echo "Running tests without coverage..."
	.venv/bin/pytest voicebridge/tests/

test-e2e: check-venv
	@echo "Running comprehensive end-to-end CLI tests..."
	@export VOICEBRIDGE_DISABLE_AUDIO=1; \
	 export VOICEBRIDGE_TEST_MODE=1; \
	 .venv/bin/pytest --disable-warnings voicebridge/tests/e2e_tests/ -v --tb=short; \
	 status=$$?; \
	 if [ $$status -eq 0 ]; then \
	   echo "✅ Comprehensive E2E tests passed"; \
	 else \
	   echo "❌ Comprehensive E2E tests failed"; \
	 fi; \
	 exit $$status

test-e2e-smoke: check-venv
	@echo "Running E2E smoke tests..."
	@export VOICEBRIDGE_DISABLE_AUDIO=1; \
	 .venv/bin/pytest --disable-warnings voicebridge/tests/test_e2e_simple.py::TestE2ESimple::test_imports_work -q; \
	 status=$$?; \
	 if [ $$status -eq 0 ]; then \
	   echo "✅ E2E smoke tests passed"; \
	 else \
	   echo "❌ E2E smoke tests failed"; \
	 fi; \
	 exit $$status

test-e2e-stt: check-venv
	@echo "Running STT command E2E tests..."
	@export VOICEBRIDGE_DISABLE_AUDIO=1; \
	 export VOICEBRIDGE_TEST_MODE=1; \
	 .venv/bin/pytest --disable-warnings voicebridge/tests/e2e_tests/test_stt_*.py -v --tb=short; \
	 status=$$?; \
	 if [ $$status -eq 0 ]; then \
	   echo "✅ STT E2E tests passed"; \
	 else \
	   echo "❌ STT E2E tests failed"; \
	 fi; \
	 exit $$status

test-e2e-audio: check-venv
	@echo "Running audio command E2E tests..."
	@export VOICEBRIDGE_DISABLE_AUDIO=1; \
	 export VOICEBRIDGE_TEST_MODE=1; \
	 .venv/bin/pytest --disable-warnings voicebridge/tests/e2e_tests/test_audio_*.py -v --tb=short; \
	 status=$$?; \
	 if [ $$status -eq 0 ]; then \
	   echo "✅ Audio E2E tests passed"; \
	 else \
	   echo "❌ Audio E2E tests failed"; \
	 fi; \
	 exit $$status

test-e2e-gpu: check-venv
	@echo "Running GPU command E2E tests..."
	@export VOICEBRIDGE_DISABLE_AUDIO=1; \
	 export VOICEBRIDGE_TEST_MODE=1; \
	 .venv/bin/pytest --disable-warnings voicebridge/tests/e2e_tests/test_gpu_*.py -v --tb=short; \
	 status=$$?; \
	 if [ $$status -eq 0 ]; then \
	   echo "✅ GPU E2E tests passed"; \
	 else \
	   echo "❌ GPU E2E tests failed"; \
	 fi; \
	 exit $$status

test-e2e-api: check-venv
	@echo "Running API command E2E tests..."
	@export VOICEBRIDGE_DISABLE_AUDIO=1; \
	 export VOICEBRIDGE_TEST_MODE=1; \
	 .venv/bin/pytest --disable-warnings voicebridge/tests/e2e_tests/test_api_*.py -v --tb=short; \
	 status=$$?; \
	 if [ $$status -eq 0 ]; then \
	   echo "✅ API E2E tests passed"; \
	 else \
	   echo "❌ API E2E tests failed"; \
	 fi; \
	 exit $$status

test-e2e-tts: check-venv
	@echo "Running TTS command E2E tests..."
	@export VOICEBRIDGE_DISABLE_AUDIO=1; \
	 export VOICEBRIDGE_TEST_MODE=1; \
	 .venv/bin/pytest --disable-warnings voicebridge/tests/e2e_tests/test_tts_*.py -v --tb=short; \
	 status=$$?; \
	 if [ $$status -eq 0 ]; then \
	   echo "✅ TTS E2E tests passed"; \
	 else \
	   echo "❌ TTS E2E tests failed"; \
	 fi; \
	 exit $$status

run: check-venv
	.venv/bin/python -m voicebridge $(ARGS)

clean:
	rm -rf .venv/
	rm -rf venv/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .ruff_cache/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true