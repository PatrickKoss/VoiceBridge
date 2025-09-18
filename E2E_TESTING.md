# VoiceBridge End-to-End Testing Guide

This document provides comprehensive guidance for running end-to-end (E2E) tests for the VoiceBridge CLI application.

## Overview

The E2E test suite validates the entire VoiceBridge application stack by running actual CLI commands against real audio files and text input. It covers:

- **Speech-to-Text (STT)**: File transcription, batch processing, real-time transcription
- **Text-to-Speech (TTS)**: Voice generation, clipboard monitoring, daemon operations
- **Audio Processing**: File analysis, preprocessing, splitting, format conversion
- **System Management**: GPU detection, performance monitoring, configuration
- **Integration Scenarios**: Complete workflows combining multiple features

## Test Structure

```
voicebridge/tests/
├── conftest.py                    # Pytest configuration and fixtures
├── e2e_helpers.py                 # Test utilities and helper classes
├── test_e2e_cli.py               # Main CLI command tests
├── test_e2e_stt_commands.py      # Speech-to-Text specific tests
├── test_e2e_tts_commands.py      # Text-to-Speech specific tests
├── test_e2e_audio_system.py      # Audio processing and system tests
└── test_e2e_simple.py            # Basic framework verification tests
```

## Prerequisites

### System Requirements

- **Python 3.10+** (tested with 3.12)
- **UV package manager** for dependency management
- **FFmpeg** for audio processing
- **Git** for version control

### Installation

1. **Install UV** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Set up the project environment**:
   ```bash
   make prepare  # or uv venv && uv pip install -e ".[dev]"
   ```

3. **Verify installation**:
   ```bash
   uv run python -m voicebridge --help
   ```

### Optional Dependencies

For full testing capabilities:

- **CUDA/PyTorch** for GPU-accelerated tests
- **Audio devices** for real-time testing (microphone/speakers)
- **GUI environment** for clipboard/hotkey tests

## Running Tests

### Quick Start

```bash
# Run basic smoke tests (5 minutes)
python run_e2e_tests.py smoke

# Run specific test suites
python run_e2e_tests.py stt tts

# Run full test suite (80+ minutes)
python run_e2e_tests.py full
```

### Test Runner Options

The `run_e2e_tests.py` script provides comprehensive test orchestration:

```bash
# List available test suites
python run_e2e_tests.py --list

# Validate test environment
python run_e2e_tests.py --validate

# Run tests with verbose output
python run_e2e_tests.py smoke --verbose

# Run tests in parallel
python run_e2e_tests.py stt tts audio --parallel 3

# Generate test report
python run_e2e_tests.py core --report results/test_report.txt

# Run with specific markers
python run_e2e_tests.py --markers "not slow" "not requires_gpu"
```

### Available Test Suites

| Suite | Description | Duration | Commands Tested |
|-------|-------------|----------|-----------------|
| `smoke` | Quick validation of all commands | ~5m | `--help`, `config --show`, basic commands |
| `core` | Core functionality tests | ~15m | Basic transcription, TTS, audio info |
| `stt` | Speech-to-Text comprehensive tests | ~20m | `transcribe`, `batch-transcribe`, `realtime` |
| `tts` | Text-to-Speech comprehensive tests | ~20m | `tts generate`, daemon, voice management |
| `audio` | Audio processing tests | ~15m | `audio info/split/preprocess` |
| `system` | System and configuration tests | ~10m | `gpu status`, `sessions`, `config` |
| `integration` | Full workflow scenarios | ~40m | End-to-end feature combinations |
| `performance` | Performance and scalability tests | ~60m | Benchmarking, stress testing |
| `full` | Complete test suite | ~80m | All of the above |

### Direct Pytest Usage

You can also run tests directly with pytest:

```bash
# Run specific test file
uv run pytest voicebridge/tests/test_e2e_stt_commands.py

# Run with coverage
uv run pytest --cov=voicebridge voicebridge/tests/

# Run specific test class
uv run pytest voicebridge/tests/test_e2e_cli.py::TestE2ECLICommands

# Run with markers
uv run pytest -m "not slow" voicebridge/tests/

# Run with verbose output
uv run pytest -v -s voicebridge/tests/test_e2e_simple.py
```

## Test Categories and Markers

Tests are categorized using pytest markers:

- `@pytest.mark.slow` - Long-running tests (>30s)
- `@pytest.mark.requires_gpu` - Tests requiring GPU/CUDA
- `@pytest.mark.requires_audio` - Tests requiring audio devices
- `@pytest.mark.requires_gui` - Tests requiring GUI environment
- `@pytest.mark.requires_network` - Tests requiring network access
- `@pytest.mark.integration` - Integration/workflow tests

### Running Filtered Tests

```bash
# Skip slow tests
uv run pytest -m "not slow"

# Run only GPU tests (if GPU available)
uv run pytest -m "requires_gpu"

# Skip GUI-dependent tests (for CI/headless)
uv run pytest -m "not requires_gui"

# Run integration tests only
uv run pytest -m "integration"
```

## Test Environment Configuration

### Environment Variables

The tests respect these environment variables:

```bash
export VOICEBRIDGE_TEST_MODE=1          # Enable test mode
export VOICEBRIDGE_NO_GUI=1             # Disable GUI features
export VOICEBRIDGE_DISABLE_AUDIO=1      # Mock audio devices
export VOICEBRIDGE_CONFIG_DIR=/tmp/test  # Isolate config
```

### Test Data

Tests generate temporary audio files and use simulated data:

- **Audio files**: Generated WAV files with speech-like characteristics
- **Voice samples**: Synthetic voice samples for TTS testing
- **Configuration**: Isolated test configuration directories
- **Sessions**: Temporary session data for testing

### CI/CD Considerations

For automated testing environments:

1. **Install system dependencies**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install -y ffmpeg portaudio19-dev python3-dev
   
   # macOS
   brew install ffmpeg portaudio
   ```

2. **Run in headless mode**:
   ```bash
   export DISPLAY=:99  # For virtual display
   python run_e2e_tests.py smoke --markers "not requires_gui"
   ```

3. **Skip hardware-dependent tests**:
   ```bash
   python run_e2e_tests.py core --markers "not requires_gpu" "not requires_audio"
   ```

## Understanding Test Results

### Success Criteria

Tests verify:

- **Command Execution**: CLI commands run without crashing
- **Output Generation**: Expected output files are created
- **Content Validation**: Output contains expected content/format
- **Error Handling**: Invalid inputs handled gracefully
- **Performance**: Operations complete within reasonable time

### Common Test Patterns

1. **Command Testing**:
   ```python
   result = cli.run_command(['transcribe', 'audio.wav'])
   assert result.returncode == 0
   assert output_file.exists()
   ```

2. **Content Validation**:
   ```python
   with open(output_file) as f:
       content = f.read()
       assert "expected text" in content.lower()
   ```

3. **Error Handling**:
   ```python
   result = cli.run_command(['transcribe', '/nonexistent.wav'])
   assert result.returncode != 0
   ```

### Expected Failures

Some tests may fail in certain environments:

- **TTS tests** may fail without proper model setup
- **GPU tests** will be skipped without CUDA
- **Audio tests** may fail in headless environments
- **Network tests** may fail without internet

This is expected and tests are designed to handle these gracefully.

## Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   # Ensure dependencies are installed
   uv pip install -e ".[dev]"
   ```

2. **Audio Generation Failures**:
   ```bash
   # Install numpy if missing
   uv pip install numpy
   ```

3. **CLI Command Failures**:
   ```bash
   # Verify VoiceBridge installation
   uv run python -m voicebridge --help
   ```

4. **Permission Errors**:
   ```bash
   # Ensure write permissions to test directory
   chmod 755 /tmp
   ```

### Debug Mode

Run tests in debug mode for more information:

```bash
# Verbose pytest output
uv run pytest -v -s --tb=long

# Debug test runner
python run_e2e_tests.py smoke --verbose

# Check environment
python run_e2e_tests.py --validate
```

### Performance Issues

If tests are slow:

1. **Run smaller test suites**:
   ```bash
   python run_e2e_tests.py smoke  # Instead of full
   ```

2. **Use parallel execution**:
   ```bash
   python run_e2e_tests.py stt tts --parallel 2
   ```

3. **Skip slow tests**:
   ```bash
   uv run pytest -m "not slow"
   ```

## Test Development

### Adding New Tests

1. **Create test file** in appropriate category:
   ```python
   # voicebridge/tests/test_e2e_my_feature.py
   class TestE2EMyFeature:
       def test_my_functionality(self, e2e_runner):
           # Test implementation
   ```

2. **Use helper classes**:
   ```python
   from voicebridge.tests.e2e_helpers import (
       VoiceBridgeCLI, TestAudioGenerator, ClipboardSimulator
   )
   ```

3. **Add appropriate markers**:
   ```python
   @pytest.mark.requires_gpu
   def test_gpu_feature(self):
       # GPU-specific test
   ```

### Test Utilities

The `e2e_helpers.py` module provides:

- **VoiceBridgeCLI**: CLI command execution
- **TestAudioGenerator**: Audio file generation
- **ClipboardSimulator**: Clipboard operation simulation
- **TestFileManager**: Test file management
- **PerformanceProfiler**: Performance measurement
- **MockSystemServices**: System service mocking

### Best Practices

1. **Use fixtures** for common setup
2. **Mock external dependencies** when appropriate
3. **Test both success and failure cases**
4. **Include performance assertions** for critical paths
5. **Clean up resources** in teardown methods
6. **Use descriptive test names** and docstrings

## Contributing

When contributing E2E tests:

1. **Run existing tests** to ensure no regressions
2. **Add tests** for new CLI features
3. **Update documentation** for new test suites
4. **Consider CI/CD compatibility** for new tests
5. **Test in multiple environments** when possible

## Support

For issues with E2E testing:

1. **Check this documentation** for common solutions
2. **Run environment validation**: `python run_e2e_tests.py --validate`
3. **Review test logs** for specific error messages
4. **Check VoiceBridge main documentation** for feature-specific help
5. **Create issues** in the repository with test output details