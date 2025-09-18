"""
Pytest configuration and fixtures for E2E testing.

Provides shared fixtures, test markers, and configuration for running
comprehensive end-to-end tests of the VoiceBridge CLI.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from voicebridge.tests.e2e_helpers import E2ETestRunner, MockSystemServices


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a session-wide test data directory."""
    with tempfile.TemporaryDirectory(prefix="voicebridge_e2e_") as temp_dir:
        yield Path(temp_dir)


@pytest.fixture(scope="function")
def isolated_test_env(tmp_path):
    """Create an isolated test environment for each test function."""
    # Set environment variables for testing
    test_env = {
        "VOICEBRIDGE_TEST_MODE": "1",
        "VOICEBRIDGE_NO_GUI": "1",
        "VOICEBRIDGE_DISABLE_AUDIO": "1",
        "HOME": str(tmp_path),  # Isolate config directory
        "XDG_CONFIG_HOME": str(tmp_path / ".config"),
        "XDG_DATA_HOME": str(tmp_path / ".local" / "share"),
    }

    with patch.dict(os.environ, test_env):
        yield tmp_path


@pytest.fixture
def e2e_runner(isolated_test_env):
    """Create an E2E test runner with isolated environment."""
    return E2ETestRunner(isolated_test_env)


@pytest.fixture
def mock_no_gpu():
    """Mock GPU as unavailable for testing."""
    return MockSystemServices.mock_gpu_unavailable()


@pytest.fixture
def mock_no_audio():
    """Mock audio devices as unavailable for testing."""
    return MockSystemServices.mock_no_audio_devices()


@pytest.fixture
def mock_no_clipboard():
    """Mock clipboard as unavailable for testing."""
    return MockSystemServices.mock_clipboard_unavailable()


# Test markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "requires_gpu: mark test as requiring GPU")
    config.addinivalue_line(
        "markers", "requires_audio: mark test as requiring audio devices"
    )
    config.addinivalue_line(
        "markers", "requires_gui: mark test as requiring GUI environment"
    )
    config.addinivalue_line(
        "markers", "requires_network: mark test as requiring network access"
    )
    config.addinivalue_line("markers", "integration: mark test as integration test")


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests based on their names and requirements."""
    for item in items:
        # Mark slow tests
        if any(
            keyword in item.name.lower()
            for keyword in ["performance", "benchmark", "scalability", "batch"]
        ):
            item.add_marker(pytest.mark.slow)

        # Mark tests requiring specific hardware/environment
        if any(keyword in item.name.lower() for keyword in ["gpu", "cuda", "metal"]):
            item.add_marker(pytest.mark.requires_gpu)

        if any(
            keyword in item.name.lower()
            for keyword in ["audio", "microphone", "speaker"]
        ):
            item.add_marker(pytest.mark.requires_audio)

        if any(
            keyword in item.name.lower()
            for keyword in ["clipboard", "selection", "hotkey", "gui"]
        ):
            item.add_marker(pytest.mark.requires_gui)

        if any(
            keyword in item.name.lower() for keyword in ["webhook", "api", "network"]
        ):
            item.add_marker(pytest.mark.requires_network)

        if "integration" in item.name.lower() or "e2e" in item.name.lower():
            item.add_marker(pytest.mark.integration)


@pytest.fixture(autouse=True)
def setup_test_logging(caplog):
    """Set up logging for tests."""
    import logging

    caplog.set_level(logging.INFO)


@pytest.fixture
def sample_audio_files(e2e_runner):
    """Create sample audio files for testing."""
    paths = e2e_runner.setup_test_environment()
    return {
        "short": paths["short_audio"],
        "medium": paths["medium_audio"],
        "long": paths["long_audio"],
        "voice_sample": paths["voice_sample"],
    }


@pytest.fixture
def test_configurations():
    """Provide test configuration data."""
    return {
        "whisper_configs": [
            {"model": "tiny", "language": "en", "temperature": 0.0},
            {"model": "base", "language": "auto", "temperature": 0.2},
        ],
        "tts_configs": [
            {"voice": None, "streaming": False},
            {"voice": "default", "streaming": True},
        ],
        "audio_processing": [
            {"noise_reduction": 0.5, "normalize": -20},
            {"trim_silence": True, "enhance_speech": True},
        ],
    }


@pytest.fixture(scope="session", autouse=True)
def setup_test_dependencies():
    """Set up test dependencies and requirements."""
    # Check if required test dependencies are available
    required_packages = ["numpy", "pytest"]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        pytest.skip(f"Missing required test packages: {missing_packages}")


def pytest_runtest_setup(item):
    """Set up individual test runs."""
    # Skip tests based on markers and environment
    if item.get_closest_marker("requires_gpu"):
        try:
            import torch

            if not torch.cuda.is_available():
                pytest.skip("GPU required but not available")
        except ImportError:
            pytest.skip("GPU tests require PyTorch")

    if item.get_closest_marker("requires_gui"):
        # Check if we're in a headless environment
        if os.environ.get("DISPLAY") is None and os.name != "nt":
            pytest.skip("GUI required but running in headless environment")

    if item.get_closest_marker("requires_network"):
        # Could add network connectivity check here
        pass


def pytest_runtest_teardown(item):
    """Clean up after individual test runs."""
    # Clean up any remaining background processes
    import subprocess

    try:
        # Only kill background VoiceBridge daemon processes, not the current test process
        # Use a more specific pattern to avoid killing the test runner itself
        subprocess.run(
            ["pkill", "-f", "voicebridge.*daemon"], stderr=subprocess.DEVNULL, timeout=2
        )
        subprocess.run(
            ["pkill", "-f", "voicebridge.*hotkey"], stderr=subprocess.DEVNULL, timeout=2
        )
    except Exception:
        pass  # Ignore cleanup errors


@pytest.fixture
def performance_profiler():
    """Provide a performance profiler for tests."""
    from voicebridge.tests.e2e_helpers import PerformanceProfiler

    return PerformanceProfiler()


class TestEnvironmentValidator:
    """Validate test environment setup."""

    @staticmethod
    def validate_audio_support():
        """Check if audio support is available."""
        try:
            import importlib.util

            return (
                importlib.util.find_spec("wave") is not None
                and importlib.util.find_spec("numpy") is not None
            )
        except ImportError:
            return False

    @staticmethod
    def validate_cli_executable():
        """Check if CLI is properly installed."""
        import subprocess

        try:
            result = subprocess.run(
                ["uv", "run", "python", "-m", "voicebridge", "--help"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    @staticmethod
    def get_system_info():
        """Get system information for test reporting."""
        import platform

        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "architecture": platform.architecture(),
            "processor": platform.processor(),
        }


@pytest.fixture(scope="session")
def test_environment_info():
    """Provide test environment information."""
    validator = TestEnvironmentValidator()
    return {
        "system_info": validator.get_system_info(),
        "audio_support": validator.validate_audio_support(),
        "cli_executable": validator.validate_cli_executable(),
    }
