"""E2E-specific pytest configuration and fixtures."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from .helpers import AudioFixtureManager, CLIRunner, E2EAssertions


@pytest.fixture(scope="session")
def e2e_test_session_dir():
    """Create a session-wide test directory for E2E tests."""
    with tempfile.TemporaryDirectory(prefix="voicebridge_e2e_session_") as temp_dir:
        yield Path(temp_dir)


@pytest.fixture(scope="function")
def e2e_test_env(tmp_path):
    """Create an isolated test environment for E2E tests.

    This fixture:
    - Creates isolated temp directory for each test
    - Sets environment variables for testing
    - Prevents actual audio/GUI operations
    """
    # Environment variables for E2E testing
    test_env = {
        "VOICEBRIDGE_TEST_MODE": "1",
        "VOICEBRIDGE_DISABLE_AUDIO": "1",
        "VOICEBRIDGE_NO_GUI": "1",
        "VOICEBRIDGE_NO_HOTKEYS": "1",
        "HOME": str(tmp_path),
        "XDG_CONFIG_HOME": str(tmp_path / ".config"),
        "XDG_DATA_HOME": str(tmp_path / ".local" / "share"),
        "NO_COLOR": "1",
        "FORCE_COLOR": "0",
        # Prevent actual model downloads
        "VOICEBRIDGE_MODEL_CACHE": str(tmp_path / ".cache" / "voicebridge"),
    }

    with patch.dict(os.environ, test_env):
        yield tmp_path


@pytest.fixture
def cli_runner(e2e_test_env):
    """Create CLI runner for E2E tests."""
    return CLIRunner(e2e_test_env)


@pytest.fixture
def audio_fixtures(e2e_test_env):
    """Set up audio fixture manager."""
    manager = AudioFixtureManager(e2e_test_env)
    fixtures = manager.setup_standard_fixtures()

    yield manager, fixtures

    # Cleanup
    manager.cleanup_fixtures()


@pytest.fixture
def assertions():
    """Provide E2E assertions helper."""
    return E2EAssertions


@pytest.fixture
def sample_config():
    """Provide sample configuration for testing."""
    return {
        "whisper": {
            "model": "tiny",
            "language": "en",
            "temperature": 0.0,
            "use_gpu": False,
        },
        "tts": {"default_voice": None, "streaming": False, "cfg_scale": 1.0},
        "audio": {"sample_rate": 16000, "channels": 1, "chunk_duration": 2.0},
        "performance": {"memory_limit_mb": 1024, "parallel_workers": 1},
    }


# E2E-specific test markers
def pytest_configure(config):
    """Configure E2E-specific pytest markers."""
    config.addinivalue_line("markers", "e2e_slow: mark E2E test as slow running")
    config.addinivalue_line(
        "markers", "e2e_audio: mark E2E test as requiring audio files"
    )
    config.addinivalue_line(
        "markers", "e2e_batch: mark E2E test as batch processing test"
    )
    config.addinivalue_line(
        "markers", "e2e_config: mark E2E test as configuration test"
    )
    config.addinivalue_line(
        "markers", "e2e_session: mark E2E test as session management test"
    )


def pytest_collection_modifyitems(config, items):
    """Auto-mark E2E tests based on their names."""
    for item in items:
        # Mark slow E2E tests
        if any(
            keyword in item.name.lower()
            for keyword in ["batch", "performance", "long", "stress"]
        ):
            item.add_marker(pytest.mark.e2e_slow)

        # Mark audio-related E2E tests
        if any(
            keyword in item.name.lower()
            for keyword in ["transcribe", "audio", "stt", "tts"]
        ):
            item.add_marker(pytest.mark.e2e_audio)

        # Mark batch processing E2E tests
        if "batch" in item.name.lower():
            item.add_marker(pytest.mark.e2e_batch)

        # Mark configuration E2E tests
        if any(
            keyword in item.name.lower()
            for keyword in ["config", "profile", "settings"]
        ):
            item.add_marker(pytest.mark.e2e_config)

        # Mark session management E2E tests
        if any(
            keyword in item.name.lower()
            for keyword in ["session", "resume", "save", "load"]
        ):
            item.add_marker(pytest.mark.e2e_session)


@pytest.fixture(autouse=True)
def e2e_test_cleanup():
    """Auto cleanup for E2E tests."""
    yield

    # Kill any stray processes that might have been started
    import subprocess

    try:
        # Kill VoiceBridge daemon processes
        subprocess.run(
            ["pkill", "-f", "voicebridge.*daemon"], stderr=subprocess.DEVNULL, timeout=2
        )
        subprocess.run(
            ["pkill", "-f", "voicebridge.*hotkey"], stderr=subprocess.DEVNULL, timeout=2
        )
    except Exception:
        pass  # Ignore cleanup errors


@pytest.fixture
def mock_whisper_model():
    """Mock Whisper model for testing without actual model loading."""
    from unittest.mock import Mock, patch

    mock_model = Mock()
    mock_model.transcribe.return_value = {
        "text": "This is a test transcription.",
        "segments": [
            {
                "start": 0.0,
                "end": 3.0,
                "text": "This is a test transcription.",
                "confidence": 0.95,
            }
        ],
    }

    with patch("whisper.load_model", return_value=mock_model):
        yield mock_model


@pytest.fixture
def mock_gpu_unavailable():
    """Mock GPU as unavailable for testing."""
    from unittest.mock import patch

    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=False),
    ):
        yield


@pytest.fixture
def test_audio_file(audio_fixtures):
    """Get a simple test audio file for single-file tests."""
    manager, fixtures = audio_fixtures
    return fixtures["short_audio"]


@pytest.fixture
def test_voice_sample(audio_fixtures):
    """Get a voice sample for TTS testing."""
    manager, fixtures = audio_fixtures

    # Return first available voice sample or create a test one
    for key, path in fixtures.items():
        if key.startswith("voice_"):
            return path

    # Fallback: create a simple test audio
    return fixtures["short_audio"]
