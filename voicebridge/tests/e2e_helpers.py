"""E2E test helpers and utilities - backwards compatibility module."""

import os
import subprocess
from pathlib import Path
from unittest.mock import patch


class MockSystemServices:
    """Mock system services for testing."""

    @staticmethod
    def mock_gpu_unavailable():
        """Mock GPU as unavailable."""
        return patch.multiple(
            "torch",
            cuda=patch(is_available=lambda: False),
            backends=patch(mps=patch(is_available=lambda: False)),
        )

    @staticmethod
    def mock_no_audio_devices():
        """Mock no audio devices available."""
        return patch.dict(
            os.environ, {"VOICEBRIDGE_DISABLE_AUDIO": "1", "VOICEBRIDGE_NO_AUDIO": "1"}
        )

    @staticmethod
    def mock_clipboard_unavailable():
        """Mock clipboard as unavailable."""
        return patch.dict(os.environ, {"VOICEBRIDGE_NO_CLIPBOARD": "1"})


class E2ETestRunner:
    """E2E test runner - backwards compatibility wrapper."""

    def __init__(self, test_dir: Path):
        """Initialize test runner."""
        self.test_dir = Path(test_dir)
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def setup_test_environment(self) -> dict[str, Path]:
        """Set up test environment and return fixture paths."""
        fixtures_dir = self.test_dir / "fixtures" / "audio"
        fixtures_dir.mkdir(parents=True, exist_ok=True)

        # Create basic test audio files using ffmpeg if available
        fixtures = {}

        try:
            # Short audio (2 seconds, 440Hz tone)
            short_audio = fixtures_dir / "short_test.wav"
            subprocess.run(
                [
                    "ffmpeg",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:duration=2:sample_rate=16000",
                    "-y",
                    str(short_audio),
                ],
                check=True,
                capture_output=True,
            )
            fixtures["short_audio"] = short_audio

            # Medium audio (5 seconds, 880Hz tone)
            medium_audio = fixtures_dir / "medium_test.wav"
            subprocess.run(
                [
                    "ffmpeg",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=880:duration=5:sample_rate=16000",
                    "-y",
                    str(medium_audio),
                ],
                check=True,
                capture_output=True,
            )
            fixtures["medium_audio"] = medium_audio

            # Long audio (10 seconds, 660Hz tone)
            long_audio = fixtures_dir / "long_test.wav"
            subprocess.run(
                [
                    "ffmpeg",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=660:duration=10:sample_rate=16000",
                    "-y",
                    str(long_audio),
                ],
                check=True,
                capture_output=True,
            )
            fixtures["long_audio"] = long_audio

            # Voice sample (copy from project if available)
            project_voices = Path(__file__).parent.parent.parent / "voices"
            if project_voices.exists():
                for voice_file in project_voices.glob("*.wav"):
                    if "alice" in voice_file.name.lower():
                        voice_sample = fixtures_dir / "voice_sample.wav"
                        import shutil

                        shutil.copy2(voice_file, voice_sample)
                        fixtures["voice_sample"] = voice_sample
                        break

        except (subprocess.CalledProcessError, FileNotFoundError):
            # FFmpeg not available, create dummy files
            for name in ["short_audio", "medium_audio", "long_audio"]:
                dummy_file = fixtures_dir / f"{name}.wav"
                dummy_file.write_bytes(b"RIFF" + b"\x00" * 40)  # Minimal WAV header
                fixtures[name] = dummy_file

        return fixtures


class PerformanceProfiler:
    """Simple performance profiler for testing."""

    def __init__(self):
        """Initialize profiler."""
        self.start_time = None
        self.end_time = None

    def start(self):
        """Start profiling."""
        import time

        self.start_time = time.time()

    def stop(self):
        """Stop profiling."""
        import time

        self.end_time = time.time()

    def get_duration(self) -> float:
        """Get profiling duration."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time
