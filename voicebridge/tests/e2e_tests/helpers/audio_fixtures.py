"""Audio fixture management for E2E tests."""

import shutil
import subprocess
from pathlib import Path


class AudioFixtureManager:
    """Manages audio files for E2E testing."""

    def __init__(self, test_dir: Path, voices_dir: Path | None = None):
        """Initialize audio fixture manager.

        Args:
            test_dir: Test directory for fixtures
            voices_dir: Source directory with voice samples (defaults to project voices/)
        """
        self.test_dir = Path(test_dir)
        self.fixtures_dir = self.test_dir / "fixtures" / "audio"
        self.fixtures_dir.mkdir(parents=True, exist_ok=True)

        # Find project voices directory if not provided
        if voices_dir is None:
            # Assume we're in voicebridge/tests/e2e_tests and go up to project root
            project_root = Path(__file__).parent.parent.parent.parent.parent
            voices_dir = project_root / "voices"

        self.voices_dir = Path(voices_dir)
        self._available_voices = {}
        self._discover_voices()

    def _discover_voices(self):
        """Discover available voice samples."""
        if not self.voices_dir.exists():
            return

        for voice_file in self.voices_dir.glob("*.wav"):
            name = voice_file.stem
            self._available_voices[name] = voice_file

    @property
    def available_voices(self) -> list[str]:
        """Get list of available voice names."""
        return list(self._available_voices.keys())

    def get_voice_path(self, voice_name: str) -> Path:
        """Get path to original voice file.

        Args:
            voice_name: Name of voice (e.g., 'en-Alice_woman')

        Returns:
            Path to voice file

        Raises:
            FileNotFoundError: If voice not found
        """
        if voice_name not in self._available_voices:
            available = ", ".join(self.available_voices)
            raise FileNotFoundError(
                f"Voice '{voice_name}' not found. Available: {available}"
            )
        return self._available_voices[voice_name]

    def copy_voice_to_fixtures(
        self, voice_name: str, new_name: str | None = None
    ) -> Path:
        """Copy a voice file to the test fixtures directory.

        Args:
            voice_name: Name of voice to copy
            new_name: New name for the fixture (optional)

        Returns:
            Path to fixture file
        """
        source_path = self.get_voice_path(voice_name)
        fixture_name = new_name or voice_name
        fixture_path = self.fixtures_dir / f"{fixture_name}.wav"

        shutil.copy2(source_path, fixture_path)
        return fixture_path

    def create_test_audio(
        self,
        duration: float = 3.0,
        frequency: int = 440,
        sample_rate: int = 16000,
        name: str = "test_audio",
    ) -> Path:
        """Create a test audio file using ffmpeg.

        Args:
            duration: Duration in seconds
            frequency: Tone frequency in Hz
            sample_rate: Sample rate in Hz
            name: Output file name (without extension)

        Returns:
            Path to created audio file
        """
        output_path = self.fixtures_dir / f"{name}.wav"

        # Generate sine wave using ffmpeg
        cmd = [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={frequency}:duration={duration}:sample_rate={sample_rate}",
            "-y",  # Overwrite output file
            str(output_path),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to create test audio: {e}") from e

    def create_silent_audio(
        self, duration: float = 1.0, sample_rate: int = 16000, name: str = "silent"
    ) -> Path:
        """Create a silent audio file.

        Args:
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            name: Output file name (without extension)

        Returns:
            Path to created audio file
        """
        output_path = self.fixtures_dir / f"{name}.wav"

        cmd = [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=channel_layout=mono:sample_rate={sample_rate}",
            "-t",
            str(duration),
            "-y",
            str(output_path),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to create silent audio: {e}") from e

    def create_noisy_audio(
        self,
        duration: float = 2.0,
        noise_level: float = 0.1,
        sample_rate: int = 16000,
        name: str = "noisy",
    ) -> Path:
        """Create an audio file with noise.

        Args:
            duration: Duration in seconds
            noise_level: Noise level (0.0-1.0)
            sample_rate: Sample rate in Hz
            name: Output file name (without extension)

        Returns:
            Path to created audio file
        """
        output_path = self.fixtures_dir / f"{name}.wav"

        # Create white noise
        cmd = [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            f"anoisesrc=duration={duration}:colour=white:seed=42:sample_rate={sample_rate}",
            "-filter_complex",
            f"volume={noise_level}",
            "-y",
            str(output_path),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to create noisy audio: {e}") from e

    def get_fixture_path(self, name: str) -> Path:
        """Get path to a fixture file.

        Args:
            name: Fixture name (with or without .wav extension)

        Returns:
            Path to fixture file
        """
        if not name.endswith(".wav"):
            name += ".wav"
        return self.fixtures_dir / name

    def list_fixtures(self) -> list[str]:
        """List all available fixture files.

        Returns:
            List of fixture file names (without .wav extension)
        """
        return [f.stem for f in self.fixtures_dir.glob("*.wav")]

    def get_audio_info(self, file_path: Path) -> dict:
        """Get audio file information using ffprobe.

        Args:
            file_path: Path to audio file

        Returns:
            Dictionary with audio information
        """
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(file_path),
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            import json

            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to get audio info: {e}") from e

    def setup_standard_fixtures(self) -> dict[str, Path]:
        """Set up standard audio fixtures for testing.

        Returns:
            Dictionary mapping fixture names to paths
        """
        fixtures = {}

        # Create standard test audio files
        fixtures["short_audio"] = self.create_test_audio(
            duration=2.0, frequency=440, name="short_test"
        )
        fixtures["medium_audio"] = self.create_test_audio(
            duration=5.0, frequency=880, name="medium_test"
        )
        fixtures["long_audio"] = self.create_test_audio(
            duration=10.0, frequency=660, name="long_test"
        )
        fixtures["silent"] = self.create_silent_audio(duration=1.0)
        fixtures["noisy"] = self.create_noisy_audio(duration=2.0)

        # Copy a few voice samples if available
        if self.available_voices:
            # Copy a short voice sample (Alice is usually short)
            for voice_name in ["en-Alice_woman", "en-Carter_man"]:
                if voice_name in self.available_voices:
                    fixtures[f"voice_{voice_name}"] = self.copy_voice_to_fixtures(
                        voice_name, f"voice_{voice_name}"
                    )
                    break

        return fixtures

    def cleanup_fixtures(self):
        """Clean up all fixture files."""
        if self.fixtures_dir.exists():
            shutil.rmtree(self.fixtures_dir)
            self.fixtures_dir.mkdir(parents=True, exist_ok=True)
