"""
Helper utilities for end-to-end CLI testing.

This module provides utilities for:
- Running CLI commands
- Generating test audio files
- Simulating clipboard operations
- Managing test files and directories
"""

import json
import subprocess
import time
import wave
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pyperclip


class VoiceBridgeCLI:
    """Helper for running VoiceBridge CLI commands in tests."""

    def __init__(self, python_executable: str = ".venv/bin/python"):
        """Initialize CLI runner."""
        self.python_executable = python_executable
        self.module_path = "-m voicebridge"

    def run_command(
        self,
        args: list[str],
        timeout: int = 30,
        capture_output: bool = True,
        env: dict = None,
    ) -> subprocess.CompletedProcess:
        """
        Run a VoiceBridge CLI command and return the result.

        Args:
            args: List of command arguments (without the base command)
            timeout: Command timeout in seconds
            capture_output: Whether to capture stdout/stderr
            env: Environment variables to pass to the process

        Returns:
            CompletedProcess with the result
        """
        cmd = [self.python_executable] + self.module_path.split() + args

        try:
            result = subprocess.run(
                cmd, capture_output=capture_output, timeout=timeout, text=True, env=env
            )
            return result
        except subprocess.TimeoutExpired:
            raise Exception(
                f"Command timed out after {timeout}s: {' '.join(cmd)}"
            ) from None
        except Exception as e:
            raise Exception(f"Failed to run command {' '.join(cmd)}: {e}") from e

    def start_background_command(self, args: list[str]) -> subprocess.Popen:
        """
        Start a VoiceBridge CLI command in the background.

        Args:
            args: List of command arguments

        Returns:
            Popen process object
        """
        cmd = [self.python_executable] + self.module_path.split() + args

        return subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

    def run_command_with_input(
        self, args: list[str], input_text: str, timeout: int = 30
    ) -> subprocess.CompletedProcess:
        """
        Run a command that requires input.

        Args:
            args: Command arguments
            input_text: Text to send to stdin
            timeout: Command timeout

        Returns:
            CompletedProcess result
        """
        cmd = [self.python_executable] + self.module_path.split() + args

        return subprocess.run(
            cmd, input=input_text, capture_output=True, timeout=timeout, text=True
        )


class AudioGenerator:
    """Generate test audio files for E2E testing."""

    def __init__(self, sample_rate: int = 22050):
        """Initialize audio generator."""
        self.sample_rate = sample_rate

    def generate_test_audio(
        self,
        output_path: Path,
        duration: float = 5.0,
        text: str = "Test audio",
        add_noise: bool = False,
    ) -> Path:
        """
        Generate a test audio file with speech-like characteristics.

        Args:
            output_path: Where to save the audio file
            duration: Duration in seconds
            text: Text content (for metadata, not actual speech)
            add_noise: Whether to add background noise

        Returns:
            Path to the generated audio file
        """
        # Generate a sine wave pattern that mimics speech characteristics
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)

        # Create speech-like frequency modulation
        # Simulate formants and fundamental frequency variations
        frequencies = [200, 400, 800, 1600]  # Typical speech formants
        weights = [1.0, 0.8, 0.6, 0.4]

        audio = np.zeros(samples)
        for freq, weight in zip(frequencies, weights):
            # Add some random modulation to make it more speech-like
            mod_freq = freq + 50 * np.sin(
                2 * np.pi * 0.5 * t
            )  # Slight frequency modulation
            audio += weight * np.sin(2 * np.pi * mod_freq * t)

        # Add amplitude modulation to simulate speech patterns
        # Create pauses and volume variations
        envelope = self._create_speech_envelope(t, duration)
        audio *= envelope

        # Add noise if requested
        if add_noise:
            noise_level = 0.1
            noise = np.random.normal(0, noise_level, samples)
            audio += noise

        # Normalize and convert to 16-bit
        audio = np.clip(audio, -1.0, 1.0)
        audio_16bit = (audio * 32767).astype(np.int16)

        # Save as WAV file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "w") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_16bit.tobytes())

        # Store metadata as extended attribute or companion file
        metadata = {
            "duration": duration,
            "text": text,
            "sample_rate": self.sample_rate,
            "generated": True,
        }

        metadata_path = output_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        return output_path

    def _create_speech_envelope(self, t: np.ndarray, duration: float) -> np.ndarray:
        """Create an envelope that simulates speech patterns with pauses."""
        envelope = np.ones_like(t)

        # Add some pauses (simulate word breaks)
        num_pauses = max(1, int(duration / 3))  # Pause every ~3 seconds
        for i in range(num_pauses):
            pause_start = (i + 1) * duration / (num_pauses + 1) - 0.1
            pause_end = pause_start + 0.2

            pause_mask = (t >= pause_start) & (t <= pause_end)
            envelope[pause_mask] *= 0.1  # Reduce volume during pauses

        # Add overall amplitude variation
        variation = 0.8 + 0.2 * np.sin(2 * np.pi * 0.3 * t)  # Slow amplitude variation
        envelope *= variation

        return envelope

    def generate_voice_sample(
        self, output_path: Path, text: str = "Voice sample"
    ) -> Path:
        """
        Generate a voice sample file for TTS testing.

        Args:
            output_path: Where to save the voice sample
            text: Sample text (for metadata)

        Returns:
            Path to generated voice sample
        """
        # Voice samples should be shorter and have different characteristics
        return self.generate_test_audio(
            output_path, duration=3.0, text=text, add_noise=False
        )

    def generate_long_form_audio(
        self, output_path: Path, duration: float = 60.0, num_segments: int = 10
    ) -> Path:
        """
        Generate longer audio with multiple segments (for testing segmentation).

        Args:
            output_path: Output file path
            duration: Total duration in seconds
            num_segments: Number of distinct segments

        Returns:
            Path to generated audio file
        """
        samples_per_segment = int(self.sample_rate * duration / num_segments)
        total_samples = samples_per_segment * num_segments

        audio = np.zeros(total_samples)

        for i in range(num_segments):
            start_idx = i * samples_per_segment
            end_idx = (i + 1) * samples_per_segment

            # Generate segment with different characteristics
            segment_duration = duration / num_segments
            t_segment = np.linspace(0, segment_duration, samples_per_segment, False)

            # Vary frequency for each segment
            base_freq = 200 + (i * 100) % 800  # Vary base frequency
            segment_audio = np.sin(2 * np.pi * base_freq * t_segment)

            # Add segment breaks
            if i < num_segments - 1:
                # Fade out at the end of segment
                fade_samples = int(0.1 * self.sample_rate)  # 0.1 second fade
                fade_out = np.linspace(1, 0, fade_samples)
                segment_audio[-fade_samples:] *= fade_out

            audio[start_idx:end_idx] = segment_audio

        # Normalize and save
        audio = np.clip(audio, -1.0, 1.0)
        audio_16bit = (audio * 32767).astype(np.int16)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "w") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_16bit.tobytes())

        return output_path


class ClipboardSimulator:
    """Simulate clipboard operations for testing."""

    def __init__(self):
        """Initialize clipboard simulator."""
        self.current_text = ""
        self.history = []

    def set_text(self, text: str) -> None:
        """
        Set clipboard text (simulated).

        Args:
            text: Text to put on clipboard
        """
        self.history.append(self.current_text)
        self.current_text = text

        # Try to actually set clipboard if possible
        try:
            pyperclip.copy(text)
        except Exception:
            pass  # Ignore clipboard errors in headless environments

    def get_text(self) -> str:
        """
        Get current clipboard text.

        Returns:
            Current clipboard text
        """
        try:
            # Try to get real clipboard first
            return pyperclip.paste()
        except Exception:
            # Fall back to simulated clipboard
            return self.current_text

    def clear(self) -> None:
        """Clear clipboard."""
        self.current_text = ""
        try:
            pyperclip.copy("")
        except Exception:
            pass

    def simulate_text_change(self, text: str, delay: float = 0.1) -> None:
        """
        Simulate a clipboard text change with delay.

        Args:
            text: New text
            delay: Delay before change
        """
        if delay > 0:
            time.sleep(delay)
        self.set_text(text)


class FileManager:
    """Manage test files and directories."""

    def __init__(self, base_path: Path):
        """Initialize file manager."""
        self.base_path = base_path
        self.created_files = []
        self.created_dirs = []

    def create_test_dir(self, name: str) -> Path:
        """
        Create a test directory.

        Args:
            name: Directory name

        Returns:
            Path to created directory
        """
        dir_path = self.base_path / name
        dir_path.mkdir(parents=True, exist_ok=True)
        self.created_dirs.append(dir_path)
        return dir_path

    def create_test_file(self, name: str, content: str = "") -> Path:
        """
        Create a test file.

        Args:
            name: File name
            content: File content

        Returns:
            Path to created file
        """
        file_path = self.base_path / name
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            f.write(content)

        self.created_files.append(file_path)
        return file_path

    def create_config_file(
        self, config_data: dict[str, Any], filename: str = "test_config.json"
    ) -> Path:
        """
        Create a configuration file.

        Args:
            config_data: Configuration data
            filename: Config file name

        Returns:
            Path to config file
        """
        config_path = self.base_path / filename
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

        self.created_files.append(config_path)
        return config_path

    def cleanup(self) -> None:
        """Clean up created test files and directories."""
        for file_path in self.created_files:
            try:
                file_path.unlink(missing_ok=True)
            except Exception:
                pass

        for dir_path in self.created_dirs:
            try:
                dir_path.rmdir()
            except Exception:
                pass

        self.created_files.clear()
        self.created_dirs.clear()

    def get_file_list(self, pattern: str = "*") -> list[Path]:
        """
        Get list of files matching pattern.

        Args:
            pattern: File pattern

        Returns:
            List of matching files
        """
        return list(self.base_path.glob(pattern))

    def assert_file_exists(self, file_path: Path, message: str = None) -> None:
        """
        Assert that a file exists.

        Args:
            file_path: Path to check
            message: Optional error message
        """
        if not file_path.exists():
            msg = message or f"Expected file does not exist: {file_path}"
            raise AssertionError(msg)

    def assert_file_contains(
        self, file_path: Path, text: str, message: str = None
    ) -> None:
        """
        Assert that a file contains specific text.

        Args:
            file_path: File to check
            text: Text to search for
            message: Optional error message
        """
        self.assert_file_exists(file_path)

        with open(file_path) as f:
            content = f.read()

        if text not in content:
            msg = message or f"File {file_path} does not contain: {text}"
            raise AssertionError(msg)


class MockSystemServices:
    """Mock system services for testing."""

    @staticmethod
    def mock_gpu_unavailable():
        """Mock GPU as unavailable."""
        return patch(
            "voicebridge.adapters.system.SystemAdapter.detect_gpu_devices",
            return_value=[],
        )

    @staticmethod
    def mock_no_audio_devices():
        """Mock no audio devices available."""
        return patch(
            "voicebridge.adapters.audio.AudioAdapter.get_audio_devices", return_value=[]
        )

    @staticmethod
    def mock_clipboard_unavailable():
        """Mock clipboard as unavailable."""

        def mock_clipboard_error(*args, **kwargs):
            raise Exception("Clipboard not available in test environment")

        return patch("pyperclip.copy", side_effect=mock_clipboard_error)


class PerformanceProfiler:
    """Profile performance of CLI operations."""

    def __init__(self):
        """Initialize profiler."""
        self.measurements = {}

    def measure_command_time(self, cli: VoiceBridgeCLI, args: list[str]) -> float:
        """
        Measure command execution time.

        Args:
            cli: CLI runner
            args: Command arguments

        Returns:
            Execution time in seconds
        """
        start_time = time.time()
        result = cli.run_command(args)
        end_time = time.time()

        execution_time = end_time - start_time
        command_name = (
            " ".join(args[:2]) if len(args) >= 2 else args[0] if args else "unknown"
        )

        self.measurements[command_name] = {
            "time": execution_time,
            "success": result.returncode == 0,
            "timestamp": start_time,
        }

        return execution_time

    def get_report(self) -> dict[str, Any]:
        """
        Get performance report.

        Returns:
            Performance measurements
        """
        return {
            "measurements": self.measurements,
            "summary": {
                "total_commands": len(self.measurements),
                "successful_commands": sum(
                    1 for m in self.measurements.values() if m["success"]
                ),
                "average_time": sum(m["time"] for m in self.measurements.values())
                / max(1, len(self.measurements)),
                "slowest_command": max(
                    self.measurements.items(),
                    key=lambda x: x[1]["time"],
                    default=("none", {"time": 0}),
                )[0],
            },
        }


class E2ETestRunner:
    """Orchestrate E2E test execution."""

    def __init__(self, test_dir: Path):
        """Initialize test runner."""
        self.test_dir = test_dir
        self.cli = VoiceBridgeCLI()
        self.file_manager = FileManager(test_dir)
        self.audio_generator = AudioGenerator()
        self.profiler = PerformanceProfiler()

    def setup_test_environment(self) -> dict[str, Path]:
        """
        Set up comprehensive test environment.

        Returns:
            Dictionary of created paths
        """
        paths = {}

        # Create directory structure
        paths["audio_dir"] = self.file_manager.create_test_dir("test_audio")
        paths["output_dir"] = self.file_manager.create_test_dir("output")
        paths["voices_dir"] = self.file_manager.create_test_dir("voices")
        paths["config_dir"] = self.file_manager.create_test_dir("config")

        # Generate test audio files
        paths["short_audio"] = self.audio_generator.generate_test_audio(
            paths["audio_dir"] / "short.wav", duration=3.0, text="Short test audio"
        )

        paths["medium_audio"] = self.audio_generator.generate_test_audio(
            paths["audio_dir"] / "medium.wav",
            duration=10.0,
            text="Medium length test audio with more content",
        )

        paths["long_audio"] = self.audio_generator.generate_long_form_audio(
            paths["audio_dir"] / "long.wav", duration=30.0, num_segments=6
        )

        # Generate voice samples
        paths["voice_sample"] = self.audio_generator.generate_voice_sample(
            paths["voices_dir"] / "en-test_voice.wav"
        )

        return paths

    def run_smoke_tests(self) -> dict[str, bool]:
        """
        Run smoke tests for all major commands.

        Returns:
            Dictionary of test results
        """
        smoke_tests = {
            "config_show": ["config", "--show"],
            "gpu_status": ["gpu", "status"],
            "audio_formats": ["audio", "formats"],
            "tts_voices": ["tts", "voices"],
            "sessions_list": ["sessions", "list"],
            "export_formats": ["export", "formats"],
            "help": ["--help"],
        }

        results = {}
        for test_name, args in smoke_tests.items():
            try:
                result = self.cli.run_command(args, timeout=10)
                results[test_name] = result.returncode == 0
            except Exception:
                results[test_name] = False

        return results

    def cleanup(self) -> None:
        """Clean up test environment."""
        self.file_manager.cleanup()
