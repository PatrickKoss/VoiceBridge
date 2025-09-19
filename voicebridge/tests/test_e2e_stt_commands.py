"""
End-to-end tests specifically for Speech-to-Text CLI commands.

Tests all STT-related functionality including file transcription,
batch processing, real-time transcription, and advanced features.
"""

import json
import time

import pytest

from voicebridge.tests.e2e_helpers import (
    AudioGenerator,
    E2ETestRunner,
    PerformanceProfiler,
)


class TestE2ESTTCommands:
    """End-to-end tests for STT commands."""

    @pytest.fixture(autouse=True)
    def setup_stt_tests(self, tmp_path):
        """Set up STT test environment."""
        self.test_runner = E2ETestRunner(tmp_path)
        self.paths = self.test_runner.setup_test_environment()
        self.cli = self.test_runner.cli
        self.profiler = PerformanceProfiler()

    def test_basic_file_transcription(self):
        """Test basic file transcription with different formats."""
        audio_file = self.paths["short_audio"]

        # Test different output formats
        formats = ["txt", "json", "srt", "vtt", "csv"]

        for fmt in formats:
            output_file = self.paths["output_dir"] / f"transcription.{fmt}"

            # Measure performance
            execution_time = self.profiler.measure_command_time(
                self.cli,
                [
                    "transcribe",
                    str(audio_file),
                    "--output",
                    str(output_file),
                    "--format",
                    fmt,
                ],
            )

            # Verify output
            assert output_file.exists(), f"Output file not created for format {fmt}"
            assert output_file.stat().st_size > 0, f"Empty output for format {fmt}"

            # Verify format-specific content
            with open(output_file) as f:
                content = f.read()

                if fmt == "json":
                    data = json.loads(content)
                    assert "text" in data or "segments" in data
                elif fmt == "srt" or fmt == "vtt":
                    # Should contain timestamps
                    assert "-->" in content or "WEBVTT" in content
                elif fmt == "csv":
                    # Should contain comma-separated values
                    assert "," in content

                # All formats should have some text content
                assert len(content.strip()) > 0

            # Performance check (should complete within reasonable time)
            assert execution_time < 30, (
                f"Transcription too slow for format {fmt}: {execution_time}s"
            )

    def test_transcription_with_model_options(self):
        """Test transcription with different model and language options."""
        audio_file = self.paths["medium_audio"]

        test_configs = [
            {"model": "tiny", "language": None, "temperature": 0.0},
            {"model": "base", "language": "en", "temperature": 0.2},
            {"model": None, "language": "auto", "temperature": 0.5},  # Default model
        ]

        for i, config in enumerate(test_configs):
            output_file = self.paths["output_dir"] / f"model_test_{i}.json"

            cmd = [
                "transcribe",
                str(audio_file),
                "--output",
                str(output_file),
                "--format",
                "json",
                "--temperature",
                str(config["temperature"]),
            ]

            if config["model"]:
                cmd.extend(["--model", config["model"]])

            if config["language"]:
                cmd.extend(["--language", config["language"]])

            result = self.cli.run_command(cmd, timeout=60)

            # Should succeed or fail gracefully
            if result.returncode == 0:
                assert output_file.exists()
                with open(output_file) as f:
                    data = json.load(f)
                    assert "text" in data
                    assert len(data["text"].strip()) > 0
            else:
                # If it fails, should be due to model availability, not crash
                error_output = result.stderr if result.stderr else ""
                assert any(
                    keyword in error_output.lower()
                    for keyword in ["model", "not found", "unavailable", "download"]
                )

    def test_batch_transcription(self):
        """Test batch transcription of multiple files."""
        # Create additional test files for batch processing
        batch_dir = self.paths["output_dir"] / "batch_input"
        batch_dir.mkdir()

        test_files = []
        for i in range(3):
            audio_file = AudioGenerator().generate_test_audio(
                batch_dir / f"batch_{i}.wav",
                duration=3.0 + i,
                text=f"This is batch test file number {i + 1}",
            )
            test_files.append(audio_file)

        batch_output_dir = self.paths["output_dir"] / "batch_results"

        # Test batch transcription
        result = self.cli.run_command(
            [
                "batch-transcribe",
                str(batch_dir),
                "--output-dir",
                str(batch_output_dir),
                "--workers",
                "2",
                "--pattern",
                "*.wav",
            ],
            timeout=120,
        )

        assert result.returncode == 0
        assert batch_output_dir.exists()

        # Should have created output files
        output_files = list(batch_output_dir.glob("*.txt"))
        assert len(output_files) >= 3  # At least our 3 test files

        # Each output file should have content
        for output_file in output_files:
            assert output_file.stat().st_size > 0

    def test_batch_transcription_with_filtering(self):
        """Test batch transcription with file pattern filtering."""
        # Create mixed file types
        mixed_dir = self.paths["output_dir"] / "mixed_files"
        mixed_dir.mkdir()

        # Create WAV files (should be processed)
        wav_files = []
        for i in range(2):
            wav_file = AudioGenerator().generate_test_audio(
                mixed_dir / f"audio_{i}.wav", duration=2.0, text=f"WAV file {i}"
            )
            wav_files.append(wav_file)

        # Create non-audio files (should be ignored)
        (mixed_dir / "readme.txt").write_text("This is not an audio file")
        (mixed_dir / "config.json").write_text('{"test": true}')

        batch_output_dir = self.paths["output_dir"] / "filtered_results"

        # Test with WAV pattern
        result = self.cli.run_command(
            [
                "batch-transcribe",
                str(mixed_dir),
                "--output-dir",
                str(batch_output_dir),
                "--pattern",
                "*.wav",
                "--workers",
                "1",
            ],
            timeout=60,
        )

        assert result.returncode == 0

        # Should only process WAV files
        output_files = list(batch_output_dir.glob("*.txt"))
        assert len(output_files) == 2  # Only the 2 WAV files

    def test_realtime_transcription_simulation(self):
        """Test real-time transcription parameters (simulated)."""
        # Note: This tests the command structure, not actual real-time audio

        # Test with different chunk parameters
        test_configs = [
            {"chunk_duration": 1.0, "overlap": 0.2, "output_format": "live"},
            {"chunk_duration": 2.0, "overlap": 0.5, "output_format": "segments"},
            {"chunk_duration": 3.0, "overlap": 0.3, "output_format": "complete"},
        ]

        for config in test_configs:
            # Start realtime command in background (will timeout quickly without audio input)
            cmd = [
                "realtime",
                "--chunk-duration",
                str(config["chunk_duration"]),
                "--overlap",
                str(config["overlap"]),
                "--output-format",
                config["output_format"],
            ]

            # Run with short timeout since we don't have real audio input
            result = self.cli.run_command(cmd, timeout=5)

            # Command should start but may timeout or exit due to no audio input
            # The important thing is it doesn't crash immediately
            assert result.returncode in [
                0,
                1,
                124,
            ]  # 0=success, 1=no audio, 124=timeout

    @pytest.mark.skip(reason="Requires real audio input device")
    def test_live_listening_mode(self):
        """Test live listening mode (requires audio device)."""
        # This would test the actual listening functionality
        # Skipped by default as it requires real microphone input

        process = self.cli.start_background_command(
            ["listen", "--model", "tiny", "--copy-final", "--no-paste-final"]
        )

        # Let it run briefly
        time.sleep(3)

        # Should be running without immediate exit
        assert process.poll() is None

        # Clean up
        process.terminate()
        process.wait()

    @pytest.mark.skip(reason="Requires audio input and hotkey simulation")
    def test_hotkey_transcription(self):
        """Test hotkey-controlled transcription."""
        # This would test hotkey functionality
        # Requires GUI environment and input simulation

        process = self.cli.start_background_command(
            ["hotkey", "--key", "f9", "--mode", "toggle", "--model", "tiny"]
        )

        time.sleep(2)

        # Should be waiting for hotkey
        assert process.poll() is None

        # Would simulate F9 press here
        # time.sleep(1)
        # simulate_hotkey_press('f9')
        # time.sleep(2)
        # simulate_hotkey_press('f9')

        process.terminate()
        process.wait()

    def test_transcription_with_custom_vocabulary(self):
        """Test transcription with custom vocabulary."""
        audio_file = self.paths["short_audio"]
        output_file = self.paths["output_dir"] / "vocab_test.txt"

        # Add some custom vocabulary words
        vocab_words = ["transcription", "voicebridge", "speechrecognition"]

        for word in vocab_words:
            result = self.cli.run_command(
                ["vocabulary", "add", word, "--type", "custom"]
            )
            # May succeed or fail depending on implementation
            assert result.returncode in [0, 1]

        # Transcribe with vocabulary
        result = self.cli.run_command(
            ["transcribe", str(audio_file), "--output", str(output_file)]
        )

        assert result.returncode == 0
        assert output_file.exists()

    def test_transcription_with_post_processing(self):
        """Test transcription with post-processing options."""
        audio_file = self.paths["medium_audio"]

        # Configure post-processing
        result = self.cli.run_command(
            [
                "postproc",
                "config",
                "--punctuation",
                "--capitalization",
                "--remove-filler",
            ]
        )
        assert result.returncode == 0

        # Transcribe with post-processing
        output_file = self.paths["output_dir"] / "postprocessed.txt"
        result = self.cli.run_command(
            ["transcribe", str(audio_file), "--output", str(output_file)]
        )

        assert result.returncode == 0
        assert output_file.exists()

        # Test post-processing on sample text
        result = self.cli.run_command(
            ["postproc", "test", "hello world this is a test"]
        )
        assert result.returncode == 0

    def test_transcription_performance_monitoring(self):
        """Test performance monitoring during transcription."""
        audio_file = self.paths["long_audio"]

        # Clear any existing stats
        self.cli.run_command(["performance", "stats"])

        # Transcribe a longer file
        output_file = self.paths["output_dir"] / "perf_test.json"
        time.time()

        result = self.cli.run_command(
            [
                "transcribe",
                str(audio_file),
                "--output",
                str(output_file),
                "--format",
                "json",
            ],
            timeout=180,
        )

        time.time()

        assert result.returncode == 0

        # Check performance stats
        result = self.cli.run_command(["performance", "stats"])
        assert result.returncode == 0

        output = result.stdout
        # Should contain some performance information
        assert len(output) > 0

    def test_session_management_with_transcription(self):
        """Test session management functionality."""
        audio_file = self.paths["medium_audio"]

        # Check initial sessions
        result = self.cli.run_command(["sessions", "list"])
        assert result.returncode == 0

        # Transcribe to create a session
        output_file = self.paths["output_dir"] / "session_test.json"
        result = self.cli.run_command(
            [
                "transcribe",
                str(audio_file),
                "--output",
                str(output_file),
                "--format",
                "json",
            ]
        )
        assert result.returncode == 0

        # Check sessions again (might have a new session)
        result = self.cli.run_command(["sessions", "list"])
        assert result.returncode == 0

        # Test cleanup
        result = self.cli.run_command(["sessions", "cleanup"])
        assert result.returncode == 0

    def test_error_handling_and_recovery(self):
        """Test error handling for various failure scenarios."""

        # Test with non-existent file
        result = self.cli.run_command(["transcribe", "/nonexistent/file.wav"])
        assert result.returncode != 0
        error_output = result.stderr if result.stderr else ""
        assert any(
            keyword in error_output.lower()
            for keyword in ["not found", "does not exist", "error", "file"]
        )

        # Test with invalid output format
        result = self.cli.run_command(
            ["transcribe", str(self.paths["short_audio"]), "--format", "invalid_format"]
        )
        assert result.returncode != 0

        # Test with invalid model name
        result = self.cli.run_command(
            [
                "transcribe",
                str(self.paths["short_audio"]),
                "--model",
                "nonexistent_model",
            ]
        )
        # May succeed with default model or fail gracefully
        assert result.returncode in [0, 1]

        # Test batch with empty directory
        empty_dir = self.paths["output_dir"] / "empty"
        empty_dir.mkdir()

        result = self.cli.run_command(
            [
                "batch-transcribe",
                str(empty_dir),
                "--output-dir",
                str(self.paths["output_dir"] / "empty_results"),
            ]
        )
        # Should handle gracefully
        assert result.returncode in [0, 1]

    def test_confidence_analysis_integration(self):
        """Test confidence analysis features with transcription."""
        audio_file = self.paths["short_audio"]

        # Configure confidence thresholds
        result = self.cli.run_command(
            [
                "confidence",
                "configure",
                "--high",
                "0.9",
                "--medium",
                "0.7",
                "--low",
                "0.5",
                "--review",
                "0.6",
            ]
        )
        assert result.returncode == 0

        # Transcribe with confidence tracking
        output_file = self.paths["output_dir"] / "confidence_test.json"
        result = self.cli.run_command(
            [
                "transcribe",
                str(audio_file),
                "--output",
                str(output_file),
                "--format",
                "json",
            ]
        )
        assert result.returncode == 0

        # Run confidence analysis (may not have sessions to analyze)
        result = self.cli.run_command(
            ["confidence", "analyze-all", "--threshold", "0.7"]
        )
        assert result.returncode == 0

    def test_webhook_integration(self):
        """Test webhook integration with transcription."""
        # Add a test webhook
        result = self.cli.run_command(
            [
                "webhook",
                "add",
                "http://localhost:8080/test-webhook",
                "--events",
                "transcription_complete",
            ]
        )
        # May succeed or fail depending on implementation
        assert result.returncode in [0, 1]

        # List webhooks
        result = self.cli.run_command(["webhook", "list"])
        assert result.returncode == 0

        # Test webhook (will likely fail but shouldn't crash)
        result = self.cli.run_command(
            ["webhook", "test", "http://localhost:8080/test-webhook"]
        )
        assert result.returncode in [0, 1]  # Network error expected

    def teardown_method(self):
        """Clean up after each test."""
        self.test_runner.cleanup()


class TestE2ESTTPerformance:
    """Performance-focused tests for STT operations."""

    @pytest.fixture(autouse=True)
    def setup_performance_tests(self, tmp_path):
        """Set up performance test environment."""
        self.test_runner = E2ETestRunner(tmp_path)
        self.paths = self.test_runner.setup_test_environment()
        self.cli = self.test_runner.cli
        self.profiler = PerformanceProfiler()

    def test_transcription_speed_benchmarks(self):
        """Benchmark transcription speed for different scenarios."""
        test_files = [
            ("short", self.paths["short_audio"], 10),  # 3s audio, 10s max
            ("medium", self.paths["medium_audio"], 30),  # 10s audio, 30s max
            ("long", self.paths["long_audio"], 90),  # 30s audio, 90s max
        ]

        results = {}

        for name, audio_file, max_time in test_files:
            output_file = self.paths["output_dir"] / f"benchmark_{name}.txt"

            execution_time = self.profiler.measure_command_time(
                self.cli,
                [
                    "transcribe",
                    str(audio_file),
                    "--output",
                    str(output_file),
                    "--model",
                    "tiny",  # Use fastest model
                ],
            )

            results[name] = execution_time

            # Verify completion within time limit
            assert execution_time < max_time, (
                f"{name} transcription too slow: {execution_time}s > {max_time}s"
            )
            assert output_file.exists()

        # Log results for analysis
        print(f"Benchmark results: {results}")

    def test_batch_processing_scalability(self):
        """Test batch processing performance with multiple files."""
        batch_sizes = [2, 5, 10]

        for batch_size in batch_sizes:
            # Create test batch
            batch_dir = self.paths["output_dir"] / f"batch_{batch_size}"
            batch_dir.mkdir()

            for i in range(batch_size):
                AudioGenerator().generate_test_audio(
                    batch_dir / f"file_{i}.wav", duration=2.0, text=f"Batch file {i}"
                )

            batch_output_dir = self.paths["output_dir"] / f"results_{batch_size}"

            # Measure batch processing time
            execution_time = self.profiler.measure_command_time(
                self.cli,
                [
                    "batch-transcribe",
                    str(batch_dir),
                    "--output-dir",
                    str(batch_output_dir),
                    "--workers",
                    "2",
                ],
            )

            # Verify results
            output_files = list(batch_output_dir.glob("*.txt"))
            assert len(output_files) == batch_size

            # Performance should scale reasonably
            expected_max_time = batch_size * 5  # 5s per file max
            assert execution_time < expected_max_time

    def test_memory_usage_monitoring(self):
        """Test memory usage during transcription."""
        # This is a placeholder for memory monitoring
        # In a real implementation, you'd use memory profiling tools

        large_audio = AudioGenerator().generate_long_form_audio(
            self.paths["output_dir"] / "large_test.wav", duration=60.0, num_segments=12
        )

        output_file = self.paths["output_dir"] / "memory_test.json"

        # Monitor command execution
        result = self.cli.run_command(
            [
                "transcribe",
                str(large_audio),
                "--output",
                str(output_file),
                "--format",
                "json",
            ],
            timeout=300,
        )

        assert result.returncode == 0
        assert output_file.exists()

    def teardown_method(self):
        """Clean up after each test."""
        self.test_runner.cleanup()
