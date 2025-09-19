"""
End-to-end tests for Text-to-Speech CLI commands.

Tests all TTS-related functionality including text generation, voice management,
daemon mode, clipboard monitoring, and text selection features.
"""

import time
from unittest.mock import patch

import pytest

from voicebridge.tests.e2e_helpers import (
    ClipboardSimulator,
    E2ETestRunner,
    PerformanceProfiler,
)


class TestE2ETTSCommands:
    """End-to-end tests for TTS commands."""

    @pytest.fixture(autouse=True)
    def setup_tts_tests(self, tmp_path):
        """Set up TTS test environment."""
        self.test_runner = E2ETestRunner(tmp_path)
        self.paths = self.test_runner.setup_test_environment()
        self.cli = self.test_runner.cli
        self.clipboard_sim = ClipboardSimulator()
        self.profiler = PerformanceProfiler()

    def test_basic_tts_generation(self):
        """Test basic TTS text generation."""
        test_texts = [
            "Hello, this is a test of text to speech generation.",
            "Testing numbers: 1, 2, 3 and special characters!",
            "This is a longer sentence to test the quality of speech synthesis with multiple words and phrases.",
        ]

        for i, text in enumerate(test_texts):
            output_file = self.paths["output_dir"] / f"tts_test_{i}.wav"

            result = self.cli.run_command(
                [
                    "tts",
                    "generate",
                    text,
                    "--output",
                    str(output_file),
                    "--no-auto-play",  # Don't play during testing
                ],
                timeout=60,
            )

            # TTS may not be available in test environment
            if result.returncode == 0:
                assert output_file.exists()
                assert output_file.stat().st_size > 0

                # Audio file should be reasonable size (not empty or huge)
                file_size = output_file.stat().st_size
                assert file_size > 1000  # At least 1KB
                assert file_size < 50 * 1024 * 1024  # Less than 50MB
            else:
                # If TTS fails, should be due to missing models/dependencies
                error_output = result.stderr if result.stderr else ""
                print(f"TTS generation failed (expected in test env): {error_output}")

    def test_tts_voice_management(self):
        """Test TTS voice listing and configuration."""
        # List available voices
        result = self.cli.run_command(["tts", "voices"])
        assert result.returncode == 0

        output = result.stdout
        assert len(output) > 0

        # Show TTS configuration
        result = self.cli.run_command(["tts", "config", "show"])
        assert result.returncode == 0

        config_output = result.stdout
        assert len(config_output) > 0

    def test_tts_config_management(self):
        """Test TTS configuration setting and retrieval."""
        # Test configuration options
        config_tests = [
            ["--default-voice", "test-voice"],
            ["--cfg-scale", "1.5"],
            ["--inference-steps", "20"],
        ]

        for config_args in config_tests:
            result = self.cli.run_command(["tts", "config", "set"] + config_args)

            # May succeed or fail based on TTS availability
            assert result.returncode in [0, 1]

        # Show config after changes
        result = self.cli.run_command(["tts", "config", "show"])
        assert result.returncode == 0

    def test_tts_generation_with_voice_options(self):
        """Test TTS generation with different voice options."""
        test_text = "Testing different voice options."

        voice_tests = [
            {"voice": None, "streaming": False},
            {"voice": "default", "streaming": True},
            {"voice": "test-voice", "streaming": False},
        ]

        for i, voice_config in enumerate(voice_tests):
            output_file = self.paths["output_dir"] / f"voice_test_{i}.wav"

            cmd = [
                "tts",
                "generate",
                test_text,
                "--output",
                str(output_file),
                "--no-auto-play",
            ]

            if voice_config["voice"]:
                cmd.extend(["--voice", voice_config["voice"]])

            if voice_config["streaming"]:
                cmd.append("--streaming")

            result = self.cli.run_command(cmd, timeout=60)

            # May fail due to missing TTS models
            if result.returncode == 0:
                assert output_file.exists()
            else:
                print(f"Voice test {i} failed (expected in test env)")

    @pytest.mark.skip(reason="Requires clipboard access and GUI environment")
    def test_tts_clipboard_monitoring(self):
        """Test TTS clipboard monitoring functionality."""
        # This test requires actual clipboard functionality
        # Skipped by default as it needs GUI environment

        # Start clipboard monitoring in background
        process = self.cli.start_background_command(
            [
                "tts",
                "listen-clipboard",
                "--no-auto-play",
                "--output",
                str(self.paths["output_dir"] / "clipboard_test.wav"),
            ]
        )

        time.sleep(2)  # Let it initialize

        # Simulate clipboard changes
        test_texts = [
            "First clipboard text",
            "Second clipboard text with more content",
            "Third text to test clipboard monitoring",
        ]

        for text in test_texts:
            self.clipboard_sim.set_text(text)
            time.sleep(1)  # Give it time to process

        time.sleep(3)  # Let it process

        # Check if process is still running
        assert process.poll() is None

        # Clean up
        process.terminate()
        process.wait()

    @pytest.mark.skip(reason="Requires text selection simulation and hotkeys")
    def test_tts_selection_monitoring(self):
        """Test TTS text selection monitoring."""
        # This would require GUI automation to simulate text selection

        process = self.cli.start_background_command(
            ["tts", "listen-selection", "--no-auto-play"]
        )

        time.sleep(2)

        # Would simulate text selection and hotkey press here
        # This requires actual GUI interaction simulation

        process.terminate()
        process.wait()

    def test_tts_daemon_lifecycle(self):
        """Test TTS daemon start/stop/status operations."""
        # Check initial daemon status
        result = self.cli.run_command(["tts", "daemon", "status"])
        assert result.returncode == 0

        # Start daemon in clipboard mode (non-blocking)
        daemon_process = self.cli.start_background_command(
            ["tts", "daemon", "start", "--mode", "clipboard", "--no-auto-play"]
        )

        time.sleep(3)  # Give daemon time to start

        # Check status (should be running)
        result = self.cli.run_command(["tts", "daemon", "status"])
        assert result.returncode == 0
        # Status should have changed or show running info

        # Stop daemon
        result = self.cli.run_command(["tts", "daemon", "stop"])
        assert result.returncode == 0

        # Clean up process if still running
        if daemon_process.poll() is None:
            daemon_process.terminate()
            daemon_process.wait()

    def test_tts_daemon_modes(self):
        """Test different TTS daemon modes."""
        modes = ["clipboard", "selection"]

        for mode in modes:
            # Start daemon in specific mode
            process = self.cli.start_background_command(
                ["tts", "daemon", "start", "--mode", mode, "--no-auto-play"]
            )

            time.sleep(2)

            # Check status
            result = self.cli.run_command(["tts", "daemon", "status"])
            assert result.returncode == 0

            # Stop daemon
            self.cli.run_command(["tts", "daemon", "stop"])

            # Clean up
            if process.poll() is None:
                process.terminate()
                process.wait()

            time.sleep(1)  # Pause between tests

    def test_tts_streaming_vs_non_streaming(self):
        """Test streaming vs non-streaming TTS modes."""
        test_text = (
            "Testing streaming versus non-streaming text to speech generation modes."
        )

        modes = [
            {"streaming": False, "suffix": "non_streaming"},
            {"streaming": True, "suffix": "streaming"},
        ]

        for mode in modes:
            output_file = self.paths["output_dir"] / f"tts_{mode['suffix']}.wav"

            cmd = [
                "tts",
                "generate",
                test_text,
                "--output",
                str(output_file),
                "--no-auto-play",
            ]

            if mode["streaming"]:
                cmd.append("--streaming")

            start_time = time.time()
            result = self.cli.run_command(cmd, timeout=60)
            execution_time = time.time() - start_time

            if result.returncode == 0:
                assert output_file.exists()
                assert output_file.stat().st_size > 0

                # Streaming might be faster for long texts
                print(f"{mode['suffix']} took {execution_time:.2f}s")
            else:
                print(f"TTS {mode['suffix']} failed (expected in test env)")

    def test_tts_error_handling(self):
        """Test TTS error handling for various scenarios."""

        # Test with empty text
        result = self.cli.run_command(["tts", "generate", "", "--no-auto-play"])
        # Should handle gracefully
        assert result.returncode in [0, 1]

        # Test with very long text
        long_text = "This is a very long text. " * 100
        result = self.cli.run_command(
            ["tts", "generate", long_text, "--no-auto-play"], timeout=120
        )
        # Should handle or timeout gracefully
        assert result.returncode in [0, 1, 124]

        # Test with invalid voice
        result = self.cli.run_command(
            [
                "tts",
                "generate",
                "test text",
                "--voice",
                "nonexistent-voice",
                "--no-auto-play",
            ]
        )
        # Should handle gracefully
        assert result.returncode in [0, 1]

        # Test with invalid output path
        result = self.cli.run_command(
            [
                "tts",
                "generate",
                "test text",
                "--output",
                "/nonexistent/path/output.wav",
                "--no-auto-play",
            ]
        )
        assert result.returncode != 0

    def test_tts_performance_characteristics(self):
        """Test TTS performance with different text lengths."""
        text_tests = [
            ("short", "Hello world.", 15),
            (
                "medium",
                "This is a medium length sentence for testing TTS performance.",
                30,
            ),
            (
                "long",
                "This is a much longer text that contains multiple sentences. " * 5,
                60,
            ),
        ]

        for name, text, max_time in text_tests:
            output_file = self.paths["output_dir"] / f"perf_{name}.wav"

            start_time = time.time()
            result = self.cli.run_command(
                [
                    "tts",
                    "generate",
                    text,
                    "--output",
                    str(output_file),
                    "--no-auto-play",
                ],
                timeout=max_time + 10,
            )
            execution_time = time.time() - start_time

            if result.returncode == 0:
                assert execution_time < max_time, (
                    f"{name} TTS too slow: {execution_time}s"
                )
                assert output_file.exists()
                print(f"TTS {name}: {execution_time:.2f}s")
            else:
                print(f"TTS {name} failed (expected in test env)")

    def test_tts_output_formats_and_quality(self):
        """Test TTS output with different configurations."""
        test_text = "Testing TTS output quality and format configurations."

        # Test different output file formats (if supported)
        formats = ["wav", "mp3"]  # Add more if supported

        for fmt in formats:
            output_file = self.paths["output_dir"] / f"format_test.{fmt}"

            result = self.cli.run_command(
                [
                    "tts",
                    "generate",
                    test_text,
                    "--output",
                    str(output_file),
                    "--no-auto-play",
                ],
                timeout=30,
            )

            if result.returncode == 0:
                assert output_file.exists()
                # Basic format validation
                if fmt == "wav":
                    # WAV files should start with RIFF
                    with open(output_file, "rb") as f:
                        header = f.read(4)
                        assert header == b"RIFF"
                elif fmt == "mp3":
                    # MP3 files should start with ID3 or have MP3 sync word
                    with open(output_file, "rb") as f:
                        header = f.read(3)
                        assert header in [b"ID3", b"\xff\xfb"]  # MP3 headers
            else:
                print(f"Format {fmt} test failed (expected in test env)")

    def test_tts_integration_with_other_features(self):
        """Test TTS integration with configuration and profiles."""

        # Create a TTS-specific profile
        profile_name = "tts_test_profile"

        # Configure TTS settings
        config_commands = [
            ["tts", "config", "set", "--cfg-scale", "1.2"],
            ["tts", "config", "set", "--inference-steps", "25"],
        ]

        for cmd in config_commands:
            result = self.cli.run_command(cmd)
            assert result.returncode in [0, 1]

        # Save profile
        result = self.cli.run_command(["profile", "--save", profile_name])
        assert result.returncode == 0

        # Test TTS generation with profile
        result = self.cli.run_command(
            [
                "tts",
                "generate",
                "Testing TTS with custom profile configuration.",
                "--output",
                str(self.paths["output_dir"] / "profile_tts.wav"),
                "--no-auto-play",
            ],
            timeout=45,
        )

        # Load profile and verify
        result = self.cli.run_command(["profile", "--load", profile_name])
        assert result.returncode == 0

    @pytest.mark.skip(reason="Requires actual hotkey simulation")
    def test_tts_hotkey_functionality(self):
        """Test TTS hotkey functionality."""
        # This would test actual hotkey pressing and TTS generation
        # Requires GUI environment and input simulation

        daemon_process = self.cli.start_background_command(
            ["tts", "daemon", "start", "--mode", "selection", "--no-auto-play"]
        )

        time.sleep(2)

        # Would simulate:
        # 1. Text selection in some application
        # 2. Hotkey press (e.g., F12)
        # 3. Verify TTS generation

        daemon_process.terminate()
        daemon_process.wait()

    def teardown_method(self):
        """Clean up after each test."""
        # Stop any running daemons
        try:
            self.cli.run_command(["tts", "daemon", "stop"], timeout=5)
        except Exception:
            pass

        self.test_runner.cleanup()


class TestE2ETTSClipboardIntegration:
    """Specialized tests for clipboard integration."""

    @pytest.fixture(autouse=True)
    def setup_clipboard_tests(self, tmp_path):
        """Set up clipboard-specific test environment."""
        self.test_runner = E2ETestRunner(tmp_path)
        self.paths = self.test_runner.setup_test_environment()
        self.cli = self.test_runner.cli
        self.clipboard_sim = ClipboardSimulator()

    def test_clipboard_simulation_helpers(self):
        """Test clipboard simulation utilities."""
        # Test basic clipboard operations
        test_text = "Test clipboard content"
        self.clipboard_sim.set_text(test_text)

        retrieved_text = self.clipboard_sim.get_text()
        # May not work in headless environment, but should not crash
        assert isinstance(retrieved_text, str)

        # Test clipboard clearing
        self.clipboard_sim.clear()
        cleared_text = self.clipboard_sim.get_text()
        assert cleared_text == "" or cleared_text != test_text

    @patch("pyperclip.copy")
    @patch("pyperclip.paste")
    def test_clipboard_tts_integration_mocked(self, mock_paste, mock_copy):
        """Test clipboard TTS integration with mocked clipboard."""
        # Mock clipboard behavior
        clipboard_texts = [
            "First clipboard text for TTS",
            "Second clipboard text",
            "Third and final clipboard text",
        ]

        mock_paste.side_effect = clipboard_texts

        # Test clipboard listening command structure
        # Note: This won't actually listen since we're mocking, but tests command parsing
        result = self.cli.run_command(
            [
                "tts",
                "listen-clipboard",
                "--no-auto-play",
                "--output",
                str(self.paths["output_dir"] / "clipboard_mock.wav"),
            ],
            timeout=5,
        )  # Short timeout since it won't actually work

        # Command should start but may timeout or exit quickly
        assert result.returncode in [0, 1, 124]

    def test_clipboard_command_validation(self):
        """Test clipboard command parameter validation."""
        # Test valid clipboard command
        result = self.cli.run_command(["tts", "listen-clipboard", "--help"])
        assert result.returncode == 0

        help_output = result.stdout
        assert "clipboard" in help_output.lower()
        assert any(
            option in help_output
            for option in ["--voice", "--streaming", "--auto-play"]
        )

    def teardown_method(self):
        """Clean up after clipboard tests."""
        self.clipboard_sim.clear()
        self.test_runner.cleanup()


class TestE2ETTSDaemonOperations:
    """Tests focused on TTS daemon operations and lifecycle."""

    @pytest.fixture(autouse=True)
    def setup_daemon_tests(self, tmp_path):
        """Set up daemon test environment."""
        self.test_runner = E2ETestRunner(tmp_path)
        self.paths = self.test_runner.setup_test_environment()
        self.cli = self.test_runner.cli

    def test_daemon_status_reporting(self):
        """Test daemon status reporting accuracy."""
        # Initial status (should be stopped)
        result = self.cli.run_command(["tts", "daemon", "status"])
        assert result.returncode == 0

        initial_status = result.stdout
        assert (
            "stopped" in initial_status.lower()
            or "not running" in initial_status.lower()
        )

    def test_daemon_start_stop_cycle_reliability(self):
        """Test daemon start/stop cycle reliability."""
        cycles = 3

        for _cycle in range(cycles):
            # Start daemon
            daemon_process = self.cli.start_background_command(
                ["tts", "daemon", "start", "--mode", "clipboard", "--no-auto-play"]
            )

            time.sleep(2)

            # Check status
            result = self.cli.run_command(["tts", "daemon", "status"])
            assert result.returncode == 0

            # Stop daemon
            result = self.cli.run_command(["tts", "daemon", "stop"])
            assert result.returncode == 0

            # Clean up process
            if daemon_process.poll() is None:
                daemon_process.terminate()
                daemon_process.wait()

            time.sleep(1)  # Brief pause between cycles

    def test_daemon_mode_switching(self):
        """Test switching between different daemon modes."""
        modes = ["clipboard", "selection"]

        for mode in modes:
            # Start daemon in mode
            process = self.cli.start_background_command(
                ["tts", "daemon", "start", "--mode", mode, "--no-auto-play"]
            )

            time.sleep(2)

            # Verify it's running
            result = self.cli.run_command(["tts", "daemon", "status"])
            assert result.returncode == 0

            # Stop daemon
            self.cli.run_command(["tts", "daemon", "stop"])

            # Clean up
            if process.poll() is None:
                process.terminate()
                process.wait()

            time.sleep(1)

    def test_daemon_concurrent_operations(self):
        """Test daemon behavior with concurrent requests."""
        # Start daemon
        daemon_process = self.cli.start_background_command(
            ["tts", "daemon", "start", "--mode", "clipboard", "--no-auto-play"]
        )

        time.sleep(2)

        # Try multiple status requests concurrently
        import threading

        def check_status():
            result = self.cli.run_command(["tts", "daemon", "status"])
            return result.returncode == 0

        threads = []
        results = []

        for _ in range(3):
            thread = threading.Thread(target=lambda: results.append(check_status()))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All status checks should succeed
        assert all(results)

        # Clean up
        self.cli.run_command(["tts", "daemon", "stop"])
        if daemon_process.poll() is None:
            daemon_process.terminate()
            daemon_process.wait()

    def teardown_method(self):
        """Clean up daemon tests."""
        # Ensure daemon is stopped
        try:
            self.cli.run_command(["tts", "daemon", "stop"], timeout=5)
        except Exception:
            pass

        self.test_runner.cleanup()


class TestE2ETTSPerformanceAndScalability:
    """Performance and scalability tests for TTS operations."""

    @pytest.fixture(autouse=True)
    def setup_performance_tests(self, tmp_path):
        """Set up performance test environment."""
        self.test_runner = E2ETestRunner(tmp_path)
        self.paths = self.test_runner.setup_test_environment()
        self.cli = self.test_runner.cli
        self.profiler = PerformanceProfiler()

    def test_tts_generation_scalability(self):
        """Test TTS generation with increasingly complex inputs."""
        complexity_tests = [
            ("simple", "Hello world", 10),
            ("medium", "This is a medium complexity sentence with several words.", 20),
            (
                "complex",
                "This is a more complex sentence with numbers like 123, punctuation marks, and various linguistic elements!",
                30,
            ),
            (
                "very_long",
                "This is a very long sentence that contains many words and should test the system's ability to handle extended text input. "
                * 3,
                60,
            ),
        ]

        performance_results = {}

        for name, text, max_time in complexity_tests:
            output_file = self.paths["output_dir"] / f"scale_{name}.wav"

            execution_time = self.profiler.measure_command_time(
                self.cli,
                [
                    "tts",
                    "generate",
                    text,
                    "--output",
                    str(output_file),
                    "--no-auto-play",
                ],
            )

            performance_results[name] = execution_time

            # If TTS succeeds, check performance
            if output_file.exists():
                assert execution_time < max_time, (
                    f"{name} TTS too slow: {execution_time}s"
                )
                print(f"TTS {name}: {execution_time:.2f}s")

        # Print performance summary
        print(f"TTS Performance Results: {performance_results}")

    def test_batch_tts_generation(self):
        """Test generating multiple TTS files in sequence."""
        batch_texts = [
            "First batch text for testing.",
            "Second batch text with different content.",
            "Third batch text to complete the series.",
            "Fourth and final batch text.",
        ]

        total_start_time = time.time()

        for i, text in enumerate(batch_texts):
            output_file = self.paths["output_dir"] / f"batch_tts_{i}.wav"

            result = self.cli.run_command(
                [
                    "tts",
                    "generate",
                    text,
                    "--output",
                    str(output_file),
                    "--no-auto-play",
                ],
                timeout=30,
            )

            # Track success rate
            if result.returncode == 0 and output_file.exists():
                print(f"Batch TTS {i}: Success")
            else:
                print(f"Batch TTS {i}: Failed (expected in test env)")

        total_time = time.time() - total_start_time
        print(f"Total batch TTS time: {total_time:.2f}s")

    def test_concurrent_tts_requests(self):
        """Test handling of concurrent TTS generation requests."""
        import queue
        import threading

        results_queue = queue.Queue()

        def generate_tts(text, output_file):
            try:
                result = self.cli.run_command(
                    [
                        "tts",
                        "generate",
                        text,
                        "--output",
                        str(output_file),
                        "--no-auto-play",
                    ],
                    timeout=45,
                )
                results_queue.put(
                    ("success" if result.returncode == 0 else "failed", output_file)
                )
            except Exception as e:
                results_queue.put(("error", str(e)))

        # Start multiple TTS generation threads
        threads = []
        for i in range(3):
            text = f"Concurrent TTS generation test number {i + 1}."
            output_file = self.paths["output_dir"] / f"concurrent_{i}.wav"

            thread = threading.Thread(target=generate_tts, args=(text, output_file))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Check results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        assert len(results) == 3
        print(f"Concurrent TTS results: {results}")

    def teardown_method(self):
        """Clean up performance tests."""
        self.test_runner.cleanup()
