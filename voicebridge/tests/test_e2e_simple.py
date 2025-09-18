"""
Simple E2E test to verify test framework is working.
"""

from voicebridge.tests.e2e_helpers import VoiceBridgeCLI


class TestE2ESimple:
    """Simple tests to verify E2E framework."""

    def test_imports_work(self):
        """Test that all imports work correctly."""
        # Test helper imports
        from voicebridge.tests.e2e_helpers import (
            ClipboardSimulator,
            TestAudioGenerator,
            VoiceBridgeCLI,
        )

        # Should be able to create instances
        cli = VoiceBridgeCLI()
        assert cli is not None

        generator = TestAudioGenerator()
        assert generator is not None

        clipboard = ClipboardSimulator()
        assert clipboard is not None

    def test_cli_help_command(self):
        """Test basic CLI help command."""
        import os

        # Set environment variable to disable audio initialization during tests
        env = os.environ.copy()
        env["VOICEBRIDGE_DISABLE_AUDIO"] = "1"

        cli = VoiceBridgeCLI()

        # Test that CLI help command works
        result = cli.run_command(["--help"], timeout=10, env=env)

        # Should return successfully with help text
        assert result.returncode == 0
        assert (
            "whisper" in result.stdout.lower() or "voicebridge" in result.stdout.lower()
        )
        assert "usage:" in result.stdout.lower() or "commands" in result.stdout.lower()

    def test_audio_generator(self, tmp_path):
        """Test audio file generation."""
        from voicebridge.tests.e2e_helpers import TestAudioGenerator

        generator = TestAudioGenerator()
        output_file = tmp_path / "test.wav"

        # Generate test audio
        result = generator.generate_test_audio(
            output_file, duration=1.0, text="Test audio"
        )

        assert result == output_file
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_file_manager(self, tmp_path):
        """Test file management utilities."""
        from voicebridge.tests.e2e_helpers import TestFileManager

        manager = TestFileManager(tmp_path)

        # Create test file
        test_file = manager.create_test_file("test.txt", "test content")
        assert test_file.exists()

        # Create test directory
        test_dir = manager.create_test_dir("test_dir")
        assert test_dir.exists()
        assert test_dir.is_dir()

    def test_clipboard_simulator(self):
        """Test clipboard simulation."""
        from voicebridge.tests.e2e_helpers import ClipboardSimulator

        clipboard = ClipboardSimulator()

        # Test basic operations
        test_text = "Test clipboard content"
        clipboard.set_text(test_text)

        # Should not crash
        retrieved = clipboard.get_text()
        assert isinstance(retrieved, str)

    def test_performance_profiler(self):
        """Test performance profiling utilities."""
        from voicebridge.tests.e2e_helpers import PerformanceProfiler

        profiler = PerformanceProfiler()
        assert profiler is not None

        report = profiler.get_report()
        assert isinstance(report, dict)
        assert "measurements" in report
        assert "summary" in report
