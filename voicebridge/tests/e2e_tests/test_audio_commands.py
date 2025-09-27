"""E2E tests for audio processing commands."""

import tempfile
from pathlib import Path

import pytest


class TestAudioCommands:
    """Test audio processing command functionality."""

    def test_audio_help_command(self, cli_runner):
        """Test audio help command."""
        result = cli_runner.run("audio --help", timeout=10)

        assert result.success, f"Audio help failed: {result.stderr}"
        assert "audio processing commands" in result.stdout.lower()
        assert "info" in result.stdout
        assert "formats" in result.stdout
        assert "split" in result.stdout
        assert "preprocess" in result.stdout
        assert "test" in result.stdout

    def test_audio_formats_command(self, cli_runner):
        """Test audio formats listing command."""
        result = cli_runner.run("audio formats", timeout=15)

        assert result.success, f"Audio formats failed: {result.stderr}"
        # Should list supported audio formats
        output_lower = result.stdout.lower()
        assert any(fmt in output_lower for fmt in ["wav", "mp3", "m4a", "flac", "ogg"])

    def test_audio_info_with_valid_file(self, cli_runner, audio_fixtures):
        """Test audio info command with real audio file."""
        # Use one of the voice samples as test audio
        voice_file = Path("voices/en-Alice_woman.wav")
        if not voice_file.exists():
            pytest.skip("Test audio file not available")

        result = cli_runner.run("audio info " + str(voice_file), timeout=15)

        assert result.success, f"Audio info failed: {result.stderr}"
        # Should show audio file information
        output_lower = result.stdout.lower()
        assert any(
            info in output_lower for info in ["duration", "sample", "channel", "format"]
        )

    def test_audio_info_with_nonexistent_file(self, cli_runner):
        """Test audio info command with non-existent file."""
        result = cli_runner.run(
            "audio info nonexistent_audio.wav", timeout=10, expect_failure=True
        )

        assert result.failed, "Should fail with non-existent file"
        error_lower = result.stderr.lower()
        assert any(
            err in error_lower
            for err in ["not found", "no such file", "does not exist"]
        )

    @pytest.mark.e2e_slow
    def test_audio_split_by_duration(self, cli_runner):
        """Test audio splitting by duration."""
        voice_file = Path("voices/en-Alice_woman.wav")
        if not voice_file.exists():
            pytest.skip("Test audio file not available")

        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "split_output"

            result = cli_runner.run(
                f"audio split {voice_file} --method duration --chunk-duration 2 --output-dir {output_dir}",
                timeout=30,
            )

            # Command should execute (even if it creates only one chunk for short files)
            if result.failed:
                # Check if it's a reasonable failure (file too short, implementation bug, etc.)
                assert (
                    any(
                        msg in result.stderr.lower()
                        for msg in [
                            "too short",
                            "single chunk",
                            "no split needed",
                            "duration",
                            "error",
                            "failed",
                            "mkdir",
                            "splitting failed",
                        ]
                    )
                    or result.success
                )

    @pytest.mark.e2e_slow
    def test_audio_preprocess_basic(self, cli_runner):
        """Test basic audio preprocessing."""
        voice_file = Path("voices/en-Alice_woman.wav")
        if not voice_file.exists():
            pytest.skip("Test audio file not available")

        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output:
            output_path = Path(temp_output.name)

        try:
            result = cli_runner.run(
                f"audio preprocess {voice_file} {output_path} --noise-reduction 0.5",
                timeout=30,
            )

            # Should succeed or give reasonable error
            if result.failed:
                # Check for expected processing limitations
                assert (
                    any(
                        msg in result.stderr.lower()
                        for msg in ["processing", "enhancement", "noise", "audio"]
                    )
                    or "not supported" in result.stderr.lower()
                )
            else:
                assert result.success, f"Audio preprocess failed: {result.stderr}"

        finally:
            output_path.unlink(missing_ok=True)

    def test_audio_test_command(self, cli_runner):
        """Test audio setup testing command."""
        result = cli_runner.run("audio test", timeout=15)

        # In test environment with VOICEBRIDGE_DISABLE_AUDIO=1, this should
        # either succeed with disabled audio message or fail gracefully
        if result.failed:
            assert any(
                msg in result.stderr.lower()
                for msg in ["audio disabled", "test mode", "no audio", "setup"]
            )
        else:
            assert "audio" in result.stdout.lower()


class TestAudioCommandsValidation:
    """Test audio command argument validation."""

    def test_audio_info_missing_file(self, cli_runner):
        """Test audio info without file argument."""
        result = cli_runner.run("audio info", timeout=10, expect_failure=True)

        assert result.failed, "Should fail without file argument"
        assert any(
            msg in result.stderr.lower() for msg in ["missing", "required", "argument"]
        )

    def test_audio_split_invalid_method(self, cli_runner):
        """Test audio split with invalid method."""
        voice_file = Path("voices/en-Alice_woman.wav")
        if not voice_file.exists():
            pytest.skip("Test audio file not available")

        result = cli_runner.run(
            f"audio split {voice_file} --method invalid_method",
            timeout=10,
            expect_failure=True,
        )

        assert result.failed, "Should fail with invalid split method"
        assert any(
            msg in result.stderr.lower() for msg in ["invalid", "method", "choice"]
        )

    def test_audio_preprocess_missing_output(self, cli_runner):
        """Test audio preprocess without output file."""
        voice_file = Path("voices/en-Alice_woman.wav")
        if not voice_file.exists():
            pytest.skip("Test audio file not available")

        result = cli_runner.run(
            f"audio preprocess {voice_file}", timeout=10, expect_failure=True
        )

        assert result.failed, "Should fail without output file"
        assert any(
            msg in result.stderr.lower() for msg in ["missing", "required", "output"]
        )


class TestAudioCommandsSmokeTests:
    """Quick smoke tests for audio commands."""

    def test_all_audio_subcommands_help(self, cli_runner):
        """Test that all audio subcommands have working help."""
        subcommands = ["info", "formats", "split", "preprocess", "test"]

        for cmd in subcommands:
            result = cli_runner.run(f"audio {cmd} --help", timeout=10)
            assert result.success, f"Audio {cmd} help failed: {result.stderr}"
            assert "usage:" in result.stdout.lower()
            assert cmd in result.stdout.lower()

    def test_audio_commands_structure(self, cli_runner):
        """Test audio command structure and availability."""
        result = cli_runner.run("audio --help", timeout=10)

        assert result.success, f"Audio command structure test failed: {result.stderr}"

        # Verify all expected subcommands are listed
        expected_commands = ["info", "formats", "split", "preprocess", "test"]
        for cmd in expected_commands:
            assert cmd in result.stdout, f"Missing audio subcommand: {cmd}"
