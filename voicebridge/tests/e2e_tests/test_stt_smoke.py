"""Smoke tests for STT E2E functionality - quick validation tests."""

import pytest


class TestSTTSmokeTests:
    """Quick smoke tests to validate basic STT functionality."""

    def test_cli_basic_availability(self, cli_runner):
        """Test that the VoiceBridge CLI is available and responds."""
        result = cli_runner.run("--help", timeout=10)

        # Should succeed and show help
        assert result.success, f"CLI not available: {result.stderr}"
        assert "voicebridge" in result.stdout.lower()
        assert "speech" in result.stdout.lower() or "stt" in result.stdout.lower()

    def test_stt_command_available(self, cli_runner):
        """Test that STT commands are available."""
        result = cli_runner.run("stt --help", timeout=10)

        assert result.success, f"STT commands not available: {result.stderr}"
        assert "transcribe" in result.stdout
        assert "batch-transcribe" in result.stdout

    def test_transcribe_command_help(self, cli_runner):
        """Test that transcribe command help works."""
        result = cli_runner.run_stt("transcribe", "--help", timeout=10)

        assert result.success, f"Transcribe help failed: {result.stderr}"
        assert "Usage:" in result.stdout
        assert "transcribe" in result.stdout

    def test_config_command_basic(self, cli_runner):
        """Test basic config command functionality."""
        result = cli_runner.run_stt("config", "config-show", timeout=10)

        # Config show should work (even if empty/default)
        assert result.success, f"Config show failed: {result.stderr}"

    def test_sessions_command_basic(self, cli_runner):
        """Test basic sessions command functionality."""
        result = cli_runner.run_stt("sessions", "sessions-list", timeout=10)

        # Sessions list should work (even if empty)
        assert result.success, f"Sessions list failed: {result.stderr}"

    def test_audio_fixture_creation(self, audio_fixtures):
        """Test that audio fixtures are created properly."""
        manager, fixtures = audio_fixtures

        # Should have created standard fixtures
        assert "short_audio" in fixtures
        assert fixtures["short_audio"].exists()
        assert fixtures["short_audio"].stat().st_size > 0

        # Audio info should be retrievable
        info = manager.get_audio_info(fixtures["short_audio"])
        assert "format" in info
        assert "streams" in info


class TestSTTSmokeBasicWorkflow:
    """Smoke test for basic STT workflow."""

    def test_basic_transcribe_workflow(self, cli_runner):
        """Test basic transcribe command structure validation."""
        # Test with non-existent file for quick validation (avoid model loading timeout)
        result = cli_runner.run_stt(
            "transcribe", "nonexistent_audio.wav", expect_failure=True, timeout=10
        )

        # Should fail quickly with file not found error, not timeout
        assert result.failed, "Should fail with file not found error"
        # Check that command structure is recognized (not syntax error)
        assert (
            "not found" in result.stderr.lower()
            or "no such file" in result.stderr.lower()
        )

    def test_basic_config_workflow(self, cli_runner):
        """Test basic configuration workflow."""
        # Show config
        show_result = cli_runner.run_stt("config", "config-show", timeout=10)
        assert show_result.success, f"Config show failed: {show_result.stderr}"

        # Set a simple config value
        set_result = cli_runner.run_stt(
            "config", "config-set", "model", "tiny", timeout=10
        )
        assert set_result.success, f"Config set failed: {set_result.stderr}"

    def test_environment_isolation(self, cli_runner):
        """Test that test environment is properly isolated."""
        # Check that test environment variables are set
        import os

        assert os.environ.get("VOICEBRIDGE_TEST_MODE") == "1"
        assert os.environ.get("VOICEBRIDGE_DISABLE_AUDIO") == "1"

        # Test directory should exist and be writable
        test_dir = cli_runner.test_dir
        assert test_dir.exists()
        assert test_dir.is_dir()

        # Should be able to create files in test directory
        test_file = test_dir / "test_write.txt"
        test_file.write_text("test content")
        assert test_file.exists()
        test_file.unlink()


@pytest.mark.e2e_slow
class TestSTTSmokeIntegration:
    """Smoke tests for STT integration points."""

    def test_multiple_commands_in_sequence(self, cli_runner):
        """Test running multiple STT commands in sequence."""
        commands = [
            ("config", "config-show"),
            ("sessions", "sessions-list"),
            ("performance", "stats"),
            ("operations", "operations-list"),
        ]

        for cmd, subcmd in commands:
            result = cli_runner.run_stt(cmd, subcmd, timeout=10)
            assert result.success, (
                f"Command 'stt {cmd} {subcmd}' failed: {result.stderr}"
            )

    def test_help_commands_comprehensive(self, cli_runner):
        """Test help for all major STT subcommands."""
        subcommands = [
            "transcribe",
            "batch-transcribe",
            "config",
            "sessions",
            "performance",
            "operations",
            "export",
            "confidence",
            "vocabulary",
            "postproc",
            "webhook",
            "profile",
        ]

        for subcmd in subcommands:
            result = cli_runner.run_stt(subcmd, "--help", timeout=10)
            assert result.success, f"Help for '{subcmd}' failed: {result.stderr}"
            assert "Usage:" in result.stdout or "help" in result.stdout.lower()
