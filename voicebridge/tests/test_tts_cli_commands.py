#!/usr/bin/env python3
"""Unit tests for TTS CLI commands."""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import typer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from voicebridge.cli.commands.tts_commands import TTSCommands
from voicebridge.domain.models import TTSConfig, VoiceInfo, WhisperConfig


class TestTTSCommands(unittest.TestCase):
    """Test TTS CLI command implementations."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock all required services
        self.mock_config_repo = Mock()
        self.mock_profile_repo = Mock()
        self.mock_logger = Mock()
        self.mock_tts_orchestrator = Mock()
        self.mock_tts_daemon_service = Mock()

        # Create TTS commands instance
        self.tts_commands = TTSCommands(
            config_repo=self.mock_config_repo,
            profile_repo=self.mock_profile_repo,
            logger=self.mock_logger,
            tts_orchestrator=self.mock_tts_orchestrator,
            tts_daemon_service=self.mock_tts_daemon_service,
        )

        # Setup default config with TTS config
        self.default_tts_config = TTSConfig(default_voice="en-Alice_woman")
        self.default_config = WhisperConfig(tts_config=self.default_tts_config)
        self.mock_config_repo.load.return_value = self.default_config

    def test_tts_generate_basic(self):
        """Test basic TTS generate command."""
        # Mock successful TTS generation
        self.mock_tts_orchestrator.generate_tts_from_text.return_value = True

        with patch("typer.echo"):
            self.tts_commands.tts_generate("Hello world")

        # Verify TTS generation was called
        self.mock_tts_orchestrator.generate_tts_from_text.assert_called_once()

    def test_tts_generate_no_orchestrator(self):
        """Test TTS generate command when TTS is not available."""
        # Create commands without TTS orchestrator
        commands_no_tts = TTSCommands(
            config_repo=self.mock_config_repo,
            profile_repo=self.mock_profile_repo,
            logger=self.mock_logger,
        )

        with patch("typer.echo") as mock_echo:
            with self.assertRaises(typer.Exit):
                commands_no_tts.tts_generate("Hello world")

        # Should show error message about TTS not being available
        mock_echo.assert_any_call("Error: TTS service not available", err=True)

    def test_tts_generate_failure(self):
        """Test TTS generate command with generation failure."""
        self.mock_tts_orchestrator.generate_tts_from_text.return_value = False

        with patch("typer.echo") as mock_echo:
            with self.assertRaises(typer.Exit):
                self.tts_commands.tts_generate("Hello world")

        # Should show error message about generation failure
        mock_echo.assert_any_call("Error: TTS generation failed", err=True)

    def test_tts_generate_exception_handling(self):
        """Test TTS generate command with exception."""
        self.mock_tts_orchestrator.generate_tts_from_text.side_effect = Exception(
            "TTS generation failed"
        )

        with patch("typer.echo") as mock_echo:
            with self.assertRaises(typer.Exit):
                self.tts_commands.tts_generate("Hello world")

        # Should show error message
        mock_echo.assert_any_call(
            "Error: TTS generation error: TTS generation failed", err=True
        )

    def test_tts_listen_clipboard_no_orchestrator(self):
        """Test TTS clipboard listening when orchestrator is not available."""
        # Create commands without TTS orchestrator
        commands_no_tts = TTSCommands(
            config_repo=self.mock_config_repo,
            profile_repo=self.mock_profile_repo,
            logger=self.mock_logger,
        )

        with patch("typer.echo") as mock_echo:
            with self.assertRaises(typer.Exit):
                commands_no_tts.tts_listen_clipboard()

        # Should show error message
        mock_echo.assert_any_call("Error: TTS service not available", err=True)

    def test_tts_daemon_start_already_running(self):
        """Test starting TTS daemon when already running."""
        self.mock_tts_daemon_service.is_daemon_running.return_value = True

        with patch("typer.echo") as mock_echo:
            with self.assertRaises(typer.Exit):
                self.tts_commands.tts_daemon_start()

        # Should show error message about daemon already running
        mock_echo.assert_any_call("Error: TTS daemon is already running", err=True)

    def test_tts_daemon_start_no_service(self):
        """Test starting TTS daemon when service is not available."""
        # Create commands without TTS daemon service
        commands_no_daemon = TTSCommands(
            config_repo=self.mock_config_repo,
            profile_repo=self.mock_profile_repo,
            logger=self.mock_logger,
            tts_orchestrator=self.mock_tts_orchestrator,
        )

        with patch("typer.echo") as mock_echo:
            with self.assertRaises(typer.Exit):
                commands_no_daemon.tts_daemon_start()

        # Should show error message
        mock_echo.assert_any_call("Error: TTS daemon service not available", err=True)

    def test_tts_daemon_stop_not_running(self):
        """Test stopping TTS daemon when not running."""
        self.mock_tts_daemon_service.is_daemon_running.return_value = False

        with patch("typer.echo") as mock_echo:
            self.tts_commands.tts_daemon_stop()

        # Should show info message about daemon not running
        mock_echo.assert_any_call("Info: TTS daemon is not running")

    def test_tts_daemon_status_no_service(self):
        """Test TTS daemon status when service is not available."""
        # Create commands without TTS daemon service
        commands_no_daemon = TTSCommands(
            config_repo=self.mock_config_repo,
            profile_repo=self.mock_profile_repo,
            logger=self.mock_logger,
            tts_orchestrator=self.mock_tts_orchestrator,
        )

        with patch("typer.echo") as mock_echo:
            with self.assertRaises(typer.Exit):
                commands_no_daemon.tts_daemon_status()

        # Should show error message
        mock_echo.assert_any_call("Error: TTS daemon service not available", err=True)

    def test_tts_list_voices_success(self):
        """Test listing TTS voices successfully."""
        # Mock voice samples
        voices = {
            "en-Alice_woman": VoiceInfo(
                name="en-Alice_woman",
                file_path="/voices/en-Alice_woman.wav",
                display_name="en Alice woman",
                language="en",
                gender="woman",
            ),
            "en-Bob_man": VoiceInfo(
                name="en-Bob_man",
                file_path="/voices/en-Bob_man.wav",
                display_name="en Bob man",
                language="en",
                gender="man",
            ),
        }
        self.mock_tts_orchestrator.list_available_voices.return_value = voices

        with patch("typer.echo") as mock_echo:
            self.tts_commands.tts_list_voices()

        # Verify voices were displayed
        mock_echo.assert_any_call("Available TTS voices:")

    def test_tts_list_voices_empty(self):
        """Test listing TTS voices when none available."""
        self.mock_tts_orchestrator.list_available_voices.return_value = {}

        with patch("typer.echo") as mock_echo:
            self.tts_commands.tts_list_voices()

        # Should show no voices message (actual message from implementation)
        mock_echo.assert_any_call(
            "Info: No TTS voices found. Please add voice samples to the voices directory."
        )

    def test_tts_list_voices_no_orchestrator(self):
        """Test listing TTS voices when orchestrator is not available."""
        # Create commands without TTS orchestrator
        commands_no_tts = TTSCommands(
            config_repo=self.mock_config_repo,
            profile_repo=self.mock_profile_repo,
            logger=self.mock_logger,
        )

        with patch("typer.echo") as mock_echo:
            with self.assertRaises(typer.Exit):
                commands_no_tts.tts_list_voices()

        # Should show error message
        mock_echo.assert_any_call("Error: TTS service not available", err=True)

    def test_tts_config_show_basic(self):
        """Test showing TTS configuration."""
        # Add required attributes to avoid attribute errors
        self.default_tts_config.voice_samples_dir = "/voices"
        self.default_tts_config.voice_samples_path = (
            "/voices"  # Add both for compatibility
        )

        with patch("typer.echo") as mock_echo:
            self.tts_commands.tts_config_show()

        # Should display configuration header
        mock_echo.assert_any_call("TTS Configuration:")

    def test_tts_config_set_voice(self):
        """Test setting TTS configuration - default voice."""
        with patch("typer.echo") as mock_echo:
            self.tts_commands.tts_config_set(default_voice="en-Bob_man")

        # Verify config was updated and saved
        self.mock_config_repo.save.assert_called_once()
        # The actual message format includes emoji
        mock_echo.assert_any_call("✓ TTS configuration updated")


if __name__ == "__main__":
    unittest.main()
