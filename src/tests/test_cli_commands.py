#!/usr/bin/env python3
"""Unit tests for CLI commands."""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import typer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.app import create_app
from cli.commands import CLICommands
from domain.models import WhisperConfig


class TestCLICommands(unittest.TestCase):
    """Test CLI command implementations."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_config_repo = Mock()
        self.mock_profile_repo = Mock()
        self.mock_daemon_service = Mock()
        self.mock_transcription_orchestrator = Mock()
        self.mock_system_service = Mock()
        self.mock_logger = Mock()

        self.commands = CLICommands(
            config_repo=self.mock_config_repo,
            profile_repo=self.mock_profile_repo,
            daemon_service=self.mock_daemon_service,
            transcription_orchestrator=self.mock_transcription_orchestrator,
            system_service=self.mock_system_service,
            logger=self.mock_logger,
        )

    def test_listen_basic(self):
        """Test basic listen command."""
        # Mock configuration
        config = WhisperConfig(model_name="medium")
        self.mock_config_repo.load.return_value = config

        # Mock transcription result
        self.mock_transcription_orchestrator.transcribe_single_recording.return_value = (
            "Hello world"
        )

        with patch("typer.echo") as mock_echo:
            self.commands.listen()

        # Verify transcription was called
        self.mock_transcription_orchestrator.transcribe_single_recording.assert_called_once()

        # Verify output
        mock_echo.assert_any_call("Starting transcription... (Press Ctrl+C to stop)")
        mock_echo.assert_any_call("Transcription: Hello world")

    def test_listen_with_parameters(self):
        """Test listen command with custom parameters."""
        config = WhisperConfig()
        self.mock_config_repo.load.return_value = config
        self.mock_transcription_orchestrator.transcribe_single_recording.return_value = (
            "Bonjour"
        )

        with patch("typer.echo"):
            self.commands.listen(
                model="large", language="fr", temperature=0.1, debug=True
            )

        # Verify the orchestrator was called with updated config
        call_args = (
            self.mock_transcription_orchestrator.transcribe_single_recording.call_args[
                0
            ][0]
        )
        self.assertEqual(call_args.model_name, "large")
        self.assertEqual(call_args.language, "fr")
        self.assertEqual(call_args.temperature, 0.1)
        self.assertTrue(call_args.debug)

    def test_listen_with_profile(self):
        """Test listen command with profile."""
        profile_config = WhisperConfig(model_name="small", language="es")
        self.mock_profile_repo.load_profile.return_value = profile_config
        self.mock_transcription_orchestrator.transcribe_single_recording.return_value = (
            "Hola"
        )

        with patch("typer.echo"):
            self.commands.listen(profile="spanish")

        self.mock_profile_repo.load_profile.assert_called_once_with("spanish")

    def test_listen_profile_not_found(self):
        """Test listen command with non-existent profile."""
        self.mock_profile_repo.load_profile.side_effect = FileNotFoundError(
            "Profile not found"
        )
        self.mock_config_repo.load.return_value = WhisperConfig()
        self.mock_transcription_orchestrator.transcribe_single_recording.return_value = (
            "Hello"
        )

        with patch("typer.echo") as mock_echo:
            self.commands.listen(profile="nonexistent")

        mock_echo.assert_any_call(
            "Profile 'nonexistent' not found, using default config.", err=True
        )

    def test_listen_no_speech(self):
        """Test listen command with no speech detected."""
        config = WhisperConfig()
        self.mock_config_repo.load.return_value = config
        self.mock_transcription_orchestrator.transcribe_single_recording.return_value = (
            ""
        )

        with patch("typer.echo") as mock_echo:
            self.commands.listen()

        mock_echo.assert_any_call("No speech detected.")

    def test_listen_keyboard_interrupt(self):
        """Test listen command with keyboard interrupt."""
        config = WhisperConfig()
        self.mock_config_repo.load.return_value = config
        self.mock_transcription_orchestrator.transcribe_single_recording.side_effect = (
            KeyboardInterrupt()
        )

        with patch("typer.echo") as mock_echo:
            self.commands.listen()

        mock_echo.assert_any_call("\nTranscription stopped.")

    def test_listen_error(self):
        """Test listen command with error."""
        config = WhisperConfig()
        self.mock_config_repo.load.return_value = config
        self.mock_transcription_orchestrator.transcribe_single_recording.side_effect = (
            Exception("Test error")
        )

        with patch("typer.echo") as mock_echo:
            with self.assertRaises(typer.Exit):
                self.commands.listen()

        mock_echo.assert_any_call("Error: Test error", err=True)

    def test_hotkey_basic(self):
        """Test basic hotkey command."""
        config = WhisperConfig()
        self.mock_config_repo.load.return_value = config

        with patch("typer.echo") as mock_echo:
            self.commands.hotkey()

        mock_echo.assert_any_call("Hotkey mode started. Press f9 to record.")
        mock_echo.assert_any_call("Press Esc to quit.")

    def test_daemon_start_success(self):
        """Test daemon start command success."""
        config = WhisperConfig()
        self.mock_config_repo.load.return_value = config

        with patch("typer.echo") as mock_echo:
            self.commands.daemon_start()

        self.mock_daemon_service.start.assert_called_once_with(config)
        mock_echo.assert_called_with("Daemon started successfully.")

    def test_daemon_start_error(self):
        """Test daemon start command with error."""
        self.mock_config_repo.load.return_value = WhisperConfig()
        self.mock_daemon_service.start.side_effect = RuntimeError("Already running")

        with patch("typer.echo") as mock_echo:
            with self.assertRaises(typer.Exit):
                self.commands.daemon_start()

        mock_echo.assert_called_with("Error: Already running", err=True)

    def test_daemon_stop_success(self):
        """Test daemon stop command success."""
        with patch("typer.echo") as mock_echo:
            self.commands.daemon_stop()

        self.mock_daemon_service.stop.assert_called_once()
        mock_echo.assert_called_with("Daemon stopped successfully.")

    def test_daemon_status_running(self):
        """Test daemon status when running."""
        # Mock configuration
        config = WhisperConfig(model_name="medium")
        self.mock_config_repo.load.return_value = config

        self.mock_daemon_service.get_status.return_value = {
            "running": True,
            "pid": 12345,
            "uptime": "01:30:45",
        }

        with patch("typer.echo") as mock_echo:
            self.commands.daemon_status()

        mock_echo.assert_any_call("Daemon is running (PID: 12345)")
        mock_echo.assert_any_call("  Uptime: 01:30:45")
        mock_echo.assert_any_call("Configuration:")

    def test_daemon_status_not_running(self):
        """Test daemon status when not running."""
        self.mock_daemon_service.get_status.return_value = {"running": False}

        with patch("typer.echo") as mock_echo:
            self.commands.daemon_status()

        mock_echo.assert_called_with("Daemon is not running")

    def test_config_show(self):
        """Test config show command."""
        config = WhisperConfig(model_name="large", debug=True)
        self.mock_config_repo.load.return_value = config

        with patch("typer.echo") as mock_echo:
            self.commands.config_show()

        mock_echo.assert_any_call("Current configuration:")
        # Should show each config value
        self.assertTrue(
            any("model_name: large" in str(call) for call in mock_echo.call_args_list)
        )

    def test_config_set_valid(self):
        """Test config set command with valid key."""
        config = WhisperConfig()
        self.mock_config_repo.load.return_value = config

        with patch("typer.echo") as mock_echo:
            self.commands.config_set("debug", "true")

        self.mock_config_repo.save.assert_called_once()
        mock_echo.assert_called_with("Set debug = True")

    def test_config_set_invalid_key(self):
        """Test config set command with invalid key."""
        config = WhisperConfig()
        self.mock_config_repo.load.return_value = config

        with patch("typer.echo") as mock_echo:
            with self.assertRaises(typer.Exit):
                self.commands.config_set("invalid_key", "value")

        mock_echo.assert_called_with("Unknown configuration key: invalid_key", err=True)

    def test_config_set_type_conversion(self):
        """Test config set command with type conversion."""
        config = WhisperConfig()
        self.mock_config_repo.load.return_value = config

        with patch("typer.echo"):
            # Test float conversion
            self.commands.config_set("temperature", "0.5")

        # Verify the config was updated with correct type
        saved_config = self.mock_config_repo.save.call_args[0][0]
        self.assertEqual(saved_config.temperature, 0.5)
        self.assertIsInstance(saved_config.temperature, float)

    def test_profile_save(self):
        """Test profile save command."""
        config = WhisperConfig(model_name="small")
        self.mock_config_repo.load.return_value = config

        with patch("typer.echo") as mock_echo:
            self.commands.profile_save("test_profile")

        self.mock_profile_repo.save_profile.assert_called_once_with(
            "test_profile", config
        )
        mock_echo.assert_called_with("Profile 'test_profile' saved.")

    def test_profile_load_success(self):
        """Test profile load command success."""
        config = WhisperConfig(model_name="tiny")
        self.mock_profile_repo.load_profile.return_value = config

        with patch("typer.echo") as mock_echo:
            self.commands.profile_load("test_profile")

        self.mock_profile_repo.load_profile.assert_called_once_with("test_profile")
        self.mock_config_repo.save.assert_called_once_with(config)
        mock_echo.assert_called_with("Profile 'test_profile' loaded.")

    def test_profile_load_not_found(self):
        """Test profile load command with non-existent profile."""
        self.mock_profile_repo.load_profile.side_effect = FileNotFoundError("Not found")

        with patch("typer.echo") as mock_echo:
            with self.assertRaises(typer.Exit):
                self.commands.profile_load("nonexistent")

        mock_echo.assert_called_with("Profile 'nonexistent' not found.", err=True)

    def test_profile_list_with_profiles(self):
        """Test profile list command with existing profiles."""
        self.mock_profile_repo.list_profiles.return_value = ["profile1", "profile2"]

        with patch("typer.echo") as mock_echo:
            self.commands.profile_list()

        mock_echo.assert_any_call("Available profiles:")
        mock_echo.assert_any_call("  profile1")
        mock_echo.assert_any_call("  profile2")

    def test_profile_list_empty(self):
        """Test profile list command with no profiles."""
        self.mock_profile_repo.list_profiles.return_value = []

        with patch("typer.echo") as mock_echo:
            self.commands.profile_list()

        mock_echo.assert_called_with("No profiles found.")

    def test_profile_delete_success(self):
        """Test profile delete command success."""
        self.mock_profile_repo.delete_profile.return_value = True

        with patch("typer.echo") as mock_echo:
            self.commands.profile_delete("test_profile")

        self.mock_profile_repo.delete_profile.assert_called_once_with("test_profile")
        mock_echo.assert_called_with("Profile 'test_profile' deleted.")

    def test_profile_delete_not_found(self):
        """Test profile delete command with non-existent profile."""
        self.mock_profile_repo.delete_profile.return_value = False

        with patch("typer.echo") as mock_echo:
            with self.assertRaises(typer.Exit):
                self.commands.profile_delete("nonexistent")

        mock_echo.assert_called_with("Profile 'nonexistent' not found.", err=True)


class TestCreateApp(unittest.TestCase):
    """Test CLI app creation."""

    def test_create_app(self):
        """Test creating Typer app."""
        mock_commands = Mock()
        app = create_app(mock_commands)

        self.assertIsInstance(app, typer.Typer)

        # Test that commands are registered
        # Command names might be None, but callbacks have the right names
        command_names = [
            command.callback.__name__ if command.callback else None
            for command in app.registered_commands
        ]
        expected_commands = [
            "listen",
            "hotkey",
            "start",
            "stop",
            "status",
            "config",
            "profile",
        ]

        for expected in expected_commands:
            self.assertIn(expected, command_names)


if __name__ == "__main__":
    unittest.main()
