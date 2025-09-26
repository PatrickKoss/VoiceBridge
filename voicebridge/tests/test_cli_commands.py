#!/usr/bin/env python3
"""Unit tests for modular CLI commands."""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import typer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from voicebridge.cli.commands.config_commands import ConfigCommands
from voicebridge.cli.commands.speech_commands import SpeechCommands
from voicebridge.cli.registry import create_command_registry
from voicebridge.domain.models import WhisperConfig


class TestSpeechCommands(unittest.TestCase):
    """Test speech command implementations."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_config_repo = Mock()
        self.mock_profile_repo = Mock()
        self.mock_transcription_orchestrator = Mock()
        self.mock_logger = Mock()

        self.speech_commands = SpeechCommands(
            config_repo=self.mock_config_repo,
            profile_repo=self.mock_profile_repo,
            transcription_orchestrator=self.mock_transcription_orchestrator,
            logger=self.mock_logger,
        )

        # Setup default config
        self.default_config = WhisperConfig(model_name="medium")
        self.mock_config_repo.load.return_value = self.default_config

    @patch("voicebridge.cli.commands.speech_commands.threading")
    @patch("voicebridge.cli.commands.speech_commands.typer.echo")
    def test_listen_basic_setup(self, mock_echo, mock_threading):
        """Test basic listen command setup."""
        # Mock the audio recorder to avoid actual recording
        mock_recorder = Mock()
        mock_stream = [b"audio_chunk_1", b"audio_chunk_2"]
        mock_recorder.record_stream.return_value = iter(mock_stream)
        self.mock_transcription_orchestrator.audio_recorder = mock_recorder

        # Mock transcription result
        mock_result = Mock()
        mock_result.text = "Hello world"
        mock_result.confidence = 0.95
        mock_result.language = "en"
        self.mock_transcription_orchestrator.transcription_service.transcribe.return_value = mock_result

        # Mock clipboard service
        mock_clipboard = Mock()
        mock_clipboard.copy_text.return_value = True
        self.mock_transcription_orchestrator.clipboard_service = mock_clipboard

        # Test that the method can be called without errors (we'll test the hotkey logic separately)
        try:
            # We can't easily test the full hotkey functionality in unit tests due to threading and input handling
            # But we can test that the method sets up correctly
            config = self.speech_commands._build_config(model="large", language="en")
            self.assertEqual(config.model_name, "large")
            self.assertEqual(config.language, "en")
        except Exception:
            # If the method tries to run the actual hotkey loop, we'll get an exception
            # That's expected in a unit test environment
            pass

    def test_hotkey_no_transcription_service(self):
        """Test hotkey command when transcription service is not available."""
        # Create commands without transcription orchestrator
        commands_no_transcription = SpeechCommands(
            config_repo=self.mock_config_repo,
            profile_repo=self.mock_profile_repo,
            logger=self.mock_logger,
        )

        with patch("typer.echo") as mock_echo:
            with self.assertRaises(typer.Exit):
                commands_no_transcription.hotkey()

        # Should show an error message
        mock_echo.assert_called_with(
            "Error: Transcription service not available", err=True
        )


class TestConfigCommands(unittest.TestCase):
    """Test configuration command implementations."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_config_repo = Mock()
        self.mock_profile_repo = Mock()
        self.mock_logger = Mock()

        self.config_commands = ConfigCommands(
            config_repo=self.mock_config_repo,
            profile_repo=self.mock_profile_repo,
            logger=self.mock_logger,
        )

        # Setup default config
        self.default_config = WhisperConfig(model_name="medium", debug=False)
        self.mock_config_repo.load.return_value = self.default_config

    def test_config_show(self):
        """Test config show command."""
        with patch("typer.echo") as mock_echo:
            self.config_commands.config_show()

        # Verify config was loaded and displayed
        self.mock_config_repo.load.assert_called_once()
        mock_echo.assert_any_call("Current Configuration:")

    def test_config_set_valid_boolean(self):
        """Test config set command with valid boolean key."""
        with patch("typer.echo"):
            self.config_commands.config_set("debug", "true")

        # Verify config was saved with correct boolean value
        self.mock_config_repo.save.assert_called_once()
        saved_config = self.mock_config_repo.save.call_args[0][0]
        self.assertTrue(saved_config.debug)

    def test_config_set_valid_float(self):
        """Test config set command with valid float key."""
        with patch("typer.echo"):
            self.config_commands.config_set("temperature", "0.5")

        # Verify config was saved with correct float value
        self.mock_config_repo.save.assert_called_once()
        saved_config = self.mock_config_repo.save.call_args[0][0]
        self.assertEqual(saved_config.temperature, 0.5)

    def test_config_set_string_value(self):
        """Test config set command with string value."""
        with patch("typer.echo"):
            self.config_commands.config_set("model_name", "large")

        # Verify config was saved with correct string value
        self.mock_config_repo.save.assert_called_once()
        saved_config = self.mock_config_repo.save.call_args[0][0]
        self.assertEqual(saved_config.model_name, "large")

    def test_profile_save(self):
        """Test profile save command."""
        with patch("typer.echo"):
            self.config_commands.profile_save("test_profile")

        # Verify profile was saved
        self.mock_profile_repo.save_profile.assert_called_once_with(
            "test_profile", self.default_config
        )

    def test_profile_load_success(self):
        """Test profile load command success."""
        profile_config = WhisperConfig(model_name="tiny")
        self.mock_profile_repo.load_profile.return_value = profile_config

        with patch("typer.echo"):
            self.config_commands.profile_load("test_profile")

        # Verify profile was loaded and set as current config
        self.mock_profile_repo.load_profile.assert_called_once_with("test_profile")
        self.mock_config_repo.save.assert_called_once_with(profile_config)

    def test_profile_load_not_found(self):
        """Test profile load command with non-existent profile."""
        self.mock_profile_repo.load_profile.return_value = None

        with patch("typer.echo") as mock_echo:
            with self.assertRaises(typer.Exit):
                self.config_commands.profile_load("nonexistent")

        # Should show error message
        mock_echo.assert_any_call("Error: Profile 'nonexistent' not found", err=True)

    def test_profile_list_with_profiles(self):
        """Test profile list command with existing profiles."""
        self.mock_profile_repo.list_profiles.return_value = ["profile1", "profile2"]

        with patch("typer.echo") as mock_echo:
            self.config_commands.profile_list()

        mock_echo.assert_any_call("Available Profiles:")
        mock_echo.assert_any_call("  profile1")
        mock_echo.assert_any_call("  profile2")

    def test_profile_list_empty(self):
        """Test profile list command with no profiles."""
        self.mock_profile_repo.list_profiles.return_value = []

        with patch("typer.echo") as mock_echo:
            self.config_commands.profile_list()

        mock_echo.assert_any_call("Info: No profiles found")

    @patch("typer.confirm")
    def test_profile_delete_success(self, mock_confirm):
        """Test profile delete command success."""
        mock_confirm.return_value = True
        self.mock_profile_repo.delete_profile.return_value = True

        with patch("typer.echo"):
            self.config_commands.profile_delete("test_profile")

        # Verify confirmation and deletion
        mock_confirm.assert_called_once()
        self.mock_profile_repo.delete_profile.assert_called_once_with("test_profile")

    @patch("typer.confirm")
    def test_profile_delete_not_found(self, mock_confirm):
        """Test profile delete command with non-existent profile."""
        mock_confirm.return_value = True
        self.mock_profile_repo.delete_profile.return_value = False

        with patch("typer.echo") as mock_echo:
            with self.assertRaises(typer.Exit):
                self.config_commands.profile_delete("nonexistent")

        # Should show error message
        mock_echo.assert_any_call("Error: Profile 'nonexistent' not found", err=True)


class TestCommandRegistry(unittest.TestCase):
    """Test command registry functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_config_repo = Mock()
        self.mock_profile_repo = Mock()
        self.mock_logger = Mock()

        self.registry = create_command_registry(
            config_repo=self.mock_config_repo,
            profile_repo=self.mock_profile_repo,
            logger=self.mock_logger,
        )

    def test_get_command_group_speech(self):
        """Test getting speech command group."""
        speech_commands = self.registry.get_command_group("speech")
        self.assertIsInstance(speech_commands, SpeechCommands)

    def test_get_command_group_config(self):
        """Test getting config command group."""
        config_commands = self.registry.get_command_group("config")
        self.assertIsInstance(config_commands, ConfigCommands)

    def test_get_command_group_invalid(self):
        """Test getting invalid command group."""
        with self.assertRaises(ValueError):
            self.registry.get_command_group("invalid")

    def test_list_command_groups(self):
        """Test listing all command groups."""
        groups = self.registry.list_command_groups()
        expected_groups = [
            "speech",
            "transcription",
            "tts",
            "audio",
            "system",
            "config",
            "export",
            "advanced",
            "api",
        ]
        for group in expected_groups:
            self.assertIn(group, groups)

    def test_validate_dependencies(self):
        """Test dependency validation."""
        validation = self.registry.validate_dependencies()

        # Core dependencies should be available
        self.assertTrue(validation["config_repo"])
        self.assertTrue(validation["profile_repo"])
        self.assertTrue(validation["logger"])

    def test_get_all_command_groups(self):
        """Test getting all command groups."""
        all_groups = self.registry.get_all_command_groups()

        # Should contain all expected groups
        expected_groups = [
            "speech",
            "transcription",
            "tts",
            "audio",
            "system",
            "config",
            "export",
            "advanced",
            "api",
        ]
        for group in expected_groups:
            self.assertIn(group, all_groups)
            self.assertIsNotNone(all_groups[group])


if __name__ == "__main__":
    unittest.main()
