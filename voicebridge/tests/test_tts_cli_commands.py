#!/usr/bin/env python3
"""Unit tests for TTS CLI commands."""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import typer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.commands import CLICommands
from domain.models import TTSConfig, TTSResult, VoiceInfo, WhisperConfig


class TestTTSCLICommands(unittest.TestCase):
    """Test TTS CLI command implementations."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock all required services
        self.mock_config_repo = Mock()
        self.mock_profile_repo = Mock()
        self.mock_daemon_service = Mock()
        self.mock_transcription_orchestrator = Mock()
        self.mock_system_service = Mock()
        self.mock_logger = Mock()
        self.mock_session_service = Mock()
        self.mock_performance_service = Mock()
        self.mock_resume_service = Mock()
        self.mock_export_service = Mock()
        self.mock_timestamp_service = Mock()
        self.mock_confidence_analyzer = Mock()
        self.mock_audio_format_service = Mock()
        self.mock_audio_preprocessing_service = Mock()
        self.mock_audio_splitting_service = Mock()
        self.mock_batch_processing_service = Mock()

        # Mock TTS services
        self.mock_tts_orchestrator = Mock()
        self.mock_tts_daemon_service = Mock()

        # Create commands instance with TTS services
        self.commands = CLICommands(
            config_repo=self.mock_config_repo,
            profile_repo=self.mock_profile_repo,
            daemon_service=self.mock_daemon_service,
            transcription_orchestrator=self.mock_transcription_orchestrator,
            system_service=self.mock_system_service,
            logger=self.mock_logger,
            session_service=self.mock_session_service,
            performance_service=self.mock_performance_service,
            resume_service=self.mock_resume_service,
            export_service=self.mock_export_service,
            timestamp_service=self.mock_timestamp_service,
            confidence_analyzer=self.mock_confidence_analyzer,
            audio_format_service=self.mock_audio_format_service,
            audio_preprocessing_service=self.mock_audio_preprocessing_service,
            audio_splitting_service=self.mock_audio_splitting_service,
            batch_processing_service=self.mock_batch_processing_service,
            tts_orchestrator=self.mock_tts_orchestrator,
            tts_daemon_service=self.mock_tts_daemon_service,
        )

        # Setup default config with TTS config
        self.default_tts_config = TTSConfig(default_voice="en-Alice_woman")
        self.default_config = WhisperConfig(tts_config=self.default_tts_config)
        self.mock_config_repo.load.return_value = self.default_config

    def test_tts_generate_basic(self):
        """Test basic TTS generate command."""
        # Mock successful TTS result
        tts_result = TTSResult(
            audio_data=b"fake_audio",
            sample_rate=24000,
            text="Hello world",
            generation_time=1.5,
        )
        self.mock_tts_orchestrator.generate_and_play_speech.return_value = tts_result

        with patch("typer.echo") as mock_echo:
            self.commands.tts_generate("Hello world")

        # Verify TTS generation was called
        self.mock_tts_orchestrator.generate_and_play_speech.assert_called_once()

        # Verify output
        mock_echo.assert_any_call("Generating TTS for: Hello world")

    def test_tts_generate_with_voice(self):
        """Test TTS generate command with specific voice."""
        tts_result = TTSResult(
            audio_data=b"fake_audio",
            sample_rate=24000,
            text="Hello world",
            generation_time=1.5,
        )
        self.mock_tts_orchestrator.generate_and_play_speech.return_value = tts_result

        with patch("typer.echo"):
            self.commands.tts_generate("Hello world", voice="en-Bob_man")

        # Verify correct voice parameter was passed
        call_args = self.mock_tts_orchestrator.generate_and_play_speech.call_args
        # The third argument (index 2) should be the voice
        self.assertEqual(call_args[0][2], "en-Bob_man")

    def test_tts_generate_streaming_mode(self):
        """Test TTS generate command with streaming enabled."""
        tts_result = TTSResult(
            audio_data=b"fake_audio",
            sample_rate=24000,
            text="Hello world",
            generation_time=1.5,
            streaming_mode=True,
        )
        self.mock_tts_orchestrator.generate_and_play_speech.return_value = tts_result

        with patch("typer.echo"):
            self.commands.tts_generate("Hello world", streaming=True)

        # Verify streaming was enabled by checking config repo load was called
        self.mock_config_repo.load.assert_called()

    def test_tts_generate_with_output_file(self):
        """Test TTS generate command with output file."""
        tts_result = TTSResult(
            audio_data=b"fake_audio",
            sample_rate=24000,
            text="Hello world",
            generation_time=1.5,
        )
        self.mock_tts_orchestrator.generate_and_play_speech.return_value = tts_result

        with patch("typer.echo"):
            self.commands.tts_generate("Hello world", output_file="test.wav")

        # Verify output file was passed as fourth argument (index 3)
        call_args = self.mock_tts_orchestrator.generate_and_play_speech.call_args
        self.assertEqual(call_args[0][3], "test.wav")

    def test_tts_generate_no_orchestrator(self):
        """Test TTS generate command when TTS is not available."""
        # Create commands without TTS orchestrator
        commands_no_tts = CLICommands(
            config_repo=self.mock_config_repo,
            profile_repo=self.mock_profile_repo,
            daemon_service=self.mock_daemon_service,
            transcription_orchestrator=self.mock_transcription_orchestrator,
            system_service=self.mock_system_service,
            logger=self.mock_logger,
            # TTS services are None
        )

        with patch("typer.echo") as mock_echo:
            with self.assertRaises(typer.Exit):
                commands_no_tts.tts_generate("Hello world")

        mock_echo.assert_called_with("TTS orchestrator not available", err=True)

    def test_tts_generate_exception_handling(self):
        """Test TTS generate command with exception."""
        self.mock_tts_orchestrator.generate_and_play_speech.side_effect = Exception(
            "TTS generation failed"
        )

        with patch("typer.echo"):
            with self.assertRaises(Exception) as context:
                self.commands.tts_generate("Hello world")

        self.assertEqual(str(context.exception), "TTS generation failed")

    @patch("time.sleep")
    def test_tts_listen_clipboard(self, mock_sleep):
        """Test TTS clipboard listening command."""
        # Mock keyboard interrupt to stop the loop
        mock_sleep.side_effect = [None, KeyboardInterrupt()]

        with patch("typer.echo") as mock_echo:
            self.commands.tts_listen_clipboard()

        # Verify TTS mode was started
        self.mock_tts_orchestrator.start_tts_mode.assert_called_once()
        mock_echo.assert_any_call(
            "Copy text to clipboard to generate TTS. Press Ctrl+C to stop."
        )
        mock_echo.assert_any_call("\nStopping clipboard monitoring...")

    @patch("time.sleep")
    def test_tts_listen_clipboard_with_options(self, mock_sleep):
        """Test TTS clipboard listening with custom options."""
        mock_sleep.side_effect = KeyboardInterrupt()

        with patch("typer.echo"):
            self.commands.tts_listen_clipboard(
                voice="en-Bob_man",
                streaming=True,
                auto_play=False,
                output_file="output.wav",
            )

        # Verify TTS mode was started
        self.mock_tts_orchestrator.start_tts_mode.assert_called_once()

    @patch("time.sleep")
    def test_tts_listen_selection(self, mock_sleep):
        """Test TTS selection listening command."""
        mock_sleep.side_effect = KeyboardInterrupt()

        with patch("typer.echo") as mock_echo:
            self.commands.tts_listen_selection()

        # Verify TTS mode was started
        self.mock_tts_orchestrator.start_tts_mode.assert_called_once()
        mock_echo.assert_any_call(
            "Select text and press hotkey to generate TTS. Press Ctrl+C to stop."
        )

    def test_tts_daemon_start_clipboard_mode(self):
        """Test starting TTS daemon in clipboard mode."""
        with patch("typer.echo") as mock_echo:
            self.commands.tts_daemon_start(mode="clipboard")

        # Verify daemon was started
        self.mock_tts_daemon_service.start_daemon.assert_called_once()
        call_args = self.mock_tts_daemon_service.start_daemon.call_args
        config = call_args[0][0]
        self.assertEqual(config.tts_mode.value, "clipboard")

        mock_echo.assert_any_call("TTS daemon started successfully")

    def test_tts_daemon_start_selection_mode(self):
        """Test starting TTS daemon in selection mode."""
        with patch("typer.echo"):
            self.commands.tts_daemon_start(mode="selection")

        # Verify daemon was started with correct mode
        call_args = self.mock_tts_daemon_service.start_daemon.call_args
        config = call_args[0][0]
        self.assertEqual(config.tts_mode.value, "mouse")  # selection maps to mouse

    def test_tts_daemon_start_invalid_mode(self):
        """Test starting TTS daemon with invalid mode."""
        with patch("typer.echo") as mock_echo:
            with self.assertRaises(typer.Exit):
                self.commands.tts_daemon_start(mode="invalid")

        mock_echo.assert_called_with(
            "Invalid mode: invalid. Use 'clipboard' or 'selection'", err=True
        )

    def test_tts_daemon_start_already_running(self):
        """Test starting TTS daemon when already running."""
        self.mock_tts_daemon_service.start_daemon.side_effect = RuntimeError(
            "TTS daemon is already running"
        )

        with patch("typer.echo") as mock_echo:
            with self.assertRaises(typer.Exit):
                self.commands.tts_daemon_start()

        mock_echo.assert_called_with(
            "Failed to start TTS daemon: TTS daemon is already running", err=True
        )

    def test_tts_daemon_stop_success(self):
        """Test stopping TTS daemon successfully."""
        with patch("typer.echo") as mock_echo:
            self.commands.tts_daemon_stop()

        self.mock_tts_daemon_service.stop_daemon.assert_called_once()
        mock_echo.assert_called_with("TTS daemon stopped")

    def test_tts_daemon_stop_not_running(self):
        """Test stopping TTS daemon when not running."""
        self.mock_tts_daemon_service.stop_daemon.side_effect = Exception(
            "Error stopping daemon: TTS daemon is not running"
        )

        with patch("typer.echo") as mock_echo:
            self.commands.tts_daemon_stop()

        mock_echo.assert_called_with(
            "Error stopping daemon: Error stopping daemon: TTS daemon is not running",
            err=True,
        )

    def test_tts_daemon_status_running(self):
        """Test TTS daemon status when running."""
        self.mock_tts_daemon_service.is_running.return_value = True

        with patch("typer.echo") as mock_echo:
            self.commands.tts_daemon_status()

        mock_echo.assert_called_with("TTS daemon status: running")

    def test_tts_daemon_status_stopped(self):
        """Test TTS daemon status when stopped."""
        self.mock_tts_daemon_service.is_running.return_value = False

        with patch("typer.echo") as mock_echo:
            self.commands.tts_daemon_status()

        mock_echo.assert_called_with("TTS daemon status: stopped")

    def test_tts_voices_list_success(self):
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
            self.commands.tts_voices()

        # Verify voices were displayed
        mock_echo.assert_any_call("Available voices (2 total):")
        # Should show both voices
        echo_calls = [str(call) for call in mock_echo.call_args_list]
        self.assertTrue(any("en-Alice_woman" in call for call in echo_calls))
        self.assertTrue(any("en-Bob_man" in call for call in echo_calls))

    def test_tts_voices_list_empty(self):
        """Test listing TTS voices when none available."""
        self.mock_tts_orchestrator.list_available_voices.return_value = {}

        with patch("typer.echo") as mock_echo:
            self.commands.tts_voices()

        mock_echo.assert_called_with("No voices found")

    def test_tts_config_show(self):
        """Test showing TTS configuration."""
        with patch("typer.echo") as mock_echo:
            self.commands.tts_config_show()

        # Should display configuration
        echo_calls = [str(call) for call in mock_echo.call_args_list]
        self.assertTrue(any("TTS Configuration" in call for call in echo_calls))

    def test_tts_config_set_voice(self):
        """Test setting TTS configuration - default voice."""
        with patch("typer.echo") as mock_echo:
            self.commands.tts_config_set(default_voice="en-Bob_man")

        # Verify config was updated and saved
        self.assertEqual(self.default_config.tts_config.default_voice, "en-Bob_man")
        self.mock_config_repo.save.assert_called_once_with(self.default_config)
        mock_echo.assert_called_with("Configuration saved")

    def test_tts_config_set_model_path(self):
        """Test setting TTS configuration - model path."""
        with patch("typer.echo"):
            self.commands.tts_config_set(model_path="custom/model/path")

        self.assertEqual(self.default_config.tts_config.model_path, "custom/model/path")
        self.mock_config_repo.save.assert_called_once()

    def test_tts_config_set_multiple_options(self):
        """Test setting multiple TTS configuration options."""
        with patch("typer.echo"):
            self.commands.tts_config_set(
                default_voice="en-Bob_man", cfg_scale=1.5, inference_steps=8
            )

        # Verify all options were set
        config = self.default_config.tts_config
        self.assertEqual(config.default_voice, "en-Bob_man")
        self.assertEqual(config.cfg_scale, 1.5)
        self.assertEqual(config.inference_steps, 8)


if __name__ == "__main__":
    unittest.main()
