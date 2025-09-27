#!/usr/bin/env python3
"""Unit tests for TTS services."""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from voicebridge.domain.models import (
    TTSConfig,
    TTSMode,
    TTSOutputMode,
    TTSResult,
    TTSStreamingMode,
    VoiceInfo,
)
from voicebridge.services.tts_service import TTSDaemonService, TTSOrchestrator


class TestTTSOrchestrator(unittest.TestCase):
    """Test TTS orchestrator service."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_tts_service = Mock()
        self.mock_text_input_service = Mock()
        self.mock_audio_playback_service = Mock()
        self.mock_logger = Mock()

        self.orchestrator = TTSOrchestrator(
            tts_service=self.mock_tts_service,
            text_input_service=self.mock_text_input_service,
            audio_playback_service=self.mock_audio_playback_service,
            logger=self.mock_logger,
        )

        # Default voice samples
        self.voice_samples = {
            "en-Alice_woman": VoiceInfo(
                name="en-Alice_woman",
                file_path="/voices/en-Alice_woman.wav",
                display_name="Alice",
                language="en",
                gender="woman",
            ),
            "en-Bob_man": VoiceInfo(
                name="en-Bob_man",
                file_path="/voices/en-Bob_man.wav",
                display_name="Bob",
                language="en",
                gender="man",
            ),
        }

    def test_initialization(self):
        """Test orchestrator initialization."""
        self.assertEqual(self.orchestrator.tts_service, self.mock_tts_service)
        self.assertEqual(
            self.orchestrator.text_input_service, self.mock_text_input_service
        )
        self.assertEqual(
            self.orchestrator.audio_playback_service, self.mock_audio_playback_service
        )
        self.assertEqual(self.orchestrator.logger, self.mock_logger)

    def test_load_voice_samples(self):
        """Test loading voice samples."""
        self.mock_tts_service.load_voice_samples.return_value = self.voice_samples

        config = TTSConfig(voice_samples_dir="/test/voices")
        voices = self.orchestrator.load_voice_samples(config)

        self.assertEqual(len(voices), 2)
        self.assertIn("en-Alice_woman", voices)
        self.mock_tts_service.load_voice_samples.assert_called_once_with("/test/voices")

    def test_get_voice_sample_paths_default_voice(self):
        """Test getting voice sample paths with default voice."""
        config = TTSConfig(default_voice="en-Alice_woman")

        paths = self.orchestrator._get_voice_sample_paths(
            self.voice_samples, config, None
        )

        self.assertEqual(paths, ["/voices/en-Alice_woman.wav"])

    def test_get_voice_sample_paths_specified_voice(self):
        """Test getting voice sample paths with specified voice."""
        config = TTSConfig()

        paths = self.orchestrator._get_voice_sample_paths(
            self.voice_samples, config, "en-Bob_man"
        )

        self.assertEqual(paths, ["/voices/en-Bob_man.wav"])

    def test_get_voice_sample_paths_voice_not_found(self):
        """Test getting voice sample paths when voice not found."""
        config = TTSConfig()

        with self.assertRaises(ValueError) as context:
            self.orchestrator._get_voice_sample_paths(
                self.voice_samples, config, "nonexistent_voice"
            )

        self.assertIn("Voice 'nonexistent_voice' not found", str(context.exception))

    def test_get_voice_sample_paths_no_voices(self):
        """Test getting voice sample paths when no voices available."""
        config = TTSConfig()

        with self.assertRaises(RuntimeError) as context:
            self.orchestrator._get_voice_sample_paths({}, config, None)

        self.assertIn("No voice samples available", str(context.exception))

    def test_generate_speech_non_streaming_success(self):
        """Test successful non-streaming speech generation."""
        # Mock TTS result
        audio_data = b"fake_audio"
        tts_result = TTSResult(
            audio_data=audio_data,
            sample_rate=24000,
            text="Hello world",
            generation_time=1.5,
            streaming_mode=False,
        )

        self.mock_tts_service.load_voice_samples.return_value = self.voice_samples
        self.mock_tts_service.generate_speech.return_value = tts_result

        config = TTSConfig(streaming_mode=TTSStreamingMode.NON_STREAMING)
        result = self.orchestrator.generate_speech("Hello world", config)

        self.assertEqual(result, tts_result)
        self.mock_tts_service.generate_speech.assert_called_once()

    def test_generate_speech_streaming_success(self):
        """Test successful streaming speech generation."""
        # Mock streaming generator
        audio_chunks = [b"chunk1", b"chunk2", b"chunk3"]

        self.mock_tts_service.load_voice_samples.return_value = self.voice_samples
        self.mock_tts_service.generate_speech_streaming.return_value = iter(
            audio_chunks
        )

        config = TTSConfig(streaming_mode=TTSStreamingMode.STREAMING)
        result_generator = self.orchestrator.generate_speech("Hello streaming", config)

        # Collect all chunks
        collected_chunks = list(result_generator)

        self.assertEqual(collected_chunks, audio_chunks)
        self.mock_tts_service.generate_speech_streaming.assert_called_once()

    def test_generate_and_play_speech_play_mode(self):
        """Test generate and play speech with play mode."""
        audio_data = b"fake_audio"
        tts_result = TTSResult(
            audio_data=audio_data,
            sample_rate=24000,
            text="Hello world",
            generation_time=1.5,
        )

        self.mock_tts_service.load_voice_samples.return_value = self.voice_samples
        self.mock_tts_service.generate_speech.return_value = tts_result

        config = TTSConfig(output_mode=TTSOutputMode.PLAY_AUDIO)
        result = self.orchestrator.generate_and_play_speech("Hello world", config)

        self.assertEqual(result, tts_result)
        self.mock_audio_playback_service.play_audio_data.assert_called_once_with(
            audio_data, 24000
        )

    @patch("pathlib.Path")
    @patch("builtins.open", create=True)
    def test_generate_and_play_speech_save_mode(self, mock_open, mock_path):
        """Test generate and play speech with save mode."""
        audio_data = b"fake_audio"
        tts_result = TTSResult(
            audio_data=audio_data,
            sample_rate=24000,
            text="Hello world",
            generation_time=1.5,
        )

        self.mock_tts_service.load_voice_samples.return_value = self.voice_samples
        self.mock_tts_service.generate_speech.return_value = tts_result

        config = TTSConfig(output_mode=TTSOutputMode.SAVE_FILE)
        result = self.orchestrator.generate_and_play_speech(
            "Hello world", config, output_file="test.wav"
        )

        self.assertEqual(result, tts_result)
        # Should not call play_audio_data in save mode
        self.mock_audio_playback_service.play_audio_data.assert_not_called()

    def test_generate_and_play_speech_both_mode(self):
        """Test generate and play speech with both mode."""
        audio_data = b"fake_audio"
        tts_result = TTSResult(
            audio_data=audio_data,
            sample_rate=24000,
            text="Hello world",
            generation_time=1.5,
        )

        self.mock_tts_service.load_voice_samples.return_value = self.voice_samples
        self.mock_tts_service.generate_speech.return_value = tts_result

        with patch("pathlib.Path"), patch("builtins.open", create=True):
            config = TTSConfig(output_mode=TTSOutputMode.BOTH)
            result = self.orchestrator.generate_and_play_speech(
                "Hello world", config, output_file="test.wav"
            )

            self.assertEqual(result, tts_result)
            self.mock_audio_playback_service.play_audio_data.assert_called_once_with(
                audio_data, 24000
            )

    def test_process_clipboard_text(self):
        """Test processing clipboard text."""
        self.mock_text_input_service.get_clipboard_text.return_value = (
            "Hello from clipboard"
        )

        config = TTSConfig()
        self.mock_tts_service.load_voice_samples.return_value = self.voice_samples
        self.mock_tts_service.generate_speech.return_value = TTSResult(
            audio_data=b"audio",
            sample_rate=24000,
            text="Hello from clipboard",
            generation_time=1.0,
        )

        result = self.orchestrator.process_clipboard_text(config)

        self.assertIsNotNone(result)
        self.mock_text_input_service.get_clipboard_text.assert_called_once()

    def test_process_clipboard_text_empty(self):
        """Test processing empty clipboard text."""
        self.mock_text_input_service.get_clipboard_text.return_value = ""

        config = TTSConfig()
        result = self.orchestrator.process_clipboard_text(config)

        self.assertIsNone(result)
        self.mock_tts_service.generate_speech.assert_not_called()

    def test_process_selected_text(self):
        """Test processing selected text."""
        self.mock_text_input_service.get_selected_text.return_value = "Selected text"

        config = TTSConfig()
        self.mock_tts_service.load_voice_samples.return_value = self.voice_samples
        self.mock_tts_service.generate_speech.return_value = TTSResult(
            audio_data=b"audio",
            sample_rate=24000,
            text="Selected text",
            generation_time=1.0,
        )

        result = self.orchestrator.process_selected_text(config)

        self.assertIsNotNone(result)
        self.mock_text_input_service.get_selected_text.assert_called_once()

    def test_stop_generation(self):
        """Test stopping generation."""
        self.orchestrator.stop_generation()

        self.mock_tts_service.stop_generation.assert_called_once()
        self.mock_audio_playback_service.stop_playback.assert_called_once()

    def test_is_generating(self):
        """Test checking if generating."""
        self.mock_tts_service.is_generating.return_value = True

        is_generating = self.orchestrator.is_generating()

        self.assertTrue(is_generating)
        self.mock_tts_service.is_generating.assert_called_once()


class TestTTSDaemonService(unittest.TestCase):
    """Test TTS daemon service."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_orchestrator = Mock()
        self.mock_logger = Mock()

        self.daemon = TTSDaemonService(
            orchestrator=self.mock_orchestrator,
            logger=self.mock_logger,
        )

        # Reset daemon state for clean tests
        self.daemon.is_running = False
        self.daemon.current_config = None
        self.daemon.hotkey_listener = None
        self.daemon.clipboard_monitor_thread = None
        self.daemon.is_monitoring_clipboard = False

        # Mock file system operations to prevent actual file creation
        self.daemon._cleanup_daemon_files = Mock()
        self.daemon._write_pid_file = Mock()
        self.daemon._write_status_file = Mock()
        self.daemon._setup_hotkeys = Mock()
        self.daemon._start_clipboard_monitoring = Mock()
        self.daemon._cleanup_hotkeys = Mock()
        self.daemon.stop_monitoring = Mock()

        # Mock orchestrator methods
        self.mock_orchestrator.start_tts_mode = Mock()
        self.mock_orchestrator.stop_tts = Mock()

        # Mock is_daemon_running to return False by default for clean tests
        self.daemon_running_patcher = patch.object(
            self.daemon, "is_daemon_running", return_value=False
        )
        self.daemon_running_patcher.start()

    def tearDown(self):
        """Clean up test fixtures."""
        self.daemon_running_patcher.stop()

    def test_initialization(self):
        """Test daemon initialization."""
        self.assertEqual(self.daemon.orchestrator, self.mock_orchestrator)
        self.assertEqual(self.daemon.logger, self.mock_logger)
        self.assertIsNone(self.daemon.hotkey_listener)
        self.assertIsNone(self.daemon.clipboard_monitor_thread)
        self.assertFalse(self.daemon.is_running)

    def test_start_daemon_clipboard_mode(self):
        """Test starting daemon in clipboard mode."""
        config = TTSConfig(tts_mode=TTSMode.CLIPBOARD)

        self.daemon.start_daemon(config)

        self.assertTrue(self.daemon.is_running)
        self.assertEqual(self.daemon.current_config, config)
        self.daemon._setup_hotkeys.assert_called_once_with(config)
        self.daemon._start_clipboard_monitoring.assert_called_once_with(config)
        self.mock_orchestrator.start_tts_mode.assert_called_once_with(config)

    def test_start_daemon_mouse_mode(self):
        """Test starting daemon in mouse mode."""
        config = TTSConfig(tts_mode=TTSMode.MOUSE)

        self.daemon.start_daemon(config)

        self.assertTrue(self.daemon.is_running)
        self.assertEqual(self.daemon.current_config, config)
        self.daemon._setup_hotkeys.assert_called_once_with(config)
        # Should not call _start_clipboard_monitoring for mouse mode
        self.daemon._start_clipboard_monitoring.assert_not_called()
        self.mock_orchestrator.start_tts_mode.assert_called_once_with(config)

    def test_start_daemon_already_running(self):
        """Test starting daemon when already running."""
        config = TTSConfig()

        # Mock is_daemon_running to return True for this test
        with patch.object(self.daemon, "is_daemon_running", return_value=True):
            with self.assertRaises(RuntimeError) as context:
                self.daemon.start_daemon(config)

            self.assertIn("TTS daemon is already running", str(context.exception))

    def test_stop_daemon_success(self):
        """Test stopping daemon successfully."""
        # Mock running daemon
        self.daemon.is_running = True
        self.daemon.hotkey_listener = Mock()
        self.daemon.clipboard_monitor_thread = Mock()

        self.daemon.stop_daemon()

        self.assertFalse(self.daemon.is_running)
        self.assertIsNone(self.daemon.current_config)
        # Verify all cleanup methods were called
        self.daemon.stop_monitoring.assert_called_once()
        self.daemon._cleanup_hotkeys.assert_called_once()
        self.daemon._cleanup_daemon_files.assert_called_once()
        self.mock_orchestrator.stop_tts.assert_called_once()

    def test_stop_daemon_not_running(self):
        """Test stopping daemon when not running."""
        # The stop_daemon method doesn't raise an exception when not running
        # It just performs cleanup operations
        self.daemon.stop_daemon()

        # Verify cleanup was called
        self.daemon._cleanup_daemon_files.assert_called_once()
        self.assertFalse(self.daemon.is_running)
        self.assertIsNone(self.daemon.current_config)

    def test_get_status_running(self):
        """Test getting daemon status when running."""
        self.daemon.current_config = TTSConfig(tts_mode=TTSMode.CLIPBOARD)

        # Mock is_daemon_running to return True and mock file reading
        with (
            patch.object(self.daemon, "is_daemon_running", return_value=True),
            patch("builtins.open", create=True),
            patch("json.load", return_value={"mode": "clipboard"}),
        ):
            status = self.daemon.get_status()

            self.assertEqual(status["status"], "running")
            self.assertEqual(status["mode"], "clipboard")

    def test_get_status_stopped(self):
        """Test getting daemon status when stopped."""
        status = self.daemon.get_status()

        self.assertEqual(status["status"], "stopped")

    @patch("time.sleep")
    def test_clipboard_monitor_loop(self, mock_sleep):
        """Test clipboard monitoring loop."""
        config = TTSConfig()

        # Mock clipboard text changes
        _clipboard_values = ["", "Hello", "Hello", "World"]  # Simulate text changes
        self.mock_orchestrator.process_clipboard_text.side_effect = [
            None,
            Mock(),
            None,
            Mock(),
        ]

        # Simulate monitoring for a few iterations then stop
        def side_effect(interval):
            if mock_sleep.call_count >= 3:
                self.daemon.is_monitoring_clipboard = False

        mock_sleep.side_effect = side_effect
        self.daemon.is_monitoring_clipboard = True

        self.daemon._clipboard_monitor_loop(config)

        self.assertGreater(mock_sleep.call_count, 0)

    def test_handle_generate_hotkey_clipboard_mode(self):
        """Test handling generate hotkey in clipboard mode."""
        self.daemon.current_config = TTSConfig(tts_mode=TTSMode.CLIPBOARD)
        self.mock_orchestrator.text_input_service.get_clipboard_text.return_value = (
            "test text"
        )
        self.mock_orchestrator.generate_tts_from_text.return_value = True

        self.daemon._handle_generate_hotkey()

        self.mock_orchestrator.text_input_service.get_clipboard_text.assert_called_once()
        self.mock_orchestrator.generate_tts_from_text.assert_called_once_with(
            "test text", self.daemon.current_config
        )

    def test_handle_generate_hotkey_mouse_mode(self):
        """Test handling generate hotkey in mouse mode."""
        self.daemon.current_config = TTSConfig(tts_mode=TTSMode.MOUSE)
        self.mock_orchestrator.text_input_service.get_selected_text.return_value = (
            "selected text"
        )
        self.mock_orchestrator.generate_tts_from_text.return_value = True

        self.daemon._handle_generate_hotkey()

        self.mock_orchestrator.text_input_service.get_selected_text.assert_called_once()
        self.mock_orchestrator.generate_tts_from_text.assert_called_once_with(
            "selected text", self.daemon.current_config
        )

    def test_handle_stop_hotkey(self):
        """Test handling stop hotkey."""
        self.daemon._handle_stop_hotkey()

        self.mock_orchestrator.stop_generation.assert_called_once()

    def test_basic_daemon_operations(self):
        """Test basic daemon operations."""
        # Test daemon initialization state
        self.assertFalse(self.daemon.is_running)
        self.assertIsNone(self.daemon.current_config)

        # Test status when not running
        status = self.daemon.get_status()
        self.assertEqual(status["status"], "stopped")

        # Test stop when not running (should not crash)
        self.daemon.stop_daemon()
        self.assertFalse(self.daemon.is_running)


if __name__ == "__main__":
    unittest.main()
