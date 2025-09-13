#!/usr/bin/env python3
"""Simplified unit tests for TTS adapters."""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from voicebridge.domain.models import TTSConfig


class TestVibeVoiceTTSAdapterInterface(unittest.TestCase):
    """Test VibeVoice TTS adapter interface (without external dependencies)."""

    def test_adapter_unavailable_initialization(self):
        """Test TTS adapter initialization when VibeVoice is not available."""
        with patch("voicebridge.adapters.vibevoice_tts.VIBEVOICE_AVAILABLE", False):
            from voicebridge.adapters.vibevoice_tts import VibeVoiceTTSAdapter

            with self.assertRaises(RuntimeError) as context:
                VibeVoiceTTSAdapter()

            self.assertIn("VibeVoice is not available", str(context.exception))

    @patch("voicebridge.adapters.vibevoice_tts.VIBEVOICE_AVAILABLE", True)
    def test_adapter_available_initialization(self):
        """Test TTS adapter when VibeVoice is available."""
        with patch.multiple(
            "voicebridge.adapters.vibevoice_tts",
            VibeVoiceForConditionalGenerationInference=Mock(),
            VibeVoiceProcessor=Mock(),
            AudioStreamer=Mock(),
        ):
            from voicebridge.adapters.vibevoice_tts import VibeVoiceTTSAdapter

            adapter = VibeVoiceTTSAdapter()

            self.assertIsNone(adapter.model)
            self.assertIsNone(adapter.processor)
            self.assertFalse(adapter.is_generating_flag)

    @patch("voicebridge.adapters.vibevoice_tts.VIBEVOICE_AVAILABLE", True)
    def test_stop_generation_interface(self):
        """Test stop generation interface."""
        with patch.multiple(
            "voicebridge.adapters.vibevoice_tts",
            VibeVoiceForConditionalGenerationInference=Mock(),
            VibeVoiceProcessor=Mock(),
            AudioStreamer=Mock(),
        ):
            from voicebridge.adapters.vibevoice_tts import VibeVoiceTTSAdapter

            adapter = VibeVoiceTTSAdapter()
            adapter.current_streamer = Mock()

            # Should not raise exception
            adapter.stop_generation()

            self.assertTrue(adapter.stop_requested)

    @patch("voicebridge.adapters.vibevoice_tts.VIBEVOICE_AVAILABLE", True)
    def test_is_generating_interface(self):
        """Test is_generating interface."""
        with patch.multiple(
            "voicebridge.adapters.vibevoice_tts",
            VibeVoiceForConditionalGenerationInference=Mock(),
            VibeVoiceProcessor=Mock(),
            AudioStreamer=Mock(),
        ):
            from voicebridge.adapters.vibevoice_tts import VibeVoiceTTSAdapter

            adapter = VibeVoiceTTSAdapter()

            # Initially not generating
            self.assertFalse(adapter.is_generating())

            # Set generating flag
            adapter.is_generating_flag = True
            self.assertTrue(adapter.is_generating())

    @patch("voicebridge.adapters.vibevoice_tts.VIBEVOICE_AVAILABLE", True)
    def test_load_voice_samples_empty_directory(self):
        """Test loading voice samples from empty directory."""
        with (
            patch.multiple(
                "voicebridge.adapters.vibevoice_tts",
                VibeVoiceForConditionalGenerationInference=Mock(),
                VibeVoiceProcessor=Mock(),
                AudioStreamer=Mock(),
            ),
            patch("voicebridge.adapters.vibevoice_tts.Path") as mock_path,
        ):
            from voicebridge.adapters.vibevoice_tts import VibeVoiceTTSAdapter

            # Mock empty directory
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = False
            mock_path.return_value = mock_path_instance

            adapter = VibeVoiceTTSAdapter()
            voices = adapter.load_voice_samples("/nonexistent/path")

            self.assertEqual(len(voices), 0)

    @patch("voicebridge.adapters.vibevoice_tts.VIBEVOICE_AVAILABLE", True)
    def test_generate_speech_empty_text(self):
        """Test generate speech with empty text."""
        with patch.multiple(
            "voicebridge.adapters.vibevoice_tts",
            VibeVoiceForConditionalGenerationInference=Mock(),
            VibeVoiceProcessor=Mock(),
            AudioStreamer=Mock(),
        ):
            from voicebridge.adapters.vibevoice_tts import VibeVoiceTTSAdapter

            adapter = VibeVoiceTTSAdapter()
            config = TTSConfig()

            with self.assertRaises(ValueError) as context:
                adapter.generate_speech("", ["test.wav"], config)

            self.assertIn("Text cannot be empty", str(context.exception))

    @patch("voicebridge.adapters.vibevoice_tts.VIBEVOICE_AVAILABLE", True)
    def test_generate_speech_streaming_empty_text(self):
        """Test streaming generate speech with empty text."""
        with patch.multiple(
            "voicebridge.adapters.vibevoice_tts",
            VibeVoiceForConditionalGenerationInference=Mock(),
            VibeVoiceProcessor=Mock(),
            AudioStreamer=Mock(),
        ):
            from voicebridge.adapters.vibevoice_tts import VibeVoiceTTSAdapter

            adapter = VibeVoiceTTSAdapter()
            config = TTSConfig()

            with self.assertRaises(ValueError) as context:
                list(adapter.generate_speech_streaming("", ["test.wav"], config))

            self.assertIn("Text cannot be empty", str(context.exception))


if __name__ == "__main__":
    unittest.main()
