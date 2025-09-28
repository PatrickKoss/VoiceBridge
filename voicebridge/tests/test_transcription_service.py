#!/usr/bin/env python3
"""Unit tests for transcription service and orchestrator."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from voicebridge.adapters.transcription import WhisperTranscriptionService
from voicebridge.domain.models import TranscriptionResult, WhisperConfig
from voicebridge.services.transcription_service import WhisperTranscriptionOrchestrator


class TestWhisperTranscriptionService(unittest.TestCase):
    """Test WhisperTranscriptionService adapter."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock whisper module
        self.mock_whisper = MagicMock()
        self.mock_model = Mock()
        self.mock_whisper.load_model.return_value = self.mock_model

        # Create service with patched whisper
        self.whisper_patcher = patch(
            "voicebridge.adapters.transcription.whisper", self.mock_whisper
        )
        self.whisper_patcher.start()

        # Mock system service
        self.mock_system_service = Mock()
        self.mock_system_service.get_memory_usage.return_value = {
            "used_mb": 500,
            "total_mb": 8000,
            "available_mb": 7500,
            "percent": 6.25,
        }
        self.mock_system_service.detect_gpu_devices.return_value = []  # Return empty list for GPU devices

        self.service = WhisperTranscriptionService(
            system_service=self.mock_system_service
        )

    def tearDown(self):
        """Clean up patches."""
        self.whisper_patcher.stop()

    def test_initialization_no_whisper(self):
        """Test initialization when whisper is not available."""
        with patch("voicebridge.adapters.transcription.whisper", None):
            with self.assertRaises(RuntimeError) as context:
                WhisperTranscriptionService()

            self.assertIn("Whisper library not available", str(context.exception))

    def test_load_model_first_time(self):
        """Test loading model for the first time."""
        config = WhisperConfig(model_name="small")

        self.service._load_model("small", config)

        self.mock_whisper.load_model.assert_called_once_with("small", device="cpu")
        self.assertEqual(self.service._current_model_name, "small")

    def test_load_model_reuse_existing(self):
        """Test reusing already loaded model."""
        config = WhisperConfig(model_name="medium")
        self.service._model = self.mock_model
        self.service._current_model_name = "medium"
        self.service._current_device = "cpu"

        self.service._load_model("medium", config)

        # Should not load again
        self.mock_whisper.load_model.assert_not_called()

    def test_transcribe_basic(self):
        """Test basic transcription."""
        config = WhisperConfig(model_name="medium")
        audio_data = b"fake_audio_data"

        # Mock whisper result
        mock_result = {
            "text": "  Hello world  ",
            "language": "en",
            "segments": [{"no_speech_prob": 0.1}],
        }
        self.mock_model.transcribe.return_value = mock_result

        result = self.service.transcribe(audio_data, config)

        # Verify result
        self.assertIsInstance(result, TranscriptionResult)
        self.assertEqual(result.text, "Hello world")  # Should be stripped
        self.assertEqual(result.language, "en")
        self.assertIsNotNone(result.confidence)

    def test_transcribe_with_options(self):
        """Test transcription with language and prompt options."""
        config = WhisperConfig(
            model_name="large", language="fr", initial_prompt="Bonjour", temperature=0.1
        )
        audio_data = b"fake_audio_data"

        mock_result = {"text": "Bonjour le monde", "language": "fr", "segments": []}
        self.mock_model.transcribe.return_value = mock_result

        self.service.transcribe(audio_data, config)

        # Verify whisper was called with correct options
        self.mock_model.transcribe.assert_called_once()
        args, kwargs = self.mock_model.transcribe.call_args

        self.assertEqual(kwargs["language"], "fr")
        self.assertEqual(kwargs["initial_prompt"], "Bonjour")
        self.assertEqual(kwargs["temperature"], 0.1)

    def test_calculate_confidence(self):
        """Test confidence calculation."""
        # Test with segments
        result_with_segments = {
            "segments": [{"no_speech_prob": 0.1}, {"no_speech_prob": 0.2}]
        }
        confidence = self.service._calculate_confidence(result_with_segments)
        expected = 1.0 - (0.1 + 0.2) / 2  # 1 - average no_speech_prob
        self.assertEqual(confidence, expected)

        # Test without segments
        result_no_segments = {"segments": []}
        confidence = self.service._calculate_confidence(result_no_segments)
        self.assertIsNone(confidence)

    def test_transcribe_stream(self):
        """Test streaming transcription."""
        config = WhisperConfig(
            model_name="medium", chunk_size=1
        )  # Small chunk size for testing

        # Mock audio stream
        audio_chunks = [b"chunk1" * 8000, b"chunk2" * 8000, b"chunk3" * 8000]

        # Mock transcription results
        mock_results = [
            {"text": "First part", "language": "en", "segments": []},
            {"text": "Second part", "language": "en", "segments": []},
        ]
        self.mock_model.transcribe.side_effect = mock_results

        # Test streaming
        results = list(self.service.transcribe_stream(iter(audio_chunks), config))

        # Should have processed chunks when buffer is large enough
        self.assertGreater(len(results), 0)
        for result in results:
            self.assertIsInstance(result, TranscriptionResult)


class TestWhisperTranscriptionOrchestrator(unittest.TestCase):
    """Test WhisperTranscriptionOrchestrator service."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_audio_recorder = Mock()
        self.mock_transcription_service = Mock()
        self.mock_clipboard_service = Mock()
        self.mock_logger = Mock()

        self.orchestrator = WhisperTranscriptionOrchestrator(
            audio_recorder=self.mock_audio_recorder,
            transcription_service=self.mock_transcription_service,
            clipboard_service=self.mock_clipboard_service,
            logger=self.mock_logger,
        )

    def test_transcribe_single_recording_success(self):
        """Test successful single recording transcription."""
        config = WhisperConfig(copy_final=True, paste_final=False)

        # Mock audio recording
        audio_data = [b"audio_chunk1", b"audio_chunk2"]
        self.mock_audio_recorder.record_stream.return_value = iter(audio_data)

        # Mock transcription
        transcription_result = TranscriptionResult(text="Hello world", confidence=0.95)
        self.mock_transcription_service.transcribe.return_value = transcription_result

        # Mock clipboard success
        self.mock_clipboard_service.copy_text.return_value = True

        result = self.orchestrator.transcribe_single_recording(config)

        # Verify result
        self.assertEqual(result, "Hello world")

        # Verify services were called
        self.mock_audio_recorder.record_stream.assert_called_once()
        self.mock_transcription_service.transcribe.assert_called_once()
        self.mock_clipboard_service.copy_text.assert_called_once_with("Hello world")

        # Verify logging
        self.mock_logger.info.assert_called()
        self.mock_logger.log_performance.assert_called()

    def test_transcribe_single_recording_no_audio(self):
        """Test transcription with no audio data."""
        config = WhisperConfig()

        # Mock empty audio recording
        self.mock_audio_recorder.record_stream.return_value = iter([])

        result = self.orchestrator.transcribe_single_recording(config)

        self.assertEqual(result, "")
        self.mock_transcription_service.transcribe.assert_not_called()

    def test_transcribe_single_recording_error(self):
        """Test error handling in single recording."""
        config = WhisperConfig()

        # Mock audio recording error
        self.mock_audio_recorder.record_stream.side_effect = Exception(
            "Recording failed"
        )

        result = self.orchestrator.transcribe_single_recording(config)

        self.assertEqual(result, "")
        self.mock_logger.error.assert_called()

    def test_transcribe_streaming(self):
        """Test streaming transcription."""
        config = WhisperConfig(copy_stream=True)

        # Mock audio stream
        audio_stream = iter([b"chunk1", b"chunk2"])
        self.mock_audio_recorder.record_stream.return_value = audio_stream

        # Mock transcription stream
        transcription_results = [
            TranscriptionResult(text="First part"),
            TranscriptionResult(text="Second part"),
        ]
        self.mock_transcription_service.transcribe_stream.return_value = iter(
            transcription_results
        )

        # Mock clipboard
        self.mock_clipboard_service.copy_text.return_value = True

        # Test streaming
        results = list(self.orchestrator.transcribe_streaming(config))

        self.assertEqual(results, ["First part", "Second part"])

        # Verify clipboard was called for each result
        self.assertEqual(self.mock_clipboard_service.copy_text.call_count, 2)

    def test_handle_output_copy_only(self):
        """Test output handling with copy only."""
        config = WhisperConfig(copy_final=True, paste_final=False)
        self.mock_clipboard_service.copy_text.return_value = True

        self.orchestrator._handle_output("Test text", config)

        self.mock_clipboard_service.copy_text.assert_called_once_with("Test text")
        self.mock_clipboard_service.type_text.assert_not_called()

    def test_handle_output_paste_only(self):
        """Test output handling with paste only."""
        config = WhisperConfig(copy_final=False, paste_final=True)
        self.mock_clipboard_service.type_text.return_value = True

        self.orchestrator._handle_output("Test text", config)

        self.mock_clipboard_service.type_text.assert_called_once_with("Test text")
        self.mock_clipboard_service.copy_text.assert_not_called()

    def test_handle_output_copy_and_paste(self):
        """Test output handling with both copy and paste."""
        config = WhisperConfig(copy_final=True, paste_final=True)
        self.mock_clipboard_service.copy_text.return_value = True
        self.mock_clipboard_service.type_text.return_value = True

        self.orchestrator._handle_output("Test text", config)

        self.mock_clipboard_service.copy_text.assert_called_once_with("Test text")
        self.mock_clipboard_service.type_text.assert_called_once_with("Test text")

    def test_handle_output_clipboard_failure(self):
        """Test output handling with clipboard failure."""
        config = WhisperConfig(copy_final=True)
        self.mock_clipboard_service.copy_text.return_value = False

        self.orchestrator._handle_output("Test text", config)

        self.mock_logger.error.assert_called_with("Failed to copy text to clipboard")

    def test_handle_output_typing_failure(self):
        """Test output handling with typing failure."""
        config = WhisperConfig(paste_final=True)
        self.mock_clipboard_service.type_text.return_value = False

        self.orchestrator._handle_output("Test text", config)

        self.mock_logger.error.assert_called_with("Failed to type text")

    def test_handle_output_exception(self):
        """Test output handling with exception."""
        config = WhisperConfig(copy_final=True)
        self.mock_clipboard_service.copy_text.side_effect = Exception("Clipboard error")

        self.orchestrator._handle_output("Test text", config)

        self.mock_logger.error.assert_called_with(
            "Output handling failed: Clipboard error"
        )


if __name__ == "__main__":
    unittest.main()
