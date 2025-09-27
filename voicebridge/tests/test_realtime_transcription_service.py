"""Tests for RealtimeTranscriptionService."""

import queue
from unittest.mock import Mock, patch

import pytest

from voicebridge.domain.models import TranscriptionResult, WhisperConfig
from voicebridge.services.realtime_transcription import RealtimeTranscriptionService


class TestRealtimeTranscriptionService:
    """Test cases for RealtimeTranscriptionService."""

    @pytest.fixture
    def mock_audio_recorder(self):
        """Mock AudioRecorder."""
        mock = Mock()
        mock.record_stream.return_value = iter([b"audio_chunk_1", b"audio_chunk_2"])
        return mock

    @pytest.fixture
    def mock_transcription_service(self):
        """Mock TranscriptionService."""
        mock = Mock()
        mock.transcribe.return_value = TranscriptionResult(
            text="test transcription", confidence=0.95
        )
        return mock

    @pytest.fixture
    def mock_logger(self):
        """Mock Logger."""
        return Mock()

    @pytest.fixture
    def config(self):
        """Sample WhisperConfig."""
        return WhisperConfig(model_name="base")

    @pytest.fixture
    def service(self, mock_audio_recorder, mock_transcription_service, mock_logger):
        """RealtimeTranscriptionService instance."""
        return RealtimeTranscriptionService(
            audio_recorder=mock_audio_recorder,
            transcription_service=mock_transcription_service,
            logger=mock_logger,
        )

    def test_init(
        self, service, mock_audio_recorder, mock_transcription_service, mock_logger
    ):
        """Test service initialization."""
        assert service.audio_recorder == mock_audio_recorder
        assert service.transcription_service == mock_transcription_service
        assert service.logger == mock_logger
        assert service.is_running is False

    def test_stop(self, service):
        """Test stopping the service."""
        service.is_running = True
        service.stop()
        assert service.is_running is False

    def test_has_voice_activity_with_valid_audio(self, service):
        """Test voice activity detection with valid audio."""
        # Create mock audio data (16-bit samples)
        import struct

        samples = [1000, 2000, 1500, 3000]  # High energy samples
        audio_chunk = struct.pack(f"<{len(samples)}h", *samples)

        result = service._has_voice_activity(audio_chunk, 0.01)
        assert result is True

    def test_has_voice_activity_with_low_energy(self, service):
        """Test voice activity detection with low energy audio."""
        import struct

        samples = [10, 20, 15, 30]  # Low energy samples
        audio_chunk = struct.pack(f"<{len(samples)}h", *samples)

        result = service._has_voice_activity(audio_chunk, 0.5)
        assert result is False

    def test_has_voice_activity_with_empty_audio(self, service):
        """Test voice activity detection with empty audio."""
        result = service._has_voice_activity(b"", 0.01)
        assert result is False

    def test_has_voice_activity_with_invalid_audio(self, service):
        """Test voice activity detection with invalid audio data."""
        # Invalid audio data should return True (assume voice activity)
        result = service._has_voice_activity(b"invalid", 0.01)
        assert result is True

    def test_format_output_live_format(self, service):
        """Test output formatting for live format."""
        result = TranscriptionResult(text="hello world", confidence=0.9)

        output = service._format_output(result, "live", 1, "")

        assert output["text"] == "hello world"
        assert output["confidence"] == 0.9
        assert output["type"] == "live"
        assert "timestamp" in output

    def test_format_output_segments_format(self, service):
        """Test output formatting for segments format."""
        result = TranscriptionResult(text="new text", confidence=0.8)

        output = service._format_output(result, "segments", 2, "old text")

        assert output["segment_id"] == 2
        assert output["text"] == "new text"
        assert output["confidence"] == 0.8
        assert output["type"] == "segment"
        assert "timestamp" in output

    def test_format_output_segments_format_same_text(self, service):
        """Test output formatting for segments format with same text."""
        result = TranscriptionResult(text="same text", confidence=0.8)

        output = service._format_output(result, "segments", 2, "same text")

        assert output is None

    def test_format_output_complete_format(self, service):
        """Test output formatting for complete format."""
        result = TranscriptionResult(text="complete text", confidence=0.95)

        output = service._format_output(result, "complete", 5, "")

        assert output["complete_text"] == "complete text"
        assert output["confidence"] == 0.95
        assert output["type"] == "complete"
        assert output["segments_processed"] == 5
        assert "timestamp" in output

    def test_format_output_unknown_format(self, service):
        """Test output formatting for unknown format."""
        result = TranscriptionResult(text="test", confidence=0.9)

        output = service._format_output(result, "unknown", 1, "")

        assert output is None

    @patch("subprocess.Popen")
    def test_transcribe_file_streaming_success(self, mock_popen, service, config):
        """Test successful file streaming transcription."""
        # Mock subprocess
        mock_process = Mock()
        mock_process.stdout.read.side_effect = [
            b"audio_chunk_1" * 1000,  # First chunk
            b"audio_chunk_2" * 1000,  # Second chunk
            b"",  # End of stream
        ]
        mock_popen.return_value = mock_process

        # Mock transcription results
        service.transcription_service.transcribe.side_effect = [
            TranscriptionResult(text="first chunk", confidence=0.9),
            TranscriptionResult(text="second chunk", confidence=0.85),
        ]

        # Create a temporary file
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
            results = list(service.transcribe_file_streaming(tmp_file.name, config))

        assert len(results) == 2
        assert results[0]["text"] == "first chunk"
        assert results[0]["chunk_id"] == 1
        assert results[1]["text"] == "second chunk"
        assert results[1]["chunk_id"] == 2

    @patch("subprocess.Popen")
    def test_transcribe_file_streaming_file_not_found(
        self, mock_popen, service, config
    ):
        """Test file streaming transcription with non-existent file."""
        results = list(
            service.transcribe_file_streaming("/nonexistent/file.wav", config)
        )
        assert results == []
        service.logger.error.assert_called()

    @patch("subprocess.Popen")
    def test_transcribe_file_streaming_with_transcription_error(
        self, mock_popen, service, config
    ):
        """Test file streaming transcription with transcription errors."""
        # Mock subprocess
        mock_process = Mock()
        mock_process.stdout.read.side_effect = [
            b"audio_chunk" * 1000,
            b"",  # End of stream
        ]
        mock_popen.return_value = mock_process

        # Mock transcription to raise error
        service.transcription_service.transcribe.side_effect = Exception(
            "Transcription failed"
        )

        # Create a temporary file
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
            results = list(service.transcribe_file_streaming(tmp_file.name, config))

        assert results == []
        service.logger.error.assert_called()

    def test_audio_capture_worker_success(self, service):
        """Test successful audio capture worker."""
        audio_buffer = queue.Queue()
        service.is_running = True

        # Mock the audio recorder to return some chunks then stop
        def mock_record_stream(sample_rate):
            yield b"chunk1"
            yield b"chunk2"
            service.is_running = False  # Stop after 2 chunks

        service.audio_recorder.record_stream = mock_record_stream

        # Run the worker
        service._audio_capture_worker(audio_buffer, 16000)

        # Check results
        assert audio_buffer.get() == b"chunk1"
        assert audio_buffer.get() == b"chunk2"
        assert audio_buffer.get() is None  # End signal

    def test_audio_capture_worker_with_exception(self, service):
        """Test audio capture worker with exception."""
        audio_buffer = queue.Queue()
        service.is_running = True

        # Mock the audio recorder to raise an exception
        service.audio_recorder.record_stream.side_effect = Exception("Audio error")

        # Run the worker
        service._audio_capture_worker(audio_buffer, 16000)

        # Should put end signal even on error
        assert audio_buffer.get() is None
        service.logger.error.assert_called()

    @patch("threading.Thread")
    @patch("time.time")
    def test_transcribe_realtime_basic_flow(
        self, mock_time, mock_thread, service, config
    ):
        """Test basic real-time transcription flow."""
        # Mock time to control timestamps
        mock_time.side_effect = [100.0, 101.0, 102.0, 103.0]

        # Mock thread
        mock_audio_thread = Mock()
        mock_thread.return_value = mock_audio_thread

        # Create a controlled audio buffer
        audio_buffer = queue.Queue()

        # Add some audio data
        import struct

        samples = [1000] * 8000  # 0.5 seconds of audio at 16kHz
        audio_chunk = struct.pack(f"<{len(samples)}h", *samples)
        audio_buffer.put(audio_chunk)
        audio_buffer.put(None)  # End signal

        # Mock the audio capture worker to use our controlled buffer
        def mock_audio_capture_worker(buffer, sample_rate):
            # Transfer our test data
            while not audio_buffer.empty():
                item = audio_buffer.get()
                buffer.put(item)

        with patch.object(service, "_audio_capture_worker", mock_audio_capture_worker):
            # Mock transcription
            service.transcription_service.transcribe.return_value = TranscriptionResult(
                text="hello", confidence=0.9
            )

            # Run transcription (limit to avoid infinite loop)
            results = []
            for i, result in enumerate(service.transcribe_realtime(config)):
                results.append(result)
                if i >= 2:  # Limit iterations
                    service.stop()
                    break

        # Verify thread was started
        mock_audio_thread.start.assert_called_once()

        # Should have logged startup
        service.logger.info.assert_any_call("Starting realtime transcription service")

    def test_transcribe_realtime_no_audio_timeout(self, service, config):
        """Test real-time transcription with no audio input."""
        with patch("time.time") as mock_time:
            # Mock time to simulate timeout
            mock_time.side_effect = [100.0, 106.0]  # 6 seconds elapsed

            with patch("threading.Thread") as mock_thread:
                mock_audio_thread = Mock()
                mock_thread.return_value = mock_audio_thread

                # Mock audio capture to not put anything in buffer
                def mock_audio_capture_worker(buffer, sample_rate):
                    pass  # Don't put anything in buffer

                with patch.object(
                    service, "_audio_capture_worker", mock_audio_capture_worker
                ):
                    # Run transcription
                    results = list(service.transcribe_realtime(config))

                    assert results == []
                    service.logger.error.assert_called_with(
                        "No audio input received after 5 seconds. Audio device may not be available."
                    )

    @patch("threading.Thread")
    def test_transcribe_realtime_with_exception(self, mock_thread, service, config):
        """Test real-time transcription with general exception."""
        # Mock thread to raise exception on start
        mock_audio_thread = Mock()
        mock_audio_thread.start.side_effect = Exception("Thread start failed")
        mock_thread.return_value = mock_audio_thread

        # Run transcription
        results = list(service.transcribe_realtime(config))

        assert results == []
        assert service.is_running is False
        service.logger.error.assert_called()
