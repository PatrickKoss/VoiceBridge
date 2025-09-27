"""Tests for speaker diarization services."""

from unittest.mock import Mock, patch

import pytest

from voicebridge.domain.models import TranscriptionSegment
from voicebridge.services.speaker_service import (
    MockSpeakerDiarizationService,
    PyAnnoteSpeakerDiarizationService,
    SimpleSpeakerDiarizationService,
)


class TestMockSpeakerDiarizationService:
    """Test cases for MockSpeakerDiarizationService."""

    @pytest.fixture
    def service(self):
        """MockSpeakerDiarizationService instance."""
        return MockSpeakerDiarizationService(max_speakers=3)

    @pytest.fixture
    def sample_segments(self):
        """Sample transcription segments."""
        return [
            TranscriptionSegment(
                text="Hello world", start_time=0.0, end_time=2.0, confidence=0.9
            ),
            TranscriptionSegment(
                text="How are you",
                start_time=6.0,
                end_time=8.0,
                confidence=0.8,  # 4s pause
            ),
            TranscriptionSegment(
                text="I'm fine",
                start_time=9.0,
                end_time=11.0,
                confidence=0.85,  # 1s pause
            ),
        ]

    def test_init(self, service):
        """Test service initialization."""
        assert service.max_speakers == 3

    def test_identify_speakers_empty_segments(self, service):
        """Test speaker identification with empty segments."""
        result = service.identify_speakers(b"audio_data", [])
        assert result == []

    def test_identify_speakers_with_long_pauses(self, service, sample_segments):
        """Test speaker identification with long pauses between segments."""
        result = service.identify_speakers(b"audio_data", sample_segments)

        assert len(result) == 3
        # First segment should be speaker 1
        assert result[0].speaker_id == 1
        # Second segment (after long pause) should be speaker 2
        assert result[1].speaker_id == 2
        # Third segment (short pause) should remain speaker 2
        assert result[2].speaker_id == 2

    def test_identify_speakers_no_pauses(self, service):
        """Test speaker identification with consecutive segments."""
        segments = [
            TranscriptionSegment(
                text="First", start_time=0.0, end_time=1.0, confidence=0.9
            ),
            TranscriptionSegment(
                text="Second", start_time=1.0, end_time=2.0, confidence=0.8
            ),
        ]

        result = service.identify_speakers(b"audio_data", segments)

        assert len(result) == 2
        # Both should be same speaker (no long pause)
        assert result[0].speaker_id == result[1].speaker_id == 1

    def test_identify_speakers_speaker_cycling(self, service):
        """Test speaker cycling with multiple long pauses."""
        segments = []
        for i in range(6):  # More than max_speakers
            segments.append(
                TranscriptionSegment(
                    text=f"Segment {i}",
                    start_time=i * 5.0,  # 5 second gaps
                    end_time=i * 5.0 + 1.0,
                    confidence=0.9,
                )
            )

        result = service.identify_speakers(b"audio_data", segments)

        assert len(result) == 6
        # Should cycle through speakers 1, 2, 3, 1, 2, 3
        expected_speakers = [1, 2, 3, 1, 2, 3]
        actual_speakers = [seg.speaker_id for seg in result]
        assert actual_speakers == expected_speakers

    def test_get_speaker_info_empty_segments(self, service):
        """Test speaker info generation with empty segments."""
        result = service.get_speaker_info([])
        assert result == []

    def test_get_speaker_info_no_speaker_ids(self, service):
        """Test speaker info generation with segments without speaker IDs."""
        segments = [
            TranscriptionSegment(
                text="Test", start_time=0.0, end_time=1.0, confidence=0.9
            )
        ]

        result = service.get_speaker_info(segments)
        assert result == []

    def test_get_speaker_info_with_speaker_data(self, service):
        """Test speaker info generation with speaker data."""
        segments = [
            TranscriptionSegment(
                text="Hello", start_time=0.0, end_time=2.0, confidence=0.9, speaker_id=1
            ),
            TranscriptionSegment(
                text="Hi", start_time=2.0, end_time=3.0, confidence=0.8, speaker_id=1
            ),
            TranscriptionSegment(
                text="How are you",
                start_time=3.0,
                end_time=6.0,
                confidence=0.85,
                speaker_id=2,
            ),
        ]

        result = service.get_speaker_info(segments)

        assert len(result) == 2

        # Both speakers have same total time, so order may vary
        speaker_ids = [r.speaker_id for r in result]
        assert 1 in speaker_ids
        assert 2 in speaker_ids

        speaker_1 = next(r for r in result if r.speaker_id == 1)
        speaker_2 = next(r for r in result if r.speaker_id == 2)

        assert speaker_1.total_speaking_time == 3.0  # 2.0 + 1.0
        assert abs(speaker_1.confidence - 0.85) < 0.01  # (0.9 + 0.8) / 2 = 0.85
        assert speaker_1.name == "Speaker 1"

        assert speaker_2.total_speaking_time == 3.0
        assert speaker_2.confidence == 0.85
        assert speaker_2.name == "Speaker 2"

    def test_get_speaker_info_with_none_confidence(self, service):
        """Test speaker info generation with None confidence values."""
        segments = [
            TranscriptionSegment(
                text="Hello",
                start_time=0.0,
                end_time=2.0,
                confidence=None,
                speaker_id=1,
            ),
            TranscriptionSegment(
                text="Hi", start_time=2.0, end_time=3.0, confidence=0.8, speaker_id=1
            ),
        ]

        result = service.get_speaker_info(segments)

        assert len(result) == 1
        assert result[0].confidence == 0.8  # Only count non-None confidence


class TestPyAnnoteSpeakerDiarizationService:
    """Test cases for PyAnnoteSpeakerDiarizationService."""

    @pytest.fixture
    def service(self):
        """PyAnnoteSpeakerDiarizationService instance."""
        return PyAnnoteSpeakerDiarizationService(max_speakers=2)

    @pytest.fixture
    def sample_segments(self):
        """Sample transcription segments."""
        return [
            TranscriptionSegment(
                text="Hello", start_time=0.0, end_time=1.0, confidence=0.9
            ),
            TranscriptionSegment(
                text="Hi there", start_time=2.0, end_time=3.0, confidence=0.8
            ),
        ]

    def test_init(self, service):
        """Test service initialization."""
        assert service.max_speakers == 2
        assert service.use_auth_token is None
        assert service._pipeline is None

    def test_init_with_auth_token(self):
        """Test service initialization with auth token."""
        service = PyAnnoteSpeakerDiarizationService(
            max_speakers=3, use_auth_token="test_token"
        )
        assert service.use_auth_token == "test_token"

    def test_load_pipeline_success(self, service):
        """Test successful pipeline loading."""
        # Create mock Pipeline class
        mock_pipeline_class = Mock()
        mock_instance = Mock()
        mock_pipeline_class.from_pretrained.return_value = mock_instance

        # Mock the local import in _load_pipeline
        mock_module = Mock()
        mock_module.Pipeline = mock_pipeline_class

        with patch.dict("sys.modules", {"pyannote.audio": mock_module}):
            pipeline = service._load_pipeline()

            assert pipeline is not None
            assert service._pipeline == pipeline
            mock_pipeline_class.from_pretrained.assert_called_once_with(
                "pyannote/speaker-diarization-3.1"
            )

    def test_load_pipeline_with_auth_token(self):
        """Test pipeline loading with auth token."""
        service = PyAnnoteSpeakerDiarizationService(use_auth_token="test_token")

        # Create mock Pipeline class
        mock_pipeline_class = Mock()
        mock_instance = Mock()
        mock_pipeline_class.from_pretrained.return_value = mock_instance

        # Mock the local import in _load_pipeline
        mock_module = Mock()
        mock_module.Pipeline = mock_pipeline_class

        with patch.dict("sys.modules", {"pyannote.audio": mock_module}):
            service._load_pipeline()

            mock_pipeline_class.from_pretrained.assert_called_once_with(
                "pyannote/speaker-diarization-3.1", use_auth_token="test_token"
            )

    def test_load_pipeline_import_error(self, service):
        """Test pipeline loading with import error."""
        # Simulate import error by not mocking the module
        with pytest.raises(ImportError, match="pyannote.audio not installed"):
            service._load_pipeline()

    def test_load_pipeline_runtime_error(self, service):
        """Test pipeline loading with runtime error."""
        # Create mock Pipeline class that raises exception
        mock_pipeline_class = Mock()
        mock_pipeline_class.from_pretrained.side_effect = Exception("Loading failed")

        # Mock the local import in _load_pipeline
        mock_module = Mock()
        mock_module.Pipeline = mock_pipeline_class

        with patch.dict("sys.modules", {"pyannote.audio": mock_module}):
            with pytest.raises(RuntimeError, match="Failed to load pyannote pipeline"):
                service._load_pipeline()

    def test_identify_speakers_empty_segments(self, service):
        """Test speaker identification with empty segments."""
        result = service.identify_speakers(b"audio_data", [])
        assert result == []

    @patch.object(PyAnnoteSpeakerDiarizationService, "_load_pipeline")
    def test_identify_speakers_pipeline_load_failure(
        self, mock_load, service, sample_segments
    ):
        """Test speaker identification when pipeline loading fails."""
        mock_load.side_effect = Exception("Pipeline failed")

        # Should fallback to mock service
        result = service.identify_speakers(b"audio_data", sample_segments)

        assert len(result) == 2
        assert all(seg.speaker_id is not None for seg in result)

    @patch.object(PyAnnoteSpeakerDiarizationService, "_load_pipeline")
    @patch("tempfile.NamedTemporaryFile")
    @patch("wave.open")
    def test_identify_speakers_success(
        self, mock_wave, mock_temp_file, mock_load, service, sample_segments
    ):
        """Test successful speaker identification."""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_diarization = Mock()
        mock_diarization.itertracks.return_value = [
            (Mock(start=0.0, end=1.5), None, "SPEAKER_00"),
            (Mock(start=1.5, end=3.0), None, "SPEAKER_01"),
        ]
        mock_pipeline.return_value = mock_diarization
        mock_load.return_value = mock_pipeline

        # Mock temporary file
        mock_temp_file.return_value.__enter__.return_value.name = "/tmp/test.wav"

        # Mock wave file
        mock_wav_file = Mock()
        mock_wave.return_value.__enter__.return_value = mock_wav_file

        # Mock Path.unlink
        with patch("pathlib.Path.unlink"):
            result = service.identify_speakers(b"audio_data", sample_segments)

        assert len(result) == 2
        assert all(seg.speaker_id is not None for seg in result)

    @patch.object(PyAnnoteSpeakerDiarizationService, "_load_pipeline")
    def test_identify_speakers_diarization_failure(
        self, mock_load, service, sample_segments
    ):
        """Test speaker identification when diarization fails."""
        mock_pipeline = Mock()
        mock_pipeline.side_effect = Exception("Diarization failed")
        mock_load.return_value = mock_pipeline

        # Should fallback to mock service
        result = service.identify_speakers(b"audio_data", sample_segments)

        assert len(result) == 2
        assert all(seg.speaker_id is not None for seg in result)

    def test_find_dominant_speaker_no_overlap(self, service):
        """Test finding dominant speaker with no overlap."""
        mock_diarization = Mock()
        mock_diarization.itertracks.return_value = []

        result = service._find_dominant_speaker(0.0, 1.0, mock_diarization)
        assert result == 1  # Default speaker

    def test_find_dominant_speaker_with_overlap(self, service):
        """Test finding dominant speaker with overlapping segments."""
        mock_diarization = Mock()
        mock_diarization.itertracks.return_value = [
            (Mock(start=0.0, end=0.5), None, "SPEAKER_00"),
            (Mock(start=0.3, end=1.2), None, "SPEAKER_01"),  # More overlap
        ]

        result = service._find_dominant_speaker(0.0, 1.0, mock_diarization)
        assert result == 2  # Speaker 01 -> ID 2

    def test_find_dominant_speaker_with_invalid_label(self, service):
        """Test finding dominant speaker with invalid speaker label."""
        mock_diarization = Mock()
        mock_diarization.itertracks.return_value = [
            (Mock(start=0.0, end=1.0), None, "INVALID_LABEL"),
        ]

        result = service._find_dominant_speaker(0.0, 1.0, mock_diarization)
        assert isinstance(result, int)
        assert 1 <= result <= service.max_speakers

    def test_get_speaker_info_same_as_mock(self, service):
        """Test that get_speaker_info works the same as MockSpeakerDiarizationService."""
        segments = [
            TranscriptionSegment(
                text="Hello", start_time=0.0, end_time=2.0, confidence=0.9, speaker_id=1
            ),
            TranscriptionSegment(
                text="Hi", start_time=2.0, end_time=5.0, confidence=0.8, speaker_id=2
            ),
        ]

        result = service.get_speaker_info(segments)

        assert len(result) == 2
        assert result[0].speaker_id == 2  # Longer speaking time
        assert result[0].total_speaking_time == 3.0
        assert result[1].speaker_id == 1
        assert result[1].total_speaking_time == 2.0


class TestSimpleSpeakerDiarizationService:
    """Test cases for SimpleSpeakerDiarizationService."""

    @pytest.fixture
    def service(self):
        """SimpleSpeakerDiarizationService instance."""
        return SimpleSpeakerDiarizationService(max_speakers=3, silence_threshold=2.0)

    def test_init(self, service):
        """Test service initialization."""
        assert service.max_speakers == 3
        assert service.silence_threshold == 2.0

    def test_identify_speakers_empty_segments(self, service):
        """Test speaker identification with empty segments."""
        result = service.identify_speakers(b"audio_data", [])
        assert result == []

    def test_identify_speakers_single_segment(self, service):
        """Test speaker identification with single segment."""
        segments = [
            TranscriptionSegment(
                text="Hello", start_time=0.0, end_time=1.0, confidence=0.9
            )
        ]

        result = service.identify_speakers(b"audio_data", segments)

        assert len(result) == 1
        assert result[0].speaker_id == 1

    def test_identify_speakers_with_long_silence(self, service):
        """Test speaker identification with long silence (> threshold)."""
        segments = [
            TranscriptionSegment(
                text="First", start_time=0.0, end_time=1.0, confidence=0.9
            ),
            TranscriptionSegment(
                text="Second",
                start_time=4.0,
                end_time=5.0,
                confidence=0.8,  # 3s gap
            ),
        ]

        result = service.identify_speakers(b"audio_data", segments)

        assert len(result) == 2
        assert result[0].speaker_id == 1
        assert result[1].speaker_id == 2  # Different speaker after long silence

    def test_identify_speakers_with_short_silence(self, service):
        """Test speaker identification with short silence (< threshold)."""
        segments = [
            TranscriptionSegment(
                text="First", start_time=0.0, end_time=1.0, confidence=0.9
            ),
            TranscriptionSegment(
                text="Second",
                start_time=1.5,
                end_time=2.5,
                confidence=0.8,  # 0.5s gap
            ),
        ]

        result = service.identify_speakers(b"audio_data", segments)

        assert len(result) == 2
        assert result[0].speaker_id == result[1].speaker_id == 1  # Same speaker

    def test_identify_speakers_duration_difference(self, service):
        """Test speaker identification based on duration differences."""
        segments = [
            TranscriptionSegment(
                text="Short",
                start_time=0.0,
                end_time=1.0,
                confidence=0.9,  # 1s duration
            ),
            TranscriptionSegment(
                text="Much longer segment",
                start_time=1.6,
                end_time=4.6,
                confidence=0.8,  # 3s duration
            ),
        ]

        result = service.identify_speakers(b"audio_data", segments)

        assert len(result) == 2
        # Should detect speaker change due to significant duration difference
        assert result[0].speaker_id != result[1].speaker_id

    def test_get_next_speaker_avoiding_recent(self, service):
        """Test getting next speaker while avoiding recent speakers."""
        # Current speaker 1, recent speakers [2]
        result = service._get_next_speaker(1, [2])
        assert result == 3  # Should pick 3 (avoiding 1 and 2)

    def test_get_next_speaker_all_recent(self, service):
        """Test getting next speaker when all speakers are recent."""
        # Current speaker 1, recent speakers [2, 3] (all speakers used)
        result = service._get_next_speaker(1, [2, 3])
        assert result == 2  # Should cycle to next (1 + 1) % 3 + 1 = 2

    def test_get_next_speaker_empty_recent(self, service):
        """Test getting next speaker with empty recent list."""
        result = service._get_next_speaker(1, [])
        assert result == 2  # Should pick next available

    def test_speaker_history_tracking(self, service):
        """Test that speaker history is properly tracked and limited."""
        segments = []
        for i in range(6):  # Create 6 segments
            segments.append(
                TranscriptionSegment(
                    text=f"Segment {i}",
                    start_time=i * 3.0,  # 3s gaps (> threshold)
                    end_time=i * 3.0 + 1.0,
                    confidence=0.9,
                )
            )

        result = service.identify_speakers(b"audio_data", segments)

        assert len(result) == 6
        # Should cycle through speakers while avoiding recent ones
        speaker_ids = [seg.speaker_id for seg in result]
        assert len(set(speaker_ids)) <= service.max_speakers

    def test_get_speaker_info_same_as_mock(self, service):
        """Test that get_speaker_info works the same as other services."""
        segments = [
            TranscriptionSegment(
                text="Hello", start_time=0.0, end_time=2.0, confidence=0.9, speaker_id=1
            ),
            TranscriptionSegment(
                text="Hi", start_time=2.0, end_time=5.0, confidence=0.8, speaker_id=2
            ),
        ]

        result = service.get_speaker_info(segments)

        assert len(result) == 2
        assert result[0].speaker_id == 2  # Longer speaking time
        assert result[0].total_speaking_time == 3.0
        assert result[1].speaker_id == 1
        assert result[1].total_speaking_time == 2.0
