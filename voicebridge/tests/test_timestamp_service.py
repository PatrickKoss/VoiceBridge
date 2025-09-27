"""Tests for timestamp services."""

import pytest

from voicebridge.domain.models import TimestampMode, TranscriptionSegment
from voicebridge.services.timestamp_service import DefaultTimestampService


class TestDefaultTimestampService:
    """Test cases for DefaultTimestampService."""

    @pytest.fixture
    def service(self):
        """DefaultTimestampService instance."""
        return DefaultTimestampService()

    @pytest.fixture
    def sample_segments(self):
        """Sample transcription segments."""
        return [
            TranscriptionSegment(
                text="Hello world", start_time=0.0, end_time=1.5, confidence=0.9
            ),
            TranscriptionSegment(
                text="How are you?", start_time=1.5, end_time=3.0, confidence=0.8
            ),
            TranscriptionSegment(
                text="I am fine", start_time=4.5, end_time=6.0, confidence=0.85
            ),
            TranscriptionSegment(
                text="Thank you!", start_time=6.0, end_time=7.0, confidence=0.92
            ),
        ]

    def test_process_segments_word_level(self, service, sample_segments):
        """Test processing segments with word level timestamps."""
        result = service.process_segments(
            sample_segments, TimestampMode.WORD_LEVEL.value
        )
        assert result == sample_segments

    def test_process_segments_sentence_level(self, service, sample_segments):
        """Test processing segments with sentence level timestamps."""
        result = service.process_segments(
            sample_segments, TimestampMode.SENTENCE_LEVEL.value
        )

        # Should group into sentences based on punctuation and pauses
        assert len(result) == 2  # Two sentences due to pause between segments 2 and 3

        # First sentence: "Hello world How are you?"
        assert result[0].text == "Hello world How are you?"
        assert result[0].start_time == 0.0
        assert result[0].end_time == 3.0

        # Second sentence: "I am fine Thank you!"
        assert result[1].text == "I am fine Thank you!"
        assert result[1].start_time == 4.5
        assert result[1].end_time == 7.0

    def test_process_segments_paragraph_level(self, service, sample_segments):
        """Test processing segments with paragraph level timestamps."""
        result = service.process_segments(
            sample_segments, TimestampMode.PARAGRAPH_LEVEL.value
        )

        # Should group all sentences into one paragraph
        assert len(result) == 1
        assert result[0].text == "Hello world How are you? I am fine Thank you!"
        assert result[0].start_time == 0.0
        assert result[0].end_time == 7.0

    def test_process_segments_unknown_mode(self, service, sample_segments):
        """Test processing segments with unknown mode."""
        result = service.process_segments(sample_segments, "unknown_mode")
        assert result == sample_segments

    def test_group_by_sentences_empty_segments(self, service):
        """Test grouping by sentences with empty segments."""
        result = service.group_by_sentences([])
        assert result == []

    def test_group_by_sentences_with_punctuation(self, service):
        """Test grouping by sentences based on punctuation."""
        segments = [
            TranscriptionSegment(
                text="Hello.", start_time=0.0, end_time=1.0, confidence=0.9
            ),
            TranscriptionSegment(
                text="World!", start_time=1.0, end_time=2.0, confidence=0.8
            ),
            TranscriptionSegment(
                text="How are you?", start_time=2.0, end_time=3.5, confidence=0.85
            ),
        ]

        result = service.group_by_sentences(segments)

        # Each punctuation marks a sentence end
        assert len(result) == 3
        assert result[0].text == "Hello."
        assert result[1].text == "World!"
        assert result[2].text == "How are you?"

    def test_group_by_sentences_with_long_pause(self, service):
        """Test grouping by sentences based on long pauses."""
        segments = [
            TranscriptionSegment(
                text="Hello", start_time=0.0, end_time=1.0, confidence=0.9
            ),
            TranscriptionSegment(
                text="world",
                start_time=2.5,
                end_time=3.5,
                confidence=0.8,  # 1.5s pause > 1.0s threshold
            ),
            TranscriptionSegment(
                text="today",
                start_time=3.6,
                end_time=4.0,
                confidence=0.85,  # 0.1s pause < 1.0s threshold
            ),
        ]

        result = service.group_by_sentences(segments)

        # Should split at long pause
        assert len(result) == 2
        assert result[0].text == "Hello"
        assert result[1].text == "world today"

    def test_group_by_paragraphs_empty_segments(self, service):
        """Test grouping by paragraphs with empty segments."""
        result = service.group_by_paragraphs([])
        assert result == []

    def test_group_by_paragraphs_with_paragraph_breaks(self, service):
        """Test grouping by paragraphs based on long pauses."""
        segments = [
            TranscriptionSegment(
                text="First sentence.", start_time=0.0, end_time=2.0, confidence=0.9
            ),
            TranscriptionSegment(
                text="Second sentence.",
                start_time=5.0,
                end_time=7.0,
                confidence=0.8,  # 3s pause > 2s threshold
            ),
        ]

        result = service.group_by_paragraphs(segments)

        # Should create separate paragraphs due to long pause
        assert len(result) == 2
        assert result[0].text == "First sentence."
        assert result[1].text == "Second sentence."

    def test_group_by_paragraphs_max_sentences(self, service):
        """Test grouping by paragraphs with max sentences limit."""
        # Create 6 sentences with no long pauses
        segments = []
        for i in range(6):
            segments.append(
                TranscriptionSegment(
                    text=f"Sentence {i + 1}.",
                    start_time=i * 1.0,
                    end_time=(i + 1) * 1.0,
                    confidence=0.9,
                )
            )

        result = service.group_by_paragraphs(segments)

        # Should split after 5 sentences
        assert len(result) == 2
        assert (
            "Sentence 1. Sentence 2. Sentence 3. Sentence 4. Sentence 5."
            in result[0].text
        )
        assert "Sentence 6." in result[1].text

    def test_is_long_pause_true(self, service):
        """Test long pause detection when pause exceeds threshold."""
        segments = [
            TranscriptionSegment(
                text="First", start_time=0.0, end_time=1.0, confidence=0.9
            ),
            TranscriptionSegment(
                text="Second",
                start_time=2.5,
                end_time=3.5,
                confidence=0.8,  # 1.5s pause > 1.0s
            ),
        ]

        result = service._is_long_pause(segments[0], segments)
        assert result is True

    def test_is_long_pause_false(self, service):
        """Test long pause detection when pause is within threshold."""
        segments = [
            TranscriptionSegment(
                text="First", start_time=0.0, end_time=1.0, confidence=0.9
            ),
            TranscriptionSegment(
                text="Second",
                start_time=1.5,
                end_time=2.5,
                confidence=0.8,  # 0.5s pause < 1.0s
            ),
        ]

        result = service._is_long_pause(segments[0], segments)
        assert result is False

    def test_is_long_pause_last_segment(self, service):
        """Test long pause detection for last segment."""
        segments = [
            TranscriptionSegment(
                text="Last", start_time=0.0, end_time=1.0, confidence=0.9
            ),
        ]

        result = service._is_long_pause(segments[0], segments)
        assert result is False

    def test_is_paragraph_break_true(self, service):
        """Test paragraph break detection when pause exceeds threshold."""
        current = TranscriptionSegment(
            text="Current", start_time=0.0, end_time=1.0, confidence=0.9
        )
        next_seg = TranscriptionSegment(
            text="Next",
            start_time=3.5,
            end_time=4.5,
            confidence=0.8,  # 2.5s pause > 2.0s
        )

        result = service._is_paragraph_break(current, next_seg)
        assert result is True

    def test_is_paragraph_break_false(self, service):
        """Test paragraph break detection when pause is within threshold."""
        current = TranscriptionSegment(
            text="Current", start_time=0.0, end_time=1.0, confidence=0.9
        )
        next_seg = TranscriptionSegment(
            text="Next",
            start_time=2.5,
            end_time=3.5,
            confidence=0.8,  # 1.5s pause < 2.0s
        )

        result = service._is_paragraph_break(current, next_seg)
        assert result is False

    def test_is_paragraph_break_no_next_segment(self, service):
        """Test paragraph break detection with no next segment."""
        current = TranscriptionSegment(
            text="Current", start_time=0.0, end_time=1.0, confidence=0.9
        )

        result = service._is_paragraph_break(current, None)
        assert result is True

    def test_merge_segments_basic(self, service):
        """Test merging segments."""
        segments = [
            TranscriptionSegment(
                text="Hello", start_time=0.0, end_time=1.0, confidence=0.9
            ),
            TranscriptionSegment(
                text="world", start_time=1.0, end_time=2.0, confidence=0.8
            ),
        ]

        result = service._merge_segments(segments, "Hello world")

        assert result.text == "Hello world"
        assert result.start_time == 0.0
        assert result.end_time == 2.0
        assert abs(result.confidence - 0.85) < 0.001  # Average of 0.9 and 0.8

    def test_merge_segments_empty_list(self, service):
        """Test merging empty segments list."""
        with pytest.raises(ValueError, match="Cannot merge empty segments list"):
            service._merge_segments([], "")

    def test_merge_segments_with_none_confidence(self, service):
        """Test merging segments where some have None confidence."""
        segments = [
            TranscriptionSegment(
                text="Hello", start_time=0.0, end_time=1.0, confidence=0.9
            ),
            TranscriptionSegment(
                text="world", start_time=1.0, end_time=2.0, confidence=None
            ),
        ]

        result = service._merge_segments(segments, "Hello world")

        assert result.confidence is None

    def test_merge_segments_with_speakers(self, service):
        """Test merging segments with speaker information."""
        segments = [
            TranscriptionSegment(
                text="Hello", start_time=0.0, end_time=2.0, confidence=0.9, speaker_id=1
            ),
            TranscriptionSegment(
                text="world", start_time=2.0, end_time=3.0, confidence=0.8, speaker_id=2
            ),
        ]

        result = service._merge_segments(segments, "Hello world")

        # Should choose speaker 1 due to longer duration (2s vs 1s)
        assert result.speaker_id == 1

    def test_determine_primary_speaker_empty_segments(self, service):
        """Test determining primary speaker with empty segments."""
        result = service._determine_primary_speaker([])
        assert result is None

    def test_determine_primary_speaker_no_speaker_ids(self, service):
        """Test determining primary speaker with no speaker IDs."""
        segments = [
            TranscriptionSegment(
                text="Hello", start_time=0.0, end_time=1.0, confidence=0.9
            ),
        ]

        result = service._determine_primary_speaker(segments)
        assert result is None

    def test_determine_primary_speaker_with_speakers(self, service):
        """Test determining primary speaker with multiple speakers."""
        segments = [
            TranscriptionSegment(
                text="Hello",
                start_time=0.0,
                end_time=3.0,
                confidence=0.9,
                speaker_id=1,  # 3s
            ),
            TranscriptionSegment(
                text="world",
                start_time=3.0,
                end_time=4.0,
                confidence=0.8,
                speaker_id=2,  # 1s
            ),
            TranscriptionSegment(
                text="today",
                start_time=4.0,
                end_time=5.5,
                confidence=0.85,
                speaker_id=1,  # 1.5s
            ),
        ]

        result = service._determine_primary_speaker(segments)

        # Speaker 1 has total 4.5s, speaker 2 has 1s
        assert result == 1

    def test_comprehensive_sentence_grouping(self, service):
        """Test comprehensive sentence grouping with mixed triggers."""
        segments = [
            TranscriptionSegment(
                text="Start of sentence", start_time=0.0, end_time=1.0, confidence=0.9
            ),
            TranscriptionSegment(
                text="continues here.", start_time=1.0, end_time=2.5, confidence=0.8
            ),
            TranscriptionSegment(
                text="Next sentence",
                start_time=4.0,
                end_time=5.0,
                confidence=0.85,  # Long pause
            ),
            TranscriptionSegment(
                text="without punctuation",
                start_time=5.0,
                end_time=6.0,
                confidence=0.87,
            ),
            TranscriptionSegment(
                text="Final!", start_time=6.0, end_time=7.0, confidence=0.9
            ),
        ]

        result = service.group_by_sentences(segments)

        assert len(result) == 2
        assert result[0].text == "Start of sentence continues here."
        assert (
            result[1].text == "Next sentence without punctuation Final!"
        )  # Grouped together

    def test_confidence_averaging_with_mixed_values(self, service):
        """Test confidence averaging with mixed confidence values."""
        segments = [
            TranscriptionSegment(
                text="High", start_time=0.0, end_time=1.0, confidence=0.95
            ),
            TranscriptionSegment(
                text="Medium", start_time=1.0, end_time=2.0, confidence=0.75
            ),
            TranscriptionSegment(
                text="Low", start_time=2.0, end_time=3.0, confidence=0.55
            ),
        ]

        result = service._merge_segments(segments, "High Medium Low")

        expected_confidence = (0.95 + 0.75 + 0.55) / 3
        assert abs(result.confidence - expected_confidence) < 0.001
