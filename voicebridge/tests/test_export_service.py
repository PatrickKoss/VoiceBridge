"""Tests for export service."""

import csv
import json
from io import StringIO
from unittest.mock import patch

import pytest

from voicebridge.domain.models import (
    ExportConfig,
    OutputFormat,
    SpeakerInfo,
    TranscriptionResult,
    TranscriptionSegment,
)
from voicebridge.services.export_service import DefaultExportService


class TestDefaultExportService:
    """Test cases for DefaultExportService."""

    @pytest.fixture
    def service(self):
        """DefaultExportService instance."""
        return DefaultExportService()

    @pytest.fixture
    def sample_result(self):
        """Sample transcription result."""
        return TranscriptionResult(
            text="Hello world. How are you today?",
            language="en",
            duration=5.5,
            confidence=0.92,
            segments=[
                TranscriptionSegment(
                    text="Hello world.",
                    start_time=0.0,
                    end_time=2.5,
                    confidence=0.95,
                    speaker_id=1,
                ),
                TranscriptionSegment(
                    text="How are you today?",
                    start_time=2.5,
                    end_time=5.5,
                    confidence=0.89,
                    speaker_id=2,
                ),
            ],
            speakers=[
                SpeakerInfo(
                    speaker_id=1,
                    name="Speaker 1",
                    total_speaking_time=2.5,
                    confidence=0.95,
                ),
                SpeakerInfo(
                    speaker_id=2,
                    name="Speaker 2",
                    total_speaking_time=3.0,
                    confidence=0.89,
                ),
            ],
        )

    @pytest.fixture
    def basic_config(self):
        """Basic export configuration."""
        return ExportConfig(
            format=OutputFormat.JSON,
            output_file=None,
            include_confidence=False,
            include_speaker_info=False,
        )

    def test_get_supported_formats(self, service):
        """Test getting supported export formats."""
        formats = service.get_supported_formats()

        expected = [
            OutputFormat.JSON,
            OutputFormat.PLAIN_TEXT,
            OutputFormat.CSV,
            OutputFormat.SRT,
            OutputFormat.VTT,
        ]
        assert formats == expected

    def test_export_transcription_json_basic(
        self, service, sample_result, basic_config
    ):
        """Test basic JSON export."""
        basic_config.format = OutputFormat.JSON

        with patch("voicebridge.services.export_service.datetime") as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = (
                "2023-01-01T12:00:00"
            )

            result = service.export_transcription(sample_result, basic_config)

        data = json.loads(result)
        assert data["text"] == "Hello world. How are you today?"
        assert data["language"] == "en"
        assert data["duration"] == 5.5
        assert data["timestamp"] == "2023-01-01T12:00:00"
        assert "confidence" not in data  # Not included by default
        assert len(data["segments"]) == 2

    def test_export_transcription_json_with_confidence(self, service, sample_result):
        """Test JSON export with confidence."""
        config = ExportConfig(
            format=OutputFormat.JSON,
            include_confidence=True,
            include_speaker_info=False,
        )

        with patch("voicebridge.services.export_service.datetime") as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = (
                "2023-01-01T12:00:00"
            )

            result = service.export_transcription(sample_result, config)

        data = json.loads(result)
        assert data["confidence"] == 0.92
        assert data["segments"][0]["confidence"] == 0.95
        assert data["segments"][1]["confidence"] == 0.89

    def test_export_transcription_json_with_speakers(self, service, sample_result):
        """Test JSON export with speaker information."""
        config = ExportConfig(
            format=OutputFormat.JSON,
            include_confidence=False,
            include_speaker_info=True,
        )

        with patch("voicebridge.services.export_service.datetime") as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = (
                "2023-01-01T12:00:00"
            )

            result = service.export_transcription(sample_result, config)

        data = json.loads(result)
        assert data["segments"][0]["speaker_id"] == 1
        assert data["segments"][1]["speaker_id"] == 2
        assert len(data["speakers"]) == 2
        assert data["speakers"][0]["speaker_id"] == 1
        assert data["speakers"][0]["name"] == "Speaker 1"

    def test_export_transcription_plain_text_basic(self, service, sample_result):
        """Test basic plain text export."""
        config = ExportConfig(
            format=OutputFormat.PLAIN_TEXT,
            include_confidence=False,
            include_speaker_info=False,
        )

        result = service.export_transcription(sample_result, config)

        lines = result.split("\n")
        assert "Language: en" in lines
        assert "Duration: 5.50s" in lines
        assert "[00:00:00] Hello world." in result
        assert "[00:00:02] How are you today?" in result

    def test_export_transcription_plain_text_with_confidence(
        self, service, sample_result
    ):
        """Test plain text export with confidence."""
        config = ExportConfig(
            format=OutputFormat.PLAIN_TEXT,
            include_confidence=True,
            include_speaker_info=False,
        )

        result = service.export_transcription(sample_result, config)

        assert "Overall Confidence: 92.00%" in result
        assert "(confidence: 95.00%)" in result
        assert "(confidence: 89.00%)" in result

    def test_export_transcription_plain_text_with_speakers(
        self, service, sample_result
    ):
        """Test plain text export with speaker information."""
        config = ExportConfig(
            format=OutputFormat.PLAIN_TEXT,
            include_confidence=False,
            include_speaker_info=True,
        )

        result = service.export_transcription(sample_result, config)

        assert "Speaker 1: Hello world." in result
        assert "Speaker 2: How are you today?" in result

    def test_export_transcription_plain_text_single_segment(self, service):
        """Test plain text export with single segment (no timestamps)."""
        result = TranscriptionResult(
            text="Single segment text",
            language="en",
            duration=3.0,
            segments=[],
        )

        config = ExportConfig(format=OutputFormat.PLAIN_TEXT)

        output = service.export_transcription(result, config)

        assert "Single segment text" in output
        assert "[00:" not in output  # No timestamps for single segment

    def test_export_transcription_csv_basic(self, service, sample_result):
        """Test basic CSV export."""
        config = ExportConfig(
            format=OutputFormat.CSV,
            include_confidence=False,
            include_speaker_info=False,
        )

        result = service.export_transcription(sample_result, config)

        # Parse CSV
        reader = csv.DictReader(StringIO(result))
        rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["start_time"] == "0.0"
        assert rows[0]["end_time"] == "2.5"
        assert rows[0]["text"] == "Hello world."
        assert rows[1]["start_time"] == "2.5"
        assert rows[1]["end_time"] == "5.5"
        assert rows[1]["text"] == "How are you today?"

    def test_export_transcription_csv_with_confidence(self, service, sample_result):
        """Test CSV export with confidence."""
        config = ExportConfig(
            format=OutputFormat.CSV,
            include_confidence=True,
            include_speaker_info=False,
        )

        result = service.export_transcription(sample_result, config)

        reader = csv.DictReader(StringIO(result))
        rows = list(reader)

        assert "confidence" in rows[0]
        assert rows[0]["confidence"] == "0.95"
        assert rows[1]["confidence"] == "0.89"

    def test_export_transcription_csv_with_speakers(self, service, sample_result):
        """Test CSV export with speaker information."""
        config = ExportConfig(
            format=OutputFormat.CSV,
            include_confidence=False,
            include_speaker_info=True,
        )

        result = service.export_transcription(sample_result, config)

        reader = csv.DictReader(StringIO(result))
        rows = list(reader)

        assert "speaker_id" in rows[0]
        assert rows[0]["speaker_id"] == "1"
        assert rows[1]["speaker_id"] == "2"

    def test_export_transcription_csv_no_segments(self, service):
        """Test CSV export with no segments."""
        result = TranscriptionResult(
            text="Single text",
            duration=3.0,
            confidence=0.8,
        )

        config = ExportConfig(
            format=OutputFormat.CSV,
            include_confidence=True,
        )

        output = service.export_transcription(result, config)

        reader = csv.DictReader(StringIO(output))
        rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["start_time"] == "0.0"
        assert rows[0]["end_time"] == "3.0"
        assert rows[0]["text"] == "Single text"
        assert rows[0]["confidence"] == "0.8"

    def test_export_transcription_srt_basic(self, service, sample_result):
        """Test basic SRT export."""
        config = ExportConfig(format=OutputFormat.SRT)

        result = service.export_transcription(sample_result, config)

        lines = result.split("\n")
        assert "1" in lines
        assert "00:00:00,000 --> 00:00:02,500" in result
        assert "Hello world." in result
        assert "2" in lines
        assert "00:00:02,500 --> 00:00:05,500" in result
        assert "How are you today?" in result

    def test_export_transcription_srt_with_speakers(self, service, sample_result):
        """Test SRT export with speaker information."""
        config = ExportConfig(
            format=OutputFormat.SRT,
            include_speaker_info=True,
        )

        result = service.export_transcription(sample_result, config)

        assert "Speaker 1: Hello world." in result
        assert "Speaker 2: How are you today?" in result

    def test_export_transcription_srt_no_segments(self, service):
        """Test SRT export with no segments."""
        result = TranscriptionResult(
            text="Single text",
            duration=5.0,
        )

        config = ExportConfig(format=OutputFormat.SRT)

        output = service.export_transcription(result, config)

        assert "1" in output
        assert "00:00:00,000 --> 00:00:05,000" in output
        assert "Single text" in output

    def test_export_transcription_vtt_basic(self, service, sample_result):
        """Test basic VTT export."""
        config = ExportConfig(format=OutputFormat.VTT)

        result = service.export_transcription(sample_result, config)

        lines = result.split("\n")
        assert lines[0] == "WEBVTT"
        assert "00:00:00.000 --> 00:00:02.500" in result
        assert "Hello world." in result
        assert "00:00:02.500 --> 00:00:05.500" in result
        assert "How are you today?" in result

    def test_export_transcription_vtt_with_speakers(self, service, sample_result):
        """Test VTT export with speaker information."""
        config = ExportConfig(
            format=OutputFormat.VTT,
            include_speaker_info=True,
        )

        result = service.export_transcription(sample_result, config)

        assert "<v Speaker 1>Hello world." in result
        assert "<v Speaker 2>How are you today?" in result

    def test_export_transcription_vtt_no_segments(self, service):
        """Test VTT export with no segments."""
        result = TranscriptionResult(
            text="Single text",
            duration=7.5,
        )

        config = ExportConfig(format=OutputFormat.VTT)

        output = service.export_transcription(result, config)

        assert "WEBVTT" in output
        assert "00:00:00.000 --> 00:00:07.500" in output
        assert "Single text" in output

    def test_export_transcription_unsupported_format(self, service, sample_result):
        """Test export with unsupported format."""
        config = ExportConfig(format="unsupported")

        with pytest.raises(ValueError, match="Unsupported format"):
            service.export_transcription(sample_result, config)

    def test_export_to_file_with_specified_path(self, service, sample_result, tmp_path):
        """Test exporting to file with specified path."""
        output_file = tmp_path / "test_export.json"
        config = ExportConfig(
            format=OutputFormat.JSON,
            output_file=str(output_file),
        )

        with patch("voicebridge.services.export_service.datetime") as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = (
                "2023-01-01T12:00:00"
            )

            result = service.export_to_file(sample_result, config)

        assert result is True
        assert output_file.exists()

        # Verify content
        content = output_file.read_text()
        data = json.loads(content)
        assert data["text"] == "Hello world. How are you today?"

    def test_export_to_file_auto_generated_name(self, service, sample_result):
        """Test exporting to file with auto-generated name."""
        config = ExportConfig(format=OutputFormat.JSON)

        with patch("voicebridge.services.export_service.datetime") as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20230101_120000"
            mock_datetime.now.return_value.isoformat.return_value = (
                "2023-01-01T12:00:00"
            )

            with patch("pathlib.Path.write_text") as mock_write:
                result = service.export_to_file(sample_result, config)

        assert result is True
        # Check that the auto-generated filename was used
        mock_write.assert_called_once()

    def test_export_to_file_exception(self, service, sample_result):
        """Test export to file with exception."""
        config = ExportConfig(
            format=OutputFormat.JSON,
            output_file="/invalid/path/file.json",
        )

        result = service.export_to_file(sample_result, config)

        assert result is False

    def test_format_timestamp(self, service):
        """Test timestamp formatting."""
        assert service._format_timestamp(0) == "00:00:00"
        assert service._format_timestamp(65) == "00:01:05"
        assert service._format_timestamp(3661) == "01:01:01"
        assert (
            service._format_timestamp(3661.5) == "01:01:01"
        )  # Truncates fractional seconds

    def test_seconds_to_srt_timestamp(self, service):
        """Test SRT timestamp formatting."""
        assert service._seconds_to_srt_timestamp(0) == "00:00:00,000"
        assert service._seconds_to_srt_timestamp(65.5) == "00:01:05,500"
        assert service._seconds_to_srt_timestamp(3661.123) == "01:01:01,123"

    def test_seconds_to_vtt_timestamp(self, service):
        """Test VTT timestamp formatting."""
        assert service._seconds_to_vtt_timestamp(0) == "00:00:00.000"
        assert service._seconds_to_vtt_timestamp(65.5) == "00:01:05.500"
        assert service._seconds_to_vtt_timestamp(3661.123) == "01:01:01.123"

    def test_json_export_no_segments(self, service):
        """Test JSON export with no segments."""
        result = TranscriptionResult(
            text="Simple text",
            language="en",
            duration=3.0,
            confidence=0.9,
        )

        config = ExportConfig(
            format=OutputFormat.JSON,
            include_confidence=True,
        )

        with patch("voicebridge.services.export_service.datetime") as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = (
                "2023-01-01T12:00:00"
            )

            output = service.export_transcription(result, config)

        data = json.loads(output)
        assert data["text"] == "Simple text"
        assert data["confidence"] == 0.9
        assert "segments" not in data or data["segments"] == []

    def test_json_export_no_speakers(self, service, sample_result):
        """Test JSON export with no speaker information."""
        # Remove speakers from result
        sample_result.speakers = []

        config = ExportConfig(
            format=OutputFormat.JSON,
            include_speaker_info=True,
        )

        with patch("voicebridge.services.export_service.datetime") as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = (
                "2023-01-01T12:00:00"
            )

            output = service.export_transcription(sample_result, config)

        data = json.loads(output)
        assert "speakers" not in data or data["speakers"] == []

    def test_edge_case_zero_duration(self, service):
        """Test export with zero duration."""
        result = TranscriptionResult(
            text="Quick text",
            duration=0.0,
        )

        config = ExportConfig(format=OutputFormat.SRT)

        output = service.export_transcription(result, config)

        assert "00:00:00,000 --> 00:00:00,000" in output

    def test_edge_case_none_values(self, service):
        """Test export with None values in segments."""
        result = TranscriptionResult(
            text="Test text",
            language=None,
            duration=None,
            confidence=None,
            segments=[
                TranscriptionSegment(
                    text="Test",
                    start_time=0.0,
                    end_time=1.0,
                    confidence=None,
                    speaker_id=None,
                ),
            ],
        )

        config = ExportConfig(
            format=OutputFormat.JSON,
            include_confidence=True,
            include_speaker_info=True,
        )

        with patch("voicebridge.services.export_service.datetime") as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = (
                "2023-01-01T12:00:00"
            )

            output = service.export_transcription(result, config)

        data = json.loads(output)
        assert data["text"] == "Test text"
        # Should handle None values gracefully
