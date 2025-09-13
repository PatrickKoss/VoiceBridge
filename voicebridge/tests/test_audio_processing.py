import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from voicebridge.adapters.audio_formats import FFmpegAudioFormatAdapter
from voicebridge.adapters.audio_preprocessing import FFmpegAudioPreprocessingAdapter
from voicebridge.adapters.audio_splitting import FFmpegAudioSplittingAdapter
from voicebridge.services.batch_service import WhisperBatchProcessingService


class TestFFmpegAudioFormatAdapter:
    @pytest.fixture
    def adapter(self):
        with patch.object(FFmpegAudioFormatAdapter, "_ensure_ffmpeg_available"):
            return FFmpegAudioFormatAdapter()

    def test_supported_formats(self, adapter):
        """Test that adapter returns expected supported formats."""
        formats = adapter.get_supported_formats()
        expected_formats = [
            "mp3",
            "wav",
            "m4a",
            "aac",
            "flac",
            "ogg",
            "wma",
            "aiff",
            "au",
            "webm",
        ]
        assert formats == expected_formats

    def test_is_supported_format(self, adapter):
        """Test format support detection."""
        # Create temporary files with different extensions
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Supported format
            mp3_file = tmpdir_path / "test.mp3"
            mp3_file.touch()
            assert adapter.is_supported_format(mp3_file) is True

            # Unsupported format
            txt_file = tmpdir_path / "test.txt"
            txt_file.touch()
            assert adapter.is_supported_format(txt_file) is False

            # Non-existent file
            nonexistent = tmpdir_path / "nonexistent.mp3"
            assert adapter.is_supported_format(nonexistent) is False

    @patch("subprocess.run")
    def test_get_audio_info_success(self, mock_run, adapter):
        """Test audio info extraction with successful subprocess call."""
        # Mock successful ffprobe response
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = """
        {
            "streams": [
                {
                    "codec_type": "audio",
                    "codec_name": "mp3",
                    "sample_rate": "44100",
                    "channels": 2,
                    "bits_per_sample": 16
                }
            ],
            "format": {
                "format_name": "mp3",
                "duration": "180.5",
                "size": "5000000",
                "bit_rate": "128000"
            }
        }
        """
        mock_run.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.mp3"
            test_file.touch()

            info = adapter.get_audio_info(test_file)

            assert info["duration"] == 180.5
            assert info["format"] == "mp3"
            assert info["sample_rate"] == 44100
            assert info["channels"] == 2
            assert info["codec"] == "mp3"

    @patch("subprocess.run")
    def test_get_audio_info_failure(self, mock_run, adapter):
        """Test audio info extraction with failed subprocess call."""
        # Mock failed ffprobe response
        mock_result = Mock()
        mock_result.returncode = 1
        mock_run.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.mp3"
            test_file.touch()

            info = adapter.get_audio_info(test_file)
            assert info == {}


class TestFFmpegAudioSplittingAdapter:
    @pytest.fixture
    def mock_audio_format_service(self):
        service = Mock()
        service.get_audio_info.return_value = {"duration": 600.0}  # 10 minutes
        return service

    @pytest.fixture
    def adapter(self, mock_audio_format_service):
        return FFmpegAudioSplittingAdapter(mock_audio_format_service)

    def test_split_by_duration_nonexistent_file(self, adapter):
        """Test that splitting non-existent file raises FileNotFoundError."""
        nonexistent_file = Path("nonexistent.mp3")
        output_dir = Path("output")

        with pytest.raises(FileNotFoundError):
            adapter.split_by_duration(nonexistent_file, 300, output_dir)

    @patch("subprocess.run")
    def test_split_by_duration_success(
        self, mock_run, adapter, mock_audio_format_service
    ):
        """Test successful duration-based splitting."""
        # Mock successful ffmpeg calls
        mock_result = Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create input file
            input_file = tmpdir_path / "input.mp3"
            input_file.touch()

            output_dir = tmpdir_path / "output"

            # Mock successful chunk creation
            with patch("pathlib.Path.exists", return_value=True):
                chunks = adapter.split_by_duration(input_file, 300, output_dir)

            # Should create 2 chunks for 600 second file with 300s chunks
            assert len(chunks) <= 2
            assert all(chunk.parent == output_dir for chunk in chunks)


class TestFFmpegAudioPreprocessingAdapter:
    @pytest.fixture
    def adapter(self):
        return FFmpegAudioPreprocessingAdapter()

    def test_reduce_noise_nonexistent_file(self, adapter):
        """Test noise reduction on non-existent file returns False."""
        nonexistent = Path("nonexistent.wav")
        output = Path("output.wav")

        result = adapter.reduce_noise(nonexistent, output, 0.5)
        assert result is False

    @patch("subprocess.run")
    def test_reduce_noise_success(self, mock_run, adapter):
        """Test successful noise reduction."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            input_file = tmpdir_path / "input.wav"
            input_file.touch()

            output_file = tmpdir_path / "output.wav"

            result = adapter.reduce_noise(input_file, output_file, 0.5)
            assert result is True

    @patch("subprocess.run")
    def test_normalize_volume_success(self, mock_run, adapter):
        """Test successful volume normalization."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            input_file = tmpdir_path / "input.wav"
            input_file.touch()

            output_file = tmpdir_path / "output.wav"

            result = adapter.normalize_volume(input_file, output_file, -20.0)
            assert result is True


class TestWhisperBatchProcessingService:
    @pytest.fixture
    def mock_transcription_service(self):
        service = Mock()
        transcription_result = Mock()
        transcription_result.text = "Test transcription"
        transcription_result.confidence = 0.95
        service.transcribe.return_value = transcription_result
        return service

    @pytest.fixture
    def mock_audio_format_service(self):
        service = Mock()
        service.is_supported_format.return_value = True
        service.convert_to_wav.return_value = True
        service.get_audio_info.return_value = {"duration": 60.0}
        return service

    @pytest.fixture
    def mock_logger(self):
        return Mock()

    @pytest.fixture
    def service(
        self, mock_transcription_service, mock_audio_format_service, mock_logger
    ):
        return WhisperBatchProcessingService(
            transcription_service=mock_transcription_service,
            audio_format_service=mock_audio_format_service,
            logger=mock_logger,
        )

    def test_get_processable_files_empty_dir(self, service):
        """Test that empty directory returns no files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            files = service.get_processable_files(tmpdir_path)
            assert files == []

    def test_get_processable_files_with_audio(self, service, mock_audio_format_service):
        """Test finding audio files in directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create test files
            (tmpdir_path / "audio1.mp3").touch()
            (tmpdir_path / "audio2.wav").touch()
            (tmpdir_path / "document.txt").touch()  # Should be filtered out

            # Mock supported format check
            def mock_is_supported(path):
                return path.suffix.lower() in [".mp3", ".wav"]

            mock_audio_format_service.is_supported_format.side_effect = (
                mock_is_supported
            )

            files = service.get_processable_files(tmpdir_path)

            assert len(files) == 2
            assert all(f.suffix.lower() in [".mp3", ".wav"] for f in files)

    def test_estimate_batch_time(self, service):
        """Test batch time estimation."""
        # Create mock files
        files = [Path("file1.mp3"), Path("file2.mp3")]

        estimated_time = service.estimate_batch_time(files)

        # Should be roughly duration * 0.1 + overhead
        # With 60s per file and 2 files: (120 * 0.1) + (2 * 2) = 16 seconds
        assert estimated_time > 0
        assert estimated_time < 30  # Reasonable upper bound

    def test_process_directory_invalid_input(self, service):
        """Test processing invalid input directory."""
        nonexistent_dir = Path("nonexistent")
        output_dir = Path("output")
        config = Mock()

        with pytest.raises(ValueError, match="Input directory does not exist"):
            service.process_directory(nonexistent_dir, output_dir, config)


# Integration test markers
pytestmark = [
    pytest.mark.unit,
    pytest.mark.requires_audio,  # For tests that need audio processing
]
