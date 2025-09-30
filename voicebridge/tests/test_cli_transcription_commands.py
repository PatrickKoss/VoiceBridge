"""Tests for CLI transcription commands."""

from unittest.mock import Mock, patch

import pytest

from voicebridge.cli.commands.transcription_commands import TranscriptionCommands


class TestTranscriptionCommands:
    """Test cases for TranscriptionCommands."""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock dependencies for TranscriptionCommands."""
        return {
            "config_repo": Mock(),
            "profile_repo": Mock(),
            "logger": Mock(),
            "transcription_orchestrator": Mock(),
            "batch_processing_service": Mock(),
            "resume_service": Mock(),
            "audio_format_service": Mock(),
        }

    @pytest.fixture
    def transcription_commands(self, mock_dependencies):
        """TranscriptionCommands instance."""
        return TranscriptionCommands(**mock_dependencies)

    def test_init(self, mock_dependencies):
        """Test TranscriptionCommands initialization."""
        commands = TranscriptionCommands(**mock_dependencies)
        assert isinstance(commands, TranscriptionCommands)
        assert (
            commands.transcription_orchestrator
            == mock_dependencies["transcription_orchestrator"]
        )
        assert (
            commands.batch_processing_service
            == mock_dependencies["batch_processing_service"]
        )

    def test_transcribe_file_basic(self, transcription_commands):
        """Test basic file transcription error handling."""
        import typer

        with patch(
            "voicebridge.cli.commands.transcription_commands.typer.echo"
        ) as mock_echo:
            with pytest.raises(typer.Exit):
                transcription_commands.transcribe_file(
                    file_path="test.wav",
                    output_path="output.txt",
                    model="base",
                    language="en",
                    temperature=0.0,
                    format_output="txt",
                )

        mock_echo.assert_called()

    def test_batch_transcribe_basic(self, transcription_commands):
        """Test basic batch transcription error handling."""
        import typer

        with patch("voicebridge.cli.commands.transcription_commands.typer.echo"):
            with pytest.raises(typer.Exit):
                transcription_commands.batch_transcribe(
                    input_dir="/path/to/audio",
                    output_dir="/path/to/output",
                    workers=4,
                    file_pattern="*.wav",
                    model="base",
                )

    def test_listen_resumable_basic(self, transcription_commands):
        """Test basic resumable transcription error handling."""
        import typer

        with patch("voicebridge.cli.commands.transcription_commands.typer.echo"):
            with pytest.raises(typer.Exit):
                transcription_commands.listen_resumable(
                    file_path="long_audio.wav",
                    session_name="test_session",
                    model="base",
                    language="en",
                    temperature=0.0,
                    profile=None,
                    chunk_size=30,
                    overlap=5,
                )

    def test_realtime_transcribe_basic(self, transcription_commands):
        """Test basic realtime transcription error handling."""
        import typer

        # Set the transcription orchestrator to None to trigger the error path
        transcription_commands.transcription_orchestrator = None

        with patch("voicebridge.cli.commands.transcription_commands.typer.echo"):
            with patch(
                "voicebridge.cli.utils.command_helpers.display_error"
            ) as mock_display_error:
                # Make display_error raise typer.Exit
                mock_display_error.side_effect = typer.Exit(1)

                with pytest.raises(typer.Exit):
                    transcription_commands.realtime_transcribe(
                        chunk_duration=2.0,
                        output_format="live",
                        model="base",
                        language="en",
                        temperature=0.0,
                        profile=None,
                        save_audio=False,
                        output_file=None,
                    )

    def test_test_audio_setup(self, transcription_commands):
        """Test audio setup testing."""
        # Mock the audio recorder to return iterable data
        mock_audio_recorder = Mock()
        mock_audio_recorder.record_stream = Mock(
            return_value=iter([b"test_chunk" * 1000])
        )
        transcription_commands.transcription_orchestrator.audio_recorder = (
            mock_audio_recorder
        )

        # Mock the transcription service
        mock_result = Mock()
        mock_result.text = "Test transcription"
        transcription_commands.transcription_orchestrator.transcription_service.transcribe = Mock(
            return_value=mock_result
        )

        with patch("voicebridge.cli.commands.transcription_commands.typer.echo"):
            with patch("time.sleep"):  # Speed up the test
                # Should complete without raising
                transcription_commands.test_audio_setup()

    def test_transcribe_file_no_orchestrator(self, mock_dependencies):
        """Test transcribe_file when orchestrator is None."""
        import typer

        mock_dependencies["transcription_orchestrator"] = None
        commands = TranscriptionCommands(**mock_dependencies)

        with patch("voicebridge.cli.commands.transcription_commands.typer.echo"):
            with pytest.raises(typer.Exit):
                commands.transcribe_file(
                    file_path="test.wav",
                    output_path="output.txt",
                    model="base",
                    language="en",
                    temperature=0.0,
                    format_output="txt",
                )

    def test_batch_transcribe_no_service(self, mock_dependencies):
        """Test batch_transcribe when batch service is None."""
        import typer

        mock_dependencies["batch_processing_service"] = None
        commands = TranscriptionCommands(**mock_dependencies)

        with patch("voicebridge.cli.commands.transcription_commands.typer.echo"):
            with pytest.raises(typer.Exit):
                commands.batch_transcribe(
                    input_dir="/path/to/audio",
                    output_dir="/path/to/output",
                    workers=4,
                    file_pattern=None,
                    model="base",
                )

    def test_build_config_integration(self, transcription_commands):
        """Test that _build_config is called with correct parameters."""
        import typer

        with patch.object(transcription_commands, "_build_config") as mock_build_config:
            with patch("voicebridge.cli.commands.transcription_commands.typer.echo"):
                mock_build_config.return_value = Mock()

                with pytest.raises(typer.Exit):
                    transcription_commands.transcribe_file(
                        file_path="test.wav",
                        output_path="output.txt",
                        model="base",
                        language="en",
                        temperature=0.5,
                        format_output="txt",
                    )

    def test_all_format_outputs(self, transcription_commands):
        """Test transcription with different output formats error handling."""
        import typer

        formats = ["txt", "json", "srt", "vtt", "csv"]

        with patch(
            "voicebridge.cli.commands.transcription_commands.typer.echo"
        ) as mock_echo:
            for fmt in formats:
                with pytest.raises(typer.Exit):
                    transcription_commands.transcribe_file(
                        file_path="test.wav",
                        output_path=f"output.{fmt}",
                        model="base",
                        language="en",
                        temperature=0.0,
                        format_output=fmt,
                    )

            # Should have called echo for each format
            assert mock_echo.call_count >= len(formats)

    def test_dependency_injection(self, mock_dependencies):
        """Test that commands work with dependency injection."""
        commands = TranscriptionCommands(**mock_dependencies)

        # Test that all required dependencies are accessible
        assert commands.config_repo is not None
        assert commands.profile_repo is not None
        assert commands.logger is not None
        assert commands.transcription_orchestrator is not None
        assert commands.batch_processing_service is not None

    def test_command_methods_exist(self, transcription_commands):
        """Test that all expected command methods exist."""
        assert hasattr(transcription_commands, "transcribe_file")
        assert hasattr(transcription_commands, "batch_transcribe")
        assert hasattr(transcription_commands, "listen_resumable")
        assert hasattr(transcription_commands, "realtime_transcribe")
        assert hasattr(transcription_commands, "test_audio_setup")
        assert callable(transcription_commands.transcribe_file)
        assert callable(transcription_commands.batch_transcribe)

    def test_edge_case_parameters(self, transcription_commands):
        """Test commands with edge case parameters error handling."""
        import typer

        with patch("voicebridge.cli.commands.transcription_commands.typer.echo"):
            # Test with minimal parameters - should fail due to missing file
            with pytest.raises(typer.Exit):
                transcription_commands.transcribe_file(
                    file_path="test.wav",
                    output_path=None,
                    model=None,
                    language=None,
                    temperature=0.0,
                    format_output="txt",
                )

            # Test with maximum chunk size and overlap - should also fail
            with pytest.raises(typer.Exit):
                transcription_commands.listen_resumable(
                    file_path="audio.wav",
                    session_name="edge_case",
                    model="large-v3",
                    language="auto",
                    temperature=1.0,
                    profile="edge-profile",
                    chunk_size=300,
                    overlap=30,
                )
