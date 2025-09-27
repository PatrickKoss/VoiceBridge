"""Tests for CLI base commands."""

from unittest.mock import Mock, patch

import pytest

from voicebridge.cli.commands.base import BaseCommands


class TestBaseCommands:
    """Test cases for BaseCommands."""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock dependencies for BaseCommands."""
        return {
            "config_repo": Mock(),
            "profile_repo": Mock(),
            "logger": Mock(),
        }

    @pytest.fixture
    def base_commands(self, mock_dependencies):
        """BaseCommands instance."""
        return BaseCommands(**mock_dependencies)

    def test_init_required_dependencies(self, base_commands, mock_dependencies):
        """Test BaseCommands initialization with required dependencies."""
        assert isinstance(base_commands, BaseCommands)
        assert base_commands.config_repo == mock_dependencies["config_repo"]
        assert base_commands.profile_repo == mock_dependencies["profile_repo"]
        assert base_commands.logger == mock_dependencies["logger"]

    def test_init_optional_services_none(self, mock_dependencies):
        """Test that optional services default to None."""
        base_commands = BaseCommands(**mock_dependencies)

        # Test a few optional services
        assert base_commands.system_service is None
        assert base_commands.daemon_service is None
        assert base_commands.transcription_orchestrator is None
        assert base_commands.session_service is None

    def test_init_with_optional_services(self, mock_dependencies):
        """Test initialization with some optional services."""
        mock_system_service = Mock()
        mock_daemon_service = Mock()

        base_commands = BaseCommands(
            system_service=mock_system_service,
            daemon_service=mock_daemon_service,
            **mock_dependencies,
        )

        assert base_commands.system_service == mock_system_service
        assert base_commands.daemon_service == mock_daemon_service

    def test_stop_audio_recorder_no_orchestrator(self, base_commands):
        """Test stopping audio recorder when no orchestrator is present."""
        # Should not raise an error
        base_commands._stop_audio_recorder()
        # Should log nothing since no orchestrator

    def test_stop_audio_recorder_with_orchestrator_no_recorder(self, base_commands):
        """Test stopping audio recorder when orchestrator has no recorder."""
        mock_orchestrator = Mock()
        mock_orchestrator.audio_recorder = None
        base_commands.transcription_orchestrator = mock_orchestrator

        # Should not raise an error
        base_commands._stop_audio_recorder()

    def test_stop_audio_recorder_with_working_recorder(self, base_commands):
        """Test stopping audio recorder successfully."""
        mock_recorder = Mock()
        mock_stop_method = Mock()
        mock_recorder.stop_current_stream = mock_stop_method

        mock_orchestrator = Mock()
        mock_orchestrator.audio_recorder = mock_recorder
        base_commands.transcription_orchestrator = mock_orchestrator

        base_commands._stop_audio_recorder()

        mock_stop_method.assert_called_once()
        base_commands.logger.info.assert_called_with("Audio recording stopped")

    def test_stop_audio_recorder_with_exception(self, base_commands):
        """Test stopping audio recorder when stop method raises exception."""
        mock_recorder = Mock()
        mock_stop_method = Mock(side_effect=Exception("Stop failed"))
        mock_recorder.stop_current_stream = mock_stop_method

        mock_orchestrator = Mock()
        mock_orchestrator.audio_recorder = mock_recorder
        base_commands.transcription_orchestrator = mock_orchestrator

        base_commands._stop_audio_recorder()

        mock_stop_method.assert_called_once()
        base_commands.logger.error.assert_called_with(
            "Failed to stop audio recording: Stop failed"
        )

    @patch("voicebridge.cli.utils.command_helpers.handle_profile_config")
    @patch("voicebridge.cli.utils.command_helpers.build_whisper_config")
    def test_build_config_basic(
        self, mock_build_whisper, mock_handle_profile, base_commands
    ):
        """Test building config with basic parameters."""
        mock_base_config = Mock()
        mock_whisper_config = {"model_name": "base", "language": "en"}

        mock_handle_profile.return_value = mock_base_config
        mock_build_whisper.return_value = mock_whisper_config

        with patch("dataclasses.replace") as mock_replace:
            mock_final_config = Mock()
            mock_replace.return_value = mock_final_config

            result = base_commands._build_config(
                model="base", language="en", temperature=0.5
            )

            assert result == mock_final_config
            mock_handle_profile.assert_called_once()
            mock_build_whisper.assert_called_once()
            mock_replace.assert_called_once_with(
                mock_base_config, **mock_whisper_config
            )

    @patch("voicebridge.cli.utils.command_helpers.handle_profile_config")
    @patch("voicebridge.cli.utils.command_helpers.build_whisper_config")
    def test_build_config_no_whisper_config(
        self, mock_build_whisper, mock_handle_profile, base_commands
    ):
        """Test building config when whisper config is None."""
        mock_base_config = Mock()

        mock_handle_profile.return_value = mock_base_config
        mock_build_whisper.return_value = None

        result = base_commands._build_config()

        assert result == mock_base_config
        mock_handle_profile.assert_called_once()
        mock_build_whisper.assert_called_once()

    @patch("voicebridge.cli.utils.command_helpers.handle_profile_config")
    @patch("voicebridge.cli.utils.command_helpers.build_whisper_config")
    def test_build_config_with_profile(
        self, mock_build_whisper, mock_handle_profile, base_commands
    ):
        """Test building config with specific profile."""
        mock_base_config = Mock()
        mock_whisper_config = {"temperature": 0.8}

        mock_handle_profile.return_value = mock_base_config
        mock_build_whisper.return_value = mock_whisper_config

        with patch("dataclasses.replace"):
            base_commands._build_config(profile="test-profile", temperature=0.8)

            mock_handle_profile.assert_called_with(
                "test-profile", base_commands.config_repo, base_commands.profile_repo
            )

    def test_all_service_attributes_initialized(self, base_commands):
        """Test that all service attributes are properly initialized."""
        # Core services
        assert hasattr(base_commands, "system_service")
        assert hasattr(base_commands, "daemon_service")

        # STT services
        assert hasattr(base_commands, "transcription_orchestrator")
        assert hasattr(base_commands, "session_service")
        assert hasattr(base_commands, "resume_service")
        assert hasattr(base_commands, "confidence_analyzer")

        # TTS services
        assert hasattr(base_commands, "tts_orchestrator")
        assert hasattr(base_commands, "tts_daemon_service")

        # Audio services
        assert hasattr(base_commands, "audio_format_service")
        assert hasattr(base_commands, "audio_preprocessing_service")
        assert hasattr(base_commands, "audio_splitting_service")

        # Processing services
        assert hasattr(base_commands, "batch_processing_service")
        assert hasattr(base_commands, "export_service")
        assert hasattr(base_commands, "performance_service")

        # Advanced services
        assert hasattr(base_commands, "vocabulary_service")
        assert hasattr(base_commands, "vocabulary_management_service")
        assert hasattr(base_commands, "webhook_service")

    def test_dependency_injection_pattern(self, mock_dependencies):
        """Test that the class follows dependency injection pattern."""
        # Should be able to create with all services
        all_services = {
            "system_service": Mock(),
            "daemon_service": Mock(),
            "transcription_orchestrator": Mock(),
            "session_service": Mock(),
            "tts_orchestrator": Mock(),
            "audio_format_service": Mock(),
            "export_service": Mock(),
        }

        base_commands = BaseCommands(**mock_dependencies, **all_services)

        for service_name, service_mock in all_services.items():
            assert getattr(base_commands, service_name) == service_mock
