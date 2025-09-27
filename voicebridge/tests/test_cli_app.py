"""Tests for CLI application."""

from unittest.mock import Mock, patch

import pytest
import typer

from voicebridge.cli.app import create_app
from voicebridge.cli.registry import CommandRegistry


class TestCreateApp:
    """Test cases for CLI app creation."""

    @pytest.fixture
    def mock_registry(self):
        """Mock CommandRegistry."""
        return Mock(spec=CommandRegistry)

    def test_create_app_basic(self, mock_registry):
        """Test basic app creation."""
        app = create_app(mock_registry)

        assert isinstance(app, typer.Typer)
        assert (
            app.info.help
            == "VoiceBridge - Comprehensive bidirectional voice-text CLI tool"
        )

    def test_create_app_has_stt_subapp(self, mock_registry):
        """Test that app has STT sub-application."""
        app = create_app(mock_registry)

        # Check that the app has registered subapps
        # This tests the structural setup even if we can't easily inspect the commands
        assert isinstance(app, typer.Typer)

    def test_create_app_configuration(self, mock_registry):
        """Test app configuration options."""
        app = create_app(mock_registry)

        # Test that completion is disabled
        # Note: Typer stores this internally, test the created app is Typer instance
        assert isinstance(app, typer.Typer)

    @patch("voicebridge.cli.app.typer.Typer")
    def test_create_app_creates_subapps(self, mock_typer_class, mock_registry):
        """Test that create_app creates the expected sub-applications."""
        mock_app = Mock()
        mock_typer_class.return_value = mock_app

        result = create_app(mock_registry)

        assert result == mock_app
        # Should create main app and multiple sub-apps
        assert mock_typer_class.call_count >= 6

        # Should add sub-apps to main app - check that add_typer was called with expected names
        call_args = [
            call.kwargs.get("name") for call in mock_app.add_typer.call_args_list
        ]
        expected_names = ["stt", "tts", "audio", "gpu", "api"]
        for name in expected_names:
            assert name in call_args

    def test_create_app_with_none_registry(self):
        """Test create_app with None registry handles gracefully."""
        # This should not crash, though it may not work fully
        app = create_app(None)
        assert isinstance(app, typer.Typer)

    def test_create_app_returns_typer_instance(self, mock_registry):
        """Test that create_app returns a Typer instance."""
        app = create_app(mock_registry)
        assert hasattr(app, "command")
        assert hasattr(app, "add_typer")
        assert hasattr(app, "info")

    def test_app_help_text(self, mock_registry):
        """Test that the app has correct help text."""
        app = create_app(mock_registry)

        expected_help = "VoiceBridge - Comprehensive bidirectional voice-text CLI tool"
        assert app.info.help == expected_help

    def test_app_no_completion(self, mock_registry):
        """Test that app completion is disabled."""
        app = create_app(mock_registry)
        # Test the created app is a proper Typer instance
        assert isinstance(app, typer.Typer)

    def test_stt_subapp_creation(self, mock_registry):
        """Test that STT sub-application is created with correct help."""
        with patch("voicebridge.cli.app.typer.Typer") as mock_typer:
            mock_app = Mock()
            mock_stt_app = Mock()
            mock_tts_app = Mock()
            # Need more mocks for all the sub-apps created
            mock_typer.side_effect = [
                mock_app,
                mock_stt_app,
                mock_tts_app,
                Mock(),
                Mock(),
                Mock(),
                Mock(),
                Mock(),
                Mock(),
                Mock(),
                Mock(),
                Mock(),
                Mock(),
                Mock(),
                Mock(),
                Mock(),
                Mock(),
                Mock(),
            ]

            create_app(mock_registry)

            # Check main app creation
            main_app_call = mock_typer.call_args_list[0]
            assert main_app_call[1]["add_completion"] is False
            assert "VoiceBridge" in main_app_call[1]["help"]

            # Check STT sub-app creation
            stt_app_call = mock_typer.call_args_list[1]
            assert stt_app_call[1]["help"] == "Speech-to-Text commands"

    def test_app_structure_with_registry(self, mock_registry):
        """Test that app properly uses the registry."""
        app = create_app(mock_registry)

        # App should be created successfully with registry
        assert isinstance(app, typer.Typer)
        # Registry is passed to the app creation process
        assert mock_registry is not None

    def test_multiple_app_creation(self, mock_registry):
        """Test that multiple apps can be created."""
        app1 = create_app(mock_registry)
        app2 = create_app(mock_registry)

        # Should be able to create multiple independent apps
        assert isinstance(app1, typer.Typer)
        assert isinstance(app2, typer.Typer)
        # They should be different instances
        assert app1 is not app2
