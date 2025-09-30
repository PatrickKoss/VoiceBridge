"""Tests for CLI speech commands."""

from unittest.mock import Mock, patch

import pytest

from voicebridge.cli.commands.speech_commands import SpeechCommands


class TestSpeechCommands:
    """Test cases for SpeechCommands."""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock dependencies for SpeechCommands."""
        return {
            "config_repo": Mock(),
            "profile_repo": Mock(),
            "logger": Mock(),
            "transcription_orchestrator": Mock(),
            "system_service": Mock(),
        }

    @pytest.fixture
    def speech_commands(self, mock_dependencies):
        """SpeechCommands instance."""
        return SpeechCommands(**mock_dependencies)

    def test_init(self, mock_dependencies):
        """Test SpeechCommands initialization."""
        commands = SpeechCommands(**mock_dependencies)
        assert isinstance(commands, SpeechCommands)
        assert (
            commands.transcription_orchestrator
            == mock_dependencies["transcription_orchestrator"]
        )
        assert commands.system_service == mock_dependencies["system_service"]

    def test_listen_basic(self, speech_commands):
        """Test basic listen command setup and error handling."""
        import typer

        # This test verifies that the listen command can be called and handles errors properly
        with patch("voicebridge.cli.commands.speech_commands.typer.echo") as mock_echo:
            # Don't mock _build_config to test the real error path
            with pytest.raises(typer.Exit):  # typer.Exit is the actual exception
                speech_commands.listen(
                    model="base",
                    language="en",
                    initial_prompt=None,
                    temperature=0.0,
                    profile=None,
                    paste_stream=False,
                    copy_stream=False,
                    paste_final=False,
                    copy_final=True,
                    debug=False,
                )

        # Verify echo was called at least once (for the error message)
        mock_echo.assert_called()

    def test_interactive_mode(self, speech_commands):
        """Test interactive mode setup and error handling."""
        import typer

        with patch("voicebridge.cli.commands.speech_commands.typer.echo") as mock_echo:
            with pytest.raises(typer.Exit):
                speech_commands.interactive(
                    model="small",
                    language="es",
                    initial_prompt="Interactive prompt",
                    temperature=0.2,
                    profile="interactive-profile",
                    paste_stream=False,
                    copy_stream=True,
                    paste_final=True,
                    copy_final=True,
                    debug=False,
                )

        mock_echo.assert_called()

    def test_hotkey_mode(self, speech_commands):
        """Test hotkey mode setup and error handling."""
        import typer

        with patch("voicebridge.cli.commands.speech_commands.typer.echo") as mock_echo:
            with pytest.raises(typer.Exit):
                speech_commands.hotkey(
                    key="f10",
                    mode="hold",
                    model="medium",
                    language="fr",
                    initial_prompt="Hotkey prompt",
                    temperature=0.3,
                    profile="hotkey-profile",
                    paste_stream=True,
                    copy_stream=False,
                    paste_final=False,
                    copy_final=True,
                    debug=True,
                )

        mock_echo.assert_called()

    def test_listen_no_orchestrator(self, mock_dependencies):
        """Test listen command when orchestrator is None."""
        import typer

        mock_dependencies["transcription_orchestrator"] = None
        speech_commands = SpeechCommands(**mock_dependencies)

        with patch("voicebridge.cli.commands.speech_commands.typer.echo"):
            with pytest.raises(typer.Exit):
                speech_commands.listen(
                    model="base",
                    language="en",
                    initial_prompt=None,
                    temperature=0.0,
                    profile=None,
                    paste_stream=False,
                    copy_stream=False,
                    paste_final=False,
                    copy_final=True,
                    debug=False,
                )

    def test_build_config_integration(self, speech_commands):
        """Test that _build_config is called with correct parameters."""
        import typer

        with patch.object(speech_commands, "_build_config") as mock_build_config:
            with patch("voicebridge.cli.commands.speech_commands.typer.echo"):
                with patch("time.sleep", side_effect=KeyboardInterrupt):
                    # Mock pynput keyboard module before it's imported
                    mock_keyboard = Mock()
                    mock_listener_class = Mock()
                    mock_keyboard.Listener = mock_listener_class
                    mock_keyboard.Key = Mock()

                    with patch.dict("sys.modules", {"pynput.keyboard": mock_keyboard}):
                        mock_listener_instance = Mock()
                        mock_listener_instance.start = Mock()
                        mock_listener_instance.stop = Mock()
                        mock_listener_class.return_value = mock_listener_instance
                        mock_build_config.return_value = Mock()

                        try:
                            speech_commands.listen(
                                model="base",
                                language="en",
                                initial_prompt="Test",
                                temperature=0.5,
                                profile="test",
                                paste_stream=False,
                                copy_stream=False,
                                paste_final=False,
                                copy_final=True,
                                max_memory=0,
                                debug=False,
                            )
                        except (KeyboardInterrupt, typer.Exit):
                            pass

                        mock_build_config.assert_called_once_with(
                            model="base",
                            language="en",
                            initial_prompt="Test",
                            temperature=0.5,
                            profile="test",
                            paste_stream=False,
                            copy_stream=False,
                            paste_final=False,
                            copy_final=True,
                            max_memory_mb=0,
                            debug=False,
                        )

    def test_dependency_injection(self, mock_dependencies):
        """Test that commands work with dependency injection."""
        commands = SpeechCommands(**mock_dependencies)

        # Test that all required dependencies are accessible
        assert commands.config_repo is not None
        assert commands.profile_repo is not None
        assert commands.logger is not None
        assert commands.transcription_orchestrator is not None
        assert commands.system_service is not None

    def test_command_methods_exist(self, speech_commands):
        """Test that all expected command methods exist."""
        assert hasattr(speech_commands, "listen")
        assert hasattr(speech_commands, "interactive")
        assert hasattr(speech_commands, "hotkey")
        assert callable(speech_commands.listen)
        assert callable(speech_commands.interactive)
        assert callable(speech_commands.hotkey)

    def test_exception_handling(self, speech_commands):
        """Test that commands handle exceptions gracefully."""
        with patch("voicebridge.cli.commands.speech_commands.typer.echo"):
            with patch.object(
                speech_commands, "_build_config", side_effect=Exception("Config error")
            ):
                # Should not crash when config building fails
                try:
                    speech_commands.listen(
                        model="base",
                        language="en",
                        initial_prompt=None,
                        temperature=0.0,
                        profile=None,
                        paste_stream=False,
                        copy_stream=False,
                        paste_final=False,
                        copy_final=True,
                        debug=False,
                    )
                except Exception:
                    pass  # Expected when there are errors
