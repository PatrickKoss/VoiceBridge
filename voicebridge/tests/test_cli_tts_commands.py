"""Tests for CLI TTS commands."""

from unittest.mock import Mock, patch

import pytest

from voicebridge.cli.commands.tts_commands import TTSCommands


class TestTTSCommands:
    """Test cases for TTSCommands."""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock dependencies for TTSCommands."""
        return {
            "config_repo": Mock(),
            "profile_repo": Mock(),
            "logger": Mock(),
            "tts_orchestrator": Mock(),
            "tts_daemon_service": Mock(),
        }

    @pytest.fixture
    def tts_commands(self, mock_dependencies):
        """TTSCommands instance."""
        return TTSCommands(**mock_dependencies)

    def test_init(self, mock_dependencies):
        """Test TTSCommands initialization."""
        commands = TTSCommands(**mock_dependencies)
        assert isinstance(commands, TTSCommands)
        assert commands.tts_orchestrator == mock_dependencies["tts_orchestrator"]
        assert commands.tts_daemon_service == mock_dependencies["tts_daemon_service"]

    def test_tts_generate_basic(self, tts_commands):
        """Test basic TTS generation."""
        with patch("voicebridge.cli.commands.tts_commands.typer.echo") as mock_echo:
            tts_commands.tts_generate(
                text="Hello world",
                voice="en-Alice_woman",
                streaming=False,
                output_file="output.wav",
                auto_play=True,
                cfg_scale=None,
                inference_steps=None,
                sample_rate=None,
                use_gpu=None,
            )

        mock_echo.assert_called()

    def test_tts_generate_streaming(self, tts_commands):
        """Test TTS generation with streaming."""
        with patch("voicebridge.cli.commands.tts_commands.typer.echo") as mock_echo:
            tts_commands.tts_generate(
                text="Streaming test",
                voice="en-Bob_man",
                streaming=True,
                output_file=None,
                auto_play=False,
                cfg_scale=1.5,
                inference_steps=20,
                sample_rate=22050,
                use_gpu=True,
            )

        mock_echo.assert_called()

    def test_tts_listen_clipboard(self, tts_commands):
        """Test TTS clipboard listening setup."""
        with patch("voicebridge.cli.commands.tts_commands.typer.echo") as mock_echo:
            with patch("time.sleep", side_effect=KeyboardInterrupt):
                try:
                    tts_commands.tts_listen_clipboard(
                        voice="en-Charlie_woman",
                        streaming=False,
                        auto_play=True,
                        output_file="clipboard.wav",
                    )
                except KeyboardInterrupt:
                    pass  # Expected due to sleep mock

        mock_echo.assert_called()

    def test_tts_listen_selection(self, tts_commands):
        """Test TTS selection listening setup."""
        with patch("voicebridge.cli.commands.tts_commands.typer.echo") as mock_echo:
            with patch("time.sleep", side_effect=KeyboardInterrupt):
                try:
                    tts_commands.tts_listen_selection(
                        voice="en-David_man", streaming=True, auto_play=False
                    )
                except KeyboardInterrupt:
                    pass  # Expected due to sleep mock

        mock_echo.assert_called()

    def test_tts_daemon_start(self, tts_commands):
        """Test TTS daemon start error handling."""
        import typer

        with patch("voicebridge.cli.commands.tts_commands.typer.echo"):
            with pytest.raises(typer.Exit):
                tts_commands.tts_daemon_start(
                    voice="en-Emma_woman",
                    mode="clipboard",
                    streaming=False,
                    auto_play=True,
                    background=False,
                )

    def test_tts_daemon_stop(self, tts_commands):
        """Test TTS daemon stop."""
        mock_daemon = Mock()
        mock_daemon.is_daemon_running.return_value = (
            True  # Make it think daemon is running
        )
        mock_daemon.stop_daemon.return_value = True  # Make stop succeed
        tts_commands.tts_daemon_service = mock_daemon

        with patch("voicebridge.cli.commands.tts_commands.typer.echo") as mock_echo:
            tts_commands.tts_daemon_stop()

        mock_daemon.stop_daemon.assert_called_once()
        mock_echo.assert_called()

    def test_tts_daemon_status(self, tts_commands):
        """Test TTS daemon status."""
        mock_daemon = Mock()
        mock_daemon.get_daemon_status.return_value = {
            "running": True,
            "pid": 12345,
            "mode": "clipboard",
            "voice": "en-Alice_woman",
            "uptime": "10m 5s",
            "stats": {
                "requests_processed": 50,
                "audio_generated_seconds": 120.5,
                "errors": 2,
            },
        }
        tts_commands.tts_daemon_service = mock_daemon

        with patch("voicebridge.cli.commands.tts_commands.typer.echo") as mock_echo:
            tts_commands.tts_daemon_status()

        mock_daemon.get_daemon_status.assert_called_once()
        mock_echo.assert_called()

    def test_tts_list_voices(self, tts_commands):
        """Test TTS voice listing."""
        mock_orchestrator = Mock()
        mock_voice_info = Mock()
        mock_voice_info.display_name = "Alice Woman"
        mock_voice_info.language = "en"
        mock_voice_info.gender = "female"
        mock_voice_info.file_path = "/path/to/en-Alice_woman.wav"

        mock_orchestrator.list_available_voices.return_value = {
            "en-Alice_woman": mock_voice_info,
            "en-Bob_man": mock_voice_info,
            "es-Carmen_woman": mock_voice_info,
        }
        tts_commands.tts_orchestrator = mock_orchestrator

        with patch("voicebridge.cli.commands.tts_commands.typer.echo") as mock_echo:
            tts_commands.tts_list_voices()

        mock_orchestrator.list_available_voices.assert_called_once()
        mock_echo.assert_called()

    def test_tts_config_show(self, tts_commands):
        """Test TTS config show."""
        mock_config = Mock()
        mock_tts_config = Mock()
        mock_tts_config.default_voice = "en-Alice_woman"
        mock_tts_config.voice_samples_path = "/path/to/voices"
        mock_tts_config.model_path = "/path/to/model"
        mock_tts_config.sample_rate = 22050
        mock_tts_config.cfg_scale = 1.0
        mock_tts_config.inference_steps = 20
        mock_tts_config.auto_play = True
        mock_tts_config.use_gpu = False
        mock_tts_config.tts_generate_key = "f12"
        mock_tts_config.tts_stop_key = "escape"
        mock_tts_config.output_mode = "play"
        mock_tts_config.streaming_mode = "streaming"
        mock_config.tts_config = mock_tts_config
        tts_commands.config_repo.load.return_value = mock_config

        with patch("voicebridge.cli.commands.tts_commands.typer.echo") as mock_echo:
            tts_commands.tts_config_show()

        tts_commands.config_repo.load.assert_called_once()
        mock_echo.assert_called()

    def test_tts_config_set_basic(self, tts_commands):
        """Test basic TTS config set."""
        with patch("voicebridge.cli.commands.tts_commands.typer.echo") as mock_echo:
            tts_commands.tts_config_set(
                default_voice="en-Bob_man",
                voice_samples_path="/path/to/voices",
                model_path="/path/to/model",
                sample_rate=24000,
                cfg_scale=1.2,
                inference_steps=25,
                auto_play=True,
                use_gpu=True,
                generate_key="f11",
                stop_key="f12",
                output_mode="play",
                streaming_mode="streaming",
            )

        tts_commands.config_repo.save.assert_called_once()
        mock_echo.assert_called()

    def test_tts_generate_no_orchestrator(self, mock_dependencies):
        """Test TTS generate when orchestrator is None."""
        import typer

        mock_dependencies["tts_orchestrator"] = None
        commands = TTSCommands(**mock_dependencies)

        with patch("voicebridge.cli.commands.tts_commands.typer.echo"):
            with pytest.raises(typer.Exit):
                commands.tts_generate(
                    text="Test text",
                    voice=None,
                    streaming=False,
                    output_file=None,
                    auto_play=True,
                    cfg_scale=None,
                    inference_steps=None,
                    sample_rate=None,
                    use_gpu=None,
                )

    def test_tts_daemon_stop_no_service(self, mock_dependencies):
        """Test daemon stop when service is None."""
        import typer

        mock_dependencies["tts_daemon_service"] = None
        commands = TTSCommands(**mock_dependencies)

        with patch("voicebridge.cli.commands.tts_commands.typer.echo"):
            with pytest.raises(typer.Exit):
                commands.tts_daemon_stop()

    def test_tts_daemon_status_no_service(self, mock_dependencies):
        """Test daemon status when service is None."""
        import typer

        mock_dependencies["tts_daemon_service"] = None
        commands = TTSCommands(**mock_dependencies)

        with patch("voicebridge.cli.commands.tts_commands.typer.echo"):
            with pytest.raises(typer.Exit):
                commands.tts_daemon_status()

    def test_tts_list_voices_no_orchestrator(self, mock_dependencies):
        """Test list voices when orchestrator is None."""
        import typer

        mock_dependencies["tts_orchestrator"] = None
        commands = TTSCommands(**mock_dependencies)

        with patch("voicebridge.cli.commands.tts_commands.typer.echo"):
            with pytest.raises(typer.Exit):
                commands.tts_list_voices()

    def test_dependency_injection(self, mock_dependencies):
        """Test that commands work with dependency injection."""
        commands = TTSCommands(**mock_dependencies)

        # Test that all required dependencies are accessible
        assert commands.config_repo is not None
        assert commands.profile_repo is not None
        assert commands.logger is not None
        assert commands.tts_orchestrator is not None
        assert commands.tts_daemon_service is not None

    def test_command_methods_exist(self, tts_commands):
        """Test that all expected command methods exist."""
        assert hasattr(tts_commands, "tts_generate")
        assert hasattr(tts_commands, "tts_listen_clipboard")
        assert hasattr(tts_commands, "tts_listen_selection")
        assert hasattr(tts_commands, "tts_daemon_start")
        assert hasattr(tts_commands, "tts_daemon_stop")
        assert hasattr(tts_commands, "tts_daemon_status")
        assert hasattr(tts_commands, "tts_list_voices")
        assert hasattr(tts_commands, "tts_config_show")
        assert hasattr(tts_commands, "tts_config_set")
        assert callable(tts_commands.tts_generate)
        assert callable(tts_commands.tts_daemon_start)

    def test_daemon_modes_coverage(self, tts_commands):
        """Test daemon start with different modes."""
        modes = ["clipboard", "selection"]

        # Set up daemon service mock to not be running
        mock_daemon = Mock()
        mock_daemon.is_daemon_running.return_value = False
        mock_daemon.start_daemon.return_value = True
        mock_daemon.is_daemon_running.side_effect = [
            False,
            True,
            False,
            True,
        ]  # alternating for while loop
        tts_commands.tts_daemon_service = mock_daemon

        with patch("voicebridge.cli.commands.tts_commands.typer.echo") as mock_echo:
            with patch("time.sleep", side_effect=KeyboardInterrupt):
                for mode in modes:
                    try:
                        tts_commands.tts_daemon_start(
                            voice="en-Test_voice",
                            mode=mode,
                            streaming=False,
                            auto_play=True,
                            background=False,
                        )
                    except KeyboardInterrupt:
                        pass  # Expected due to sleep mock

            # Should have called echo for each mode
            assert mock_echo.call_count >= len(modes)

    def test_config_edge_cases(self, tts_commands):
        """Test TTS config with edge case values."""
        with patch("voicebridge.cli.commands.tts_commands.typer.echo"):
            # Test with boolean False values
            tts_commands.tts_config_set(
                default_voice="en-Test_voice",
                voice_samples_path="",
                model_path="",
                sample_rate=8000,  # Low sample rate
                cfg_scale=0.1,  # Low CFG scale
                inference_steps=1,  # Minimal steps
                auto_play=False,
                use_gpu=False,
                generate_key="escape",
                stop_key="space",
                output_mode="save",
                streaming_mode="non_streaming",
            )

        tts_commands.config_repo.save.assert_called_once()
