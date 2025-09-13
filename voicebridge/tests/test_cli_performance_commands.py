from unittest.mock import Mock, patch

from cli.app import create_app
from typer.testing import CliRunner


class TestCLICommands:
    """Test that new CLI commands are properly registered."""

    @patch("cli.commands.CLICommands")
    def test_gpu_status_command_exists(self, mock_commands_class):
        """Test that gpu status command is registered."""
        mock_commands = Mock()
        mock_commands_class.return_value = mock_commands

        app = create_app(mock_commands)
        runner = CliRunner()

        # Test that command exists - it should call the method
        with patch.object(mock_commands, "gpu_status"):
            runner.invoke(app, ["gpu", "status"])
            mock_commands.gpu_status.assert_called_once()

    @patch("cli.commands.CLICommands")
    def test_performance_stats_command_exists(self, mock_commands_class):
        """Test that performance stats command is registered."""
        mock_commands = Mock()
        mock_commands_class.return_value = mock_commands

        app = create_app(mock_commands)
        runner = CliRunner()

        with patch.object(mock_commands, "performance_stats"):
            runner.invoke(app, ["performance", "stats"])
            mock_commands.performance_stats.assert_called_once()

    @patch("cli.commands.CLICommands")
    def test_sessions_list_command_exists(self, mock_commands_class):
        """Test that sessions list command is registered."""
        mock_commands = Mock()
        mock_commands_class.return_value = mock_commands

        app = create_app(mock_commands)
        runner = CliRunner()

        with patch.object(mock_commands, "sessions_list"):
            runner.invoke(app, ["sessions", "list"])
            mock_commands.sessions_list.assert_called_once()

    @patch("cli.commands.CLICommands")
    def test_transcribe_command_exists(self, mock_commands_class):
        """Test that transcribe command is registered."""
        mock_commands = Mock()
        mock_commands_class.return_value = mock_commands

        app = create_app(mock_commands)
        runner = CliRunner()

        with patch.object(mock_commands, "transcribe_file"):
            runner.invoke(app, ["transcribe", "test.wav"])
            mock_commands.transcribe_file.assert_called_once_with(
                "test.wav", None, None, None, 0.0, "txt"
            )


# Additional tests can be added here for specific command behavior if needed
