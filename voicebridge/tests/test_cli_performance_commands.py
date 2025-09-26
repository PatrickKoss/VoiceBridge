#!/usr/bin/env python3
"""Unit tests for CLI command structure and command group integration."""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from voicebridge.cli.commands.api_commands import APICommands
from voicebridge.cli.commands.audio_commands import AudioCommands
from voicebridge.cli.commands.system_commands import SystemCommands
from voicebridge.cli.commands.transcription_commands import TranscriptionCommands
from voicebridge.cli.registry import CommandRegistry


class TestCLICommands(unittest.TestCase):
    """Test that CLI command groups are properly working."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock dependencies to match BaseCommands constructor
        self.mock_services = {
            "config_repo": Mock(),
            "profile_repo": Mock(),
            "logger": Mock(),
            "system_service": Mock(),
            "transcription_orchestrator": Mock(),
            "audio_format_service": Mock(),
            "session_service": Mock(),
            "performance_service": Mock(),
        }

    def test_gpu_status_command_exists(self):
        """Test that gpu status command method exists on SystemCommands."""
        system_commands = SystemCommands(**self.mock_services)
        self.assertTrue(hasattr(system_commands, "gpu_status"))
        self.assertTrue(callable(system_commands.gpu_status))

    def test_performance_stats_command_exists(self):
        """Test that performance stats command method exists on SystemCommands."""
        system_commands = SystemCommands(**self.mock_services)
        self.assertTrue(hasattr(system_commands, "performance_stats"))
        self.assertTrue(callable(system_commands.performance_stats))

    def test_sessions_list_command_exists(self):
        """Test that sessions list command method exists on SystemCommands."""
        system_commands = SystemCommands(**self.mock_services)
        self.assertTrue(hasattr(system_commands, "sessions_list"))
        self.assertTrue(callable(system_commands.sessions_list))

    def test_transcribe_command_exists(self):
        """Test that transcribe command method exists on TranscriptionCommands."""
        transcription_commands = TranscriptionCommands(**self.mock_services)
        self.assertTrue(hasattr(transcription_commands, "transcribe_file"))
        self.assertTrue(callable(transcription_commands.transcribe_file))

    def test_audio_info_command_exists(self):
        """Test that audio info command method exists on AudioCommands."""
        audio_commands = AudioCommands(**self.mock_services)
        self.assertTrue(hasattr(audio_commands, "audio_info"))
        self.assertTrue(callable(audio_commands.audio_info))

    def test_api_status_command_exists(self):
        """Test that API status command method exists on APICommands."""
        api_commands = APICommands(**self.mock_services)
        self.assertTrue(hasattr(api_commands, "api_status"))
        self.assertTrue(callable(api_commands.api_status))

    def test_command_registry_instantiation(self):
        """Test that CommandRegistry can be created and provides expected command groups."""
        registry = CommandRegistry(**self.mock_services)

        # Test that all expected command groups are available
        expected_groups = [
            "speech",
            "transcription",
            "tts",
            "audio",
            "system",
            "config",
            "export",
            "advanced",
            "api",
        ]

        available_groups = registry.list_command_groups()

        for group in expected_groups:
            self.assertIn(
                group, available_groups, f"Command group '{group}' should be available"
            )

    def test_command_groups_can_be_retrieved(self):
        """Test that all command groups can be instantiated through registry."""
        registry = CommandRegistry(**self.mock_services)

        # Test each command group can be retrieved
        test_groups = ["system", "transcription", "audio", "api"]

        for group_name in test_groups:
            try:
                command_group = registry.get_command_group(group_name)
                self.assertIsNotNone(
                    command_group, f"Command group '{group_name}' should instantiate"
                )
            except Exception as e:
                self.fail(f"Command group '{group_name}' failed to instantiate: {e}")

    def test_system_commands_methods_available(self):
        """Test that SystemCommands has all expected methods."""
        system_commands = SystemCommands(**self.mock_services)

        expected_methods = [
            "gpu_status",
            "gpu_benchmark",
            "performance_stats",
            "sessions_list",
            "sessions_resume",
            "sessions_cleanup",
            "sessions_delete",
        ]

        for method_name in expected_methods:
            self.assertTrue(
                hasattr(system_commands, method_name),
                f"SystemCommands should have method '{method_name}'",
            )

    def test_transcription_commands_methods_available(self):
        """Test that TranscriptionCommands has all expected methods."""
        transcription_commands = TranscriptionCommands(**self.mock_services)

        expected_methods = [
            "transcribe_file",
            "batch_transcribe",
            "listen_resumable",
            "realtime_transcribe",
        ]

        for method_name in expected_methods:
            self.assertTrue(
                hasattr(transcription_commands, method_name),
                f"TranscriptionCommands should have method '{method_name}'",
            )

    def test_audio_commands_methods_available(self):
        """Test that AudioCommands has all expected methods."""
        audio_commands = AudioCommands(**self.mock_services)

        expected_methods = [
            "audio_info",
            "audio_formats",
            "audio_convert",
            "audio_preprocess",
            "audio_split",
        ]

        for method_name in expected_methods:
            self.assertTrue(
                hasattr(audio_commands, method_name),
                f"AudioCommands should have method '{method_name}'",
            )


if __name__ == "__main__":
    unittest.main()
