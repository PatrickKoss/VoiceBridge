"""
Command registry for organizing and managing CLI command groups.
"""

from voicebridge.cli.commands.advanced_commands import AdvancedCommands
from voicebridge.cli.commands.api_commands import APICommands
from voicebridge.cli.commands.audio_commands import AudioCommands
from voicebridge.cli.commands.base import BaseCommands
from voicebridge.cli.commands.config_commands import ConfigCommands
from voicebridge.cli.commands.export_commands import ExportCommands
from voicebridge.cli.commands.speech_commands import SpeechCommands
from voicebridge.cli.commands.system_commands import SystemCommands
from voicebridge.cli.commands.transcription_commands import TranscriptionCommands
from voicebridge.cli.commands.tts_commands import TTSCommands


class CommandRegistry:
    """Registry for managing command groups and their dependencies."""

    # Map command group names to their classes
    COMMAND_GROUPS: dict[str, type[BaseCommands]] = {
        "speech": SpeechCommands,
        "transcription": TranscriptionCommands,
        "tts": TTSCommands,
        "audio": AudioCommands,
        "system": SystemCommands,
        "config": ConfigCommands,
        "export": ExportCommands,
        "advanced": AdvancedCommands,
        "api": APICommands,
    }

    def __init__(self, **service_dependencies):
        """
        Initialize the command registry with service dependencies.

        Args:
            **service_dependencies: All the service dependencies needed by command classes
        """
        self.dependencies = service_dependencies
        self._command_instances = {}

    def get_command_group(self, group_name: str) -> BaseCommands:
        """
        Get or create a command group instance.

        Args:
            group_name: Name of the command group

        Returns:
            Command group instance

        Raises:
            ValueError: If command group is not registered
        """
        if group_name not in self.COMMAND_GROUPS:
            available = ", ".join(self.COMMAND_GROUPS.keys())
            raise ValueError(
                f"Unknown command group '{group_name}'. Available: {available}"
            )

        if group_name not in self._command_instances:
            command_class = self.COMMAND_GROUPS[group_name]
            self._command_instances[group_name] = command_class(**self.dependencies)

        return self._command_instances[group_name]

    def get_all_command_groups(self) -> dict[str, BaseCommands]:
        """
        Get all command group instances.

        Returns:
            Dictionary mapping group names to instances
        """
        return {
            name: self.get_command_group(name) for name in self.COMMAND_GROUPS.keys()
        }

    def list_command_groups(self) -> list[str]:
        """
        List all available command group names.

        Returns:
            List of command group names
        """
        return list(self.COMMAND_GROUPS.keys())

    def validate_dependencies(self) -> dict[str, bool]:
        """
        Validate that all required dependencies are available.

        Returns:
            Dictionary mapping dependency names to availability status
        """
        # Core dependencies that should always be present
        required_deps = ["config_repo", "profile_repo", "logger"]

        validation_results = {}

        for dep in required_deps:
            validation_results[dep] = (
                dep in self.dependencies and self.dependencies[dep] is not None
            )

        # Optional dependencies
        optional_deps = [
            "transcription_orchestrator",
            "tts_orchestrator",
            "tts_daemon_service",
            "system_service",
            "daemon_service",
            "session_service",
            "performance_service",
            "audio_format_service",
            "audio_preprocessing_service",
            "audio_splitting_service",
            "batch_processing_service",
            "export_service",
            "resume_service",
            "confidence_analyzer",
            "vocabulary_service",
            "vocabulary_management_service",
            "postprocessing_service",
            "webhook_service",
            "progress_service",
            "retry_service",
            "circuit_breaker_service",
            "timestamp_service",
        ]

        for dep in optional_deps:
            validation_results[dep] = (
                dep in self.dependencies and self.dependencies[dep] is not None
            )

        return validation_results

    def get_dependency_summary(self) -> dict[str, dict[str, int]]:
        """
        Get a summary of dependency availability by command group.

        Returns:
            Dictionary with command group dependency statistics
        """
        validation = self.validate_dependencies()
        summary = {}

        # Define which dependencies each command group needs
        group_dependencies = {
            "speech": ["transcription_orchestrator", "system_service"],
            "transcription": [
                "transcription_orchestrator",
                "audio_format_service",
                "batch_processing_service",
                "resume_service",
            ],
            "tts": ["tts_orchestrator", "tts_daemon_service"],
            "audio": [
                "audio_format_service",
                "audio_preprocessing_service",
                "audio_splitting_service",
            ],
            "system": [
                "system_service",
                "performance_service",
                "session_service",
                "progress_service",
                "circuit_breaker_service",
            ],
            "config": ["config_repo", "profile_repo"],
            "export": ["export_service", "confidence_analyzer", "session_service"],
            "advanced": [
                "vocabulary_management_service",
                "postprocessing_service",
                "webhook_service",
                "progress_service",
            ],
            "api": [
                "transcription_orchestrator",
                "vocabulary_service",
                "postprocessing_service",
                "webhook_service",
                "progress_service",
            ],
        }

        for group_name, deps in group_dependencies.items():
            available = sum(1 for dep in deps if validation.get(dep, False))
            summary[group_name] = {
                "available": available,
                "total": len(deps),
                "percentage": (available / len(deps)) * 100 if deps else 0,
            }

        return summary


def create_command_registry(**service_dependencies) -> CommandRegistry:
    """
    Factory function to create a command registry with dependencies.

    Args:
        **service_dependencies: All service dependencies

    Returns:
        Configured CommandRegistry instance
    """
    return CommandRegistry(**service_dependencies)
