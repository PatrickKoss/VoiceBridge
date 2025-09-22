"""
Modular CLI commands package.

This package contains the refactored command structure that replaces the monolithic
CLICommands class. Each command group is now in its own module for better maintainability.
"""

from voicebridge.cli.commands.advanced_commands import AdvancedCommands
from voicebridge.cli.commands.audio_commands import AudioCommands
from voicebridge.cli.commands.base import BaseCommands
from voicebridge.cli.commands.config_commands import ConfigCommands
from voicebridge.cli.commands.export_commands import ExportCommands
from voicebridge.cli.commands.speech_commands import SpeechCommands
from voicebridge.cli.commands.system_commands import SystemCommands
from voicebridge.cli.commands.transcription_commands import TranscriptionCommands
from voicebridge.cli.commands.tts_commands import TTSCommands

# Keep CLICommands for backward compatibility (if needed)
# This imports the original large class
try:
    from voicebridge.cli.commands_original import CLICommands
except ImportError:
    # If the original file is not available, create a stub
    class CLICommands:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "CLICommands has been refactored into modular command groups. "
                "Use the CommandRegistry and individual command classes instead."
            )

__all__ = [
    'BaseCommands',
    'SpeechCommands',
    'TranscriptionCommands',
    'TTSCommands',
    'AudioCommands',
    'SystemCommands',
    'ConfigCommands',
    'ExportCommands',
    'AdvancedCommands',
    'CLICommands',  # For backward compatibility
]
