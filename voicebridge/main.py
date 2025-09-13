#!/usr/bin/env python3

import sys
from pathlib import Path

# Add current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Adapters
from adapters.audio import FFmpegAudioRecorder
from adapters.audio_formats import FFmpegAudioFormatAdapter
from adapters.audio_playback import create_audio_playback_service
from adapters.audio_preprocessing import FFmpegAudioPreprocessingAdapter
from adapters.audio_splitting import FFmpegAudioSplittingAdapter
from adapters.config import FileConfigRepository, FileProfileRepository
from adapters.logging import FileLogger
from adapters.session import FileSessionService
from adapters.system import PlatformClipboardService, StandardSystemService
from adapters.text_input import create_text_input_service
from adapters.transcription import WhisperTranscriptionService
from adapters.vibevoice_tts import VibeVoiceTTSAdapter
from cli.app import create_app
from cli.commands import CLICommands

# Services
from services.batch_service import WhisperBatchProcessingService
from services.confidence_service import ConfidenceAnalyzer
from services.daemon_service import WhisperDaemonService
from services.export_service import DefaultExportService
from services.performance_service import WhisperPerformanceService
from services.resume_service import TranscriptionResumeService
from services.timestamp_service import DefaultTimestampService
from services.transcription_service import WhisperTranscriptionOrchestrator
from services.tts_service import TTSDaemonService, TTSOrchestrator


def setup_dependencies(config_dir=None):
    """Setup dependency injection container."""

    # Configuration paths
    if config_dir is None:
        config_dir = Path.home() / ".config" / "whisper-cli"
    profiles_dir = config_dir / "profiles"
    sessions_dir = config_dir / "sessions"
    log_file = config_dir / "whisper.log"
    performance_log = config_dir / "performance.log"
    pid_file = config_dir / "daemon.pid"

    # Repositories
    config_repo = FileConfigRepository(config_dir)
    profile_repo = FileProfileRepository(profiles_dir)

    # Adapters
    audio_recorder = FFmpegAudioRecorder()
    clipboard_service = PlatformClipboardService()
    system_service = StandardSystemService()

    # Load initial config for logger setup
    config = config_repo.load()
    logger = FileLogger(log_file, performance_log, debug=config.debug)

    # Performance and session services
    performance_service = WhisperPerformanceService(system_service)
    session_service = FileSessionService(sessions_dir)

    # Enhanced transcription service with performance monitoring
    transcription_service = WhisperTranscriptionService(
        system_service=system_service, performance_service=performance_service
    )

    # Services
    daemon_service = WhisperDaemonService(pid_file, logger)
    transcription_orchestrator = WhisperTranscriptionOrchestrator(
        audio_recorder=audio_recorder,
        transcription_service=transcription_service,
        clipboard_service=clipboard_service,
        logger=logger,
    )

    # Resume service
    resume_service = TranscriptionResumeService(
        transcription_service=transcription_service, session_service=session_service
    )

    # Audio processing services
    audio_format_service = FFmpegAudioFormatAdapter()
    audio_preprocessing_service = FFmpegAudioPreprocessingAdapter()
    audio_splitting_service = FFmpegAudioSplittingAdapter(audio_format_service)
    batch_processing_service = WhisperBatchProcessingService(
        transcription_service=transcription_service,
        audio_format_service=audio_format_service,
        logger=logger,
    )

    # Export and analysis services
    export_service = DefaultExportService()
    timestamp_service = DefaultTimestampService()
    confidence_analyzer = ConfidenceAnalyzer()

    # TTS Services
    try:
        tts_service = VibeVoiceTTSAdapter()
        logger.info("VibeVoice TTS service initialized")
    except RuntimeError as e:
        logger.warning(f"VibeVoice TTS not available: {e}")
        tts_service = None

    text_input_service = create_text_input_service()
    audio_playback_service = create_audio_playback_service()

    # TTS Orchestrator
    if tts_service:
        tts_orchestrator = TTSOrchestrator(
            tts_service=tts_service,
            text_input_service=text_input_service,
            audio_playback_service=audio_playback_service,
            logger=logger,
        )
        tts_daemon_service = TTSDaemonService(
            orchestrator=tts_orchestrator,
            logger=logger,
        )
    else:
        tts_orchestrator = None
        tts_daemon_service = None

    # Services for future use (not currently wired into CLI)
    # translation_service = MockTranslationService()
    # speaker_service = MockSpeakerDiarizationService(max_speakers=config.max_speakers)

    # CLI Commands
    commands = CLICommands(
        config_repo=config_repo,
        profile_repo=profile_repo,
        daemon_service=daemon_service,
        transcription_orchestrator=transcription_orchestrator,
        system_service=system_service,
        logger=logger,
        session_service=session_service,
        performance_service=performance_service,
        resume_service=resume_service,
        export_service=export_service,
        timestamp_service=timestamp_service,
        confidence_analyzer=confidence_analyzer,
        audio_format_service=audio_format_service,
        audio_preprocessing_service=audio_preprocessing_service,
        audio_splitting_service=audio_splitting_service,
        batch_processing_service=batch_processing_service,
        # TTS Services
        tts_orchestrator=tts_orchestrator,
        tts_daemon_service=tts_daemon_service,
    )

    return commands


def main():
    """Main entry point."""
    try:
        # Setup dependencies
        commands = setup_dependencies()

        # Ensure system dependencies
        commands.system_service.ensure_dependencies()

        # Create and run Typer app
        app = create_app(commands)
        app()

    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
