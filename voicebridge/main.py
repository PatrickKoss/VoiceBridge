#!/usr/bin/env python3

import sys
from pathlib import Path

# Add current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Adapters
from voicebridge.adapters.audio_factory import create_audio_recorder
from voicebridge.adapters.audio_formats import FFmpegAudioFormatAdapter
from voicebridge.adapters.audio_playback import create_audio_playback_service
from voicebridge.adapters.audio_preprocessing import FFmpegAudioPreprocessingAdapter
from voicebridge.adapters.audio_splitting import FFmpegAudioSplittingAdapter
from voicebridge.adapters.config import FileConfigRepository, FileProfileRepository
from voicebridge.adapters.logging import FileLogger
from voicebridge.adapters.session import FileSessionService
from voicebridge.adapters.system import PlatformClipboardService, StandardSystemService
from voicebridge.adapters.text_input import create_text_input_service
from voicebridge.adapters.transcription import WhisperTranscriptionService
from voicebridge.adapters.vibevoice_tts import VibeVoiceTTSAdapter
from voicebridge.adapters.vocabulary import VocabularyAdapter
from voicebridge.cli.app import create_app
from voicebridge.cli.registry import CommandRegistry

# Services
from voicebridge.services.batch_service import WhisperBatchProcessingService
from voicebridge.services.confidence_service import ConfidenceAnalyzer
from voicebridge.services.daemon_service import WhisperDaemonService
from voicebridge.services.export_service import DefaultExportService
from voicebridge.services.performance_service import WhisperPerformanceService
from voicebridge.services.progress_service import WhisperProgressService
from voicebridge.services.resume_service import TranscriptionResumeService
from voicebridge.services.simple_post_processing_service import (
    SimplePostProcessingService,
)
from voicebridge.services.simple_webhook_service import SimpleWebhookService
from voicebridge.services.timestamp_service import DefaultTimestampService
from voicebridge.services.transcription_service import WhisperTranscriptionOrchestrator
from voicebridge.services.tts_service import TTSDaemonService, TTSOrchestrator
from voicebridge.services.vocabulary_management_service import (
    VocabularyManagementService,
)


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
    audio_recorder = create_audio_recorder()
    clipboard_service = PlatformClipboardService()
    system_service = StandardSystemService()

    # Load initial config for logger setup
    config = config_repo.load()
    logger = FileLogger(log_file, performance_log, debug=config.debug)

    # Performance and session services
    performance_service = WhisperPerformanceService(system_service)
    session_service = FileSessionService(sessions_dir)
    progress_service = WhisperProgressService()

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

    # Vocabulary management services
    vocabulary_adapter = VocabularyAdapter(config_dir / "vocabulary")
    vocabulary_management_service = VocabularyManagementService(
        vocabulary_adapter, logger
    )

    # Post-processing and webhook services
    postprocessing_service = SimplePostProcessingService(config_dir)
    webhook_service = SimpleWebhookService(config_dir)

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

    # CLI Command Registry
    command_registry = CommandRegistry(
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
        # Advanced Services
        vocabulary_management_service=vocabulary_management_service,
        postprocessing_service=postprocessing_service,
        webhook_service=webhook_service,
        progress_service=progress_service,
    )

    return command_registry


def main():
    """Main entry point."""
    try:
        # Setup dependencies
        command_registry = setup_dependencies()

        # Ensure system dependencies
        command_registry.dependencies["system_service"].ensure_dependencies()

        # Create and run Typer app
        app = create_app(command_registry)
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
