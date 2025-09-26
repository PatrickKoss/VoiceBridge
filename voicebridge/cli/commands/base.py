from voicebridge.ports.interfaces import (
    AudioFormatService,
    AudioPreprocessingService,
    AudioSplittingService,
    BatchProcessingService,
    CircuitBreakerService,
    ConfigRepository,
    DaemonService,
    ExportService,
    Logger,
    PerformanceService,
    PostProcessingService,
    ProfileRepository,
    ProgressService,
    RetryService,
    SessionService,
    SystemService,
    TimestampService,
    VocabularyManagementService,
    VocabularyService,
    WebhookService,
)
from voicebridge.services.confidence_service import ConfidenceAnalyzer
from voicebridge.services.resume_service import TranscriptionResumeService
from voicebridge.services.transcription_service import WhisperTranscriptionOrchestrator
from voicebridge.services.tts_service import TTSDaemonService, TTSOrchestrator


class BaseCommands:
    """Base class for all command groups with shared service dependencies."""

    def __init__(
        self,
        config_repo: ConfigRepository,
        profile_repo: ProfileRepository,
        logger: Logger,
        # Core services
        system_service: SystemService | None = None,
        daemon_service: DaemonService | None = None,
        # STT services
        transcription_orchestrator: WhisperTranscriptionOrchestrator | None = None,
        session_service: SessionService | None = None,
        resume_service: TranscriptionResumeService | None = None,
        confidence_analyzer: ConfidenceAnalyzer | None = None,
        # TTS services
        tts_orchestrator: TTSOrchestrator | None = None,
        tts_daemon_service: TTSDaemonService | None = None,
        # Audio services
        audio_format_service: AudioFormatService | None = None,
        audio_preprocessing_service: AudioPreprocessingService | None = None,
        audio_splitting_service: AudioSplittingService | None = None,
        # Processing services
        batch_processing_service: BatchProcessingService | None = None,
        export_service: ExportService | None = None,
        performance_service: PerformanceService | None = None,
        # Advanced services
        vocabulary_service: VocabularyService | None = None,
        vocabulary_management_service: VocabularyManagementService | None = None,
        postprocessing_service: PostProcessingService | None = None,
        webhook_service: WebhookService | None = None,
        progress_service: ProgressService | None = None,
        retry_service: RetryService | None = None,
        circuit_breaker_service: CircuitBreakerService | None = None,
        timestamp_service: TimestampService | None = None,
    ):
        # Core dependencies (required)
        self.config_repo = config_repo
        self.profile_repo = profile_repo
        self.logger = logger

        # Core services
        self.system_service = system_service
        self.daemon_service = daemon_service

        # STT services
        self.transcription_orchestrator = transcription_orchestrator
        self.session_service = session_service
        self.resume_service = resume_service
        self.confidence_analyzer = confidence_analyzer

        # TTS services
        self.tts_orchestrator = tts_orchestrator
        self.tts_daemon_service = tts_daemon_service

        # Audio services
        self.audio_format_service = audio_format_service
        self.audio_preprocessing_service = audio_preprocessing_service
        self.audio_splitting_service = audio_splitting_service

        # Processing services
        self.batch_processing_service = batch_processing_service
        self.export_service = export_service
        self.performance_service = performance_service

        # Advanced services
        self.vocabulary_service = vocabulary_service
        self.vocabulary_management_service = vocabulary_management_service
        self.postprocessing_service = postprocessing_service
        self.webhook_service = webhook_service
        self.progress_service = progress_service
        self.retry_service = retry_service
        self.circuit_breaker_service = circuit_breaker_service
        self.timestamp_service = timestamp_service

    def _stop_audio_recorder(self) -> None:
        """Attempt to stop the active audio recorder stream if one is running."""
        if not self.transcription_orchestrator:
            return

        recorder = getattr(self.transcription_orchestrator, "audio_recorder", None)
        stop_method = getattr(recorder, "stop_current_stream", None)

        if recorder and stop_method and callable(stop_method):
            try:
                stop_method()
                self.logger.info("Audio recording stopped")
            except Exception as e:
                self.logger.error(f"Failed to stop audio recording: {e}")

    def _build_config(
        self,
        model: str | None = None,
        language: str | None = None,
        initial_prompt: str | None = None,
        temperature: float = 0.0,
        profile: str | None = None,
        **kwargs,
    ):
        """Build configuration by merging profile, defaults, and provided options."""
        from dataclasses import replace

        from voicebridge.cli.utils.command_helpers import (
            build_whisper_config,
            handle_profile_config,
        )

        # Load base config (profile or default)
        config = handle_profile_config(profile, self.config_repo, self.profile_repo)

        # Build whisper config from command options
        whisper_config = build_whisper_config(
            model=model,
            language=language,
            initial_prompt=initial_prompt,
            temperature=temperature,
            **kwargs,
        )

        # Merge configurations
        if whisper_config:
            config = replace(config, **whisper_config)

        return config
