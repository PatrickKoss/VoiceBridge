from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from voicebridge.domain.models import (
    AudioDeviceInfo,
    CircuitBreakerState,
    EventType,
    ExportConfig,
    GPUInfo,
    OutputFormat,
    PerformanceMetrics,
    PostProcessingConfig,
    ProgressTracker,
    RetryConfig,
    SessionInfo,
    SystemInfo,
    TranscriptionResult,
    TranscriptionSegment,
    TTSConfig,
    TTSResult,
    VocabularyConfig,
    VoiceInfo,
    WebhookEvent,
    WhisperConfig,
)


class ConfigRepository(ABC):
    @abstractmethod
    def load(self) -> WhisperConfig:
        pass

    @abstractmethod
    def save(self, config: WhisperConfig) -> None:
        pass


class ProfileRepository(ABC):
    @abstractmethod
    def save_profile(self, name: str, config: WhisperConfig) -> None:
        pass

    @abstractmethod
    def load_profile(self, name: str) -> WhisperConfig:
        pass

    @abstractmethod
    def list_profiles(self) -> list[str]:
        pass

    @abstractmethod
    def delete_profile(self, name: str) -> bool:
        pass


class AudioRecorder(ABC):
    @abstractmethod
    def record_stream(self, sample_rate: int = 16000) -> Iterator[bytes]:
        pass

    @abstractmethod
    def list_devices(self) -> list[AudioDeviceInfo]:
        pass


class TranscriptionService(ABC):
    @abstractmethod
    def transcribe(
        self, audio_data: bytes, config: WhisperConfig
    ) -> TranscriptionResult:
        pass

    @abstractmethod
    def transcribe_stream(
        self, audio_stream: Iterator[bytes], config: WhisperConfig
    ) -> Iterator[TranscriptionResult]:
        pass


class ClipboardService(ABC):
    @abstractmethod
    def copy_text(self, text: str) -> bool:
        pass

    @abstractmethod
    def type_text(self, text: str) -> bool:
        pass


class DaemonService(ABC):
    @abstractmethod
    def is_running(self) -> bool:
        pass

    @abstractmethod
    def start(self, config: WhisperConfig) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def get_status(self) -> dict[str, Any]:
        pass


class Logger(ABC):
    @abstractmethod
    def info(self, message: str) -> None:
        pass

    @abstractmethod
    def debug(self, message: str) -> None:
        pass

    @abstractmethod
    def error(self, message: str) -> None:
        pass

    @abstractmethod
    def warning(self, message: str) -> None:
        pass

    @abstractmethod
    def log_performance(self, metrics: PerformanceMetrics) -> None:
        pass


class UpdateChecker(ABC):
    @abstractmethod
    def check_for_updates(self) -> str | None:
        pass


class SystemService(ABC):
    @abstractmethod
    def get_system_info(self) -> SystemInfo:
        pass

    @abstractmethod
    def ensure_dependencies(self) -> bool:
        pass

    @abstractmethod
    def detect_gpu_devices(self) -> list[GPUInfo]:
        pass

    @abstractmethod
    def get_memory_usage(self) -> dict[str, float]:
        pass


class SessionService(ABC):
    @abstractmethod
    def create_session(
        self, audio_file: str, session_name: str | None = None
    ) -> SessionInfo:
        pass

    @abstractmethod
    def save_session(self, session: SessionInfo) -> None:
        pass

    @abstractmethod
    def load_session(self, session_id: str) -> SessionInfo:
        pass

    @abstractmethod
    def list_sessions(self) -> list[SessionInfo]:
        pass

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        pass

    @abstractmethod
    def cleanup_completed_sessions(self) -> int:
        pass


class PerformanceService(ABC):
    @abstractmethod
    def start_timing(self, operation: str) -> str:
        pass

    @abstractmethod
    def end_timing(self, timing_id: str, **details) -> PerformanceMetrics:
        pass

    @abstractmethod
    def get_performance_stats(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def benchmark_model(self, model_name: str, use_gpu: bool = True) -> dict[str, Any]:
        pass


class VocabularyService(ABC):
    @abstractmethod
    def load_vocabulary(self, config: VocabularyConfig) -> None:
        pass

    @abstractmethod
    def enhance_transcription(self, text: str, config: VocabularyConfig) -> str:
        pass

    @abstractmethod
    def add_domain_terms(self, domain: str, terms: list[str]) -> None:
        pass


class VocabularyManagementService(ABC):
    """Abstract interface for vocabulary management operations from CLI"""

    @abstractmethod
    def add_words(
        self,
        words: list[str],
        vocabulary_type: str = "custom",
        profile: str = "default",
        weight: float = 1.0,
    ) -> bool:
        pass

    @abstractmethod
    def remove_words(
        self,
        words: list[str],
        vocabulary_type: str = "custom",
        profile: str = "default",
    ) -> bool:
        pass

    @abstractmethod
    def list_vocabularies(
        self, vocabulary_type: str | None = None, profile: str = "default"
    ) -> dict[str, list[str]]:
        pass

    @abstractmethod
    def import_vocabulary(
        self,
        file_path: str,
        vocabulary_type: str = "custom",
        profile: str = "default",
        format: str = "txt",
    ) -> bool:
        pass

    @abstractmethod
    def export_vocabulary(self, file_path: str, profile: str = "default") -> bool:
        pass


class PostProcessingService(ABC):
    @abstractmethod
    def process_text(self, text: str, config: PostProcessingConfig) -> str:
        pass

    @abstractmethod
    def clean_punctuation(self, text: str) -> str:
        pass

    @abstractmethod
    def normalize_capitalization(self, text: str) -> str:
        pass

    @abstractmethod
    def filter_profanity(self, text: str) -> str:
        pass


class WebhookService(ABC):
    @abstractmethod
    def send_webhook(self, event: WebhookEvent, url: str) -> bool:
        pass

    @abstractmethod
    def register_webhook(self, url: str, event_types: list[EventType]) -> None:
        pass

    @abstractmethod
    def trigger_event(self, event: WebhookEvent) -> None:
        pass


class RetryService(ABC):
    @abstractmethod
    def execute_with_retry(self, operation: callable, config: RetryConfig) -> Any:
        pass

    @abstractmethod
    def is_retryable_error(self, error: Exception, config: RetryConfig) -> bool:
        pass


class CircuitBreakerService(ABC):
    @abstractmethod
    def call(self, operation: callable, service_name: str) -> Any:
        pass

    @abstractmethod
    def get_state(self, service_name: str) -> CircuitBreakerState:
        pass

    @abstractmethod
    def reset(self, service_name: str) -> None:
        pass


class ProgressService(ABC):
    @abstractmethod
    def create_tracker(
        self, operation_id: str, operation_type: str, total_steps: int = 0
    ) -> ProgressTracker:
        pass

    @abstractmethod
    def update_progress(
        self, operation_id: str, progress: float, current_step: str = ""
    ) -> None:
        pass

    @abstractmethod
    def complete_operation(self, operation_id: str) -> None:
        pass

    @abstractmethod
    def get_tracker(self, operation_id: str) -> ProgressTracker | None:
        pass

    @abstractmethod
    def list_active_operations(self) -> list[dict]:
        """List all active operations"""
        pass

    @abstractmethod
    def cancel_operation(self, operation_id: str) -> bool:
        """Cancel an operation and return success status"""
        pass

    @abstractmethod
    def get_operation_status(self, operation_id: str) -> dict | None:
        """Get operation status as a dictionary"""
        pass


class ExportService(ABC):
    @abstractmethod
    def export_transcription(
        self, result: TranscriptionResult, config: ExportConfig
    ) -> str:
        pass

    @abstractmethod
    def get_supported_formats(self) -> list[OutputFormat]:
        pass

    @abstractmethod
    def export_to_file(self, result: TranscriptionResult, config: ExportConfig) -> bool:
        pass


class TimestampService(ABC):
    @abstractmethod
    def process_segments(
        self, segments: list[TranscriptionSegment], mode: str
    ) -> list[TranscriptionSegment]:
        pass

    @abstractmethod
    def group_by_sentences(
        self, segments: list[TranscriptionSegment]
    ) -> list[TranscriptionSegment]:
        pass

    @abstractmethod
    def group_by_paragraphs(
        self, segments: list[TranscriptionSegment]
    ) -> list[TranscriptionSegment]:
        pass


class TranslationService(ABC):
    @abstractmethod
    def translate_text(self, text: str, target_language: str) -> str:
        pass

    @abstractmethod
    def translate_segments(
        self, segments: list[TranscriptionSegment], target_language: str
    ) -> list[TranscriptionSegment]:
        pass

    @abstractmethod
    def detect_language(self, text: str) -> str:
        pass

    @abstractmethod
    def get_supported_languages(self) -> dict[str, str]:
        pass


class SpeakerDiarizationService(ABC):
    @abstractmethod
    def identify_speakers(
        self, audio_data: bytes, segments: list[TranscriptionSegment]
    ) -> list[TranscriptionSegment]:
        pass

    @abstractmethod
    def get_speaker_info(self, segments: list[TranscriptionSegment]) -> list[Any]:
        pass


class AudioFormatService(ABC):
    @abstractmethod
    def convert_to_wav(
        self, input_path: Path, output_path: Path, sample_rate: int = 16000
    ) -> bool:
        """Convert audio file to WAV format."""
        pass

    @abstractmethod
    def get_audio_info(self, file_path: Path) -> dict[str, Any]:
        """Get audio file metadata (duration, format, sample rate, etc.)."""
        pass

    @abstractmethod
    def get_supported_formats(self) -> list[str]:
        """Get list of supported input audio formats."""
        pass

    @abstractmethod
    def is_supported_format(self, file_path: Path) -> bool:
        """Check if file format is supported."""
        pass


class AudioSplittingService(ABC):
    @abstractmethod
    def split_by_duration(
        self, file_path: Path, chunk_duration: int, output_dir: Path
    ) -> list[Path]:
        """Split audio file by duration (in seconds)."""
        pass

    @abstractmethod
    def split_by_silence(
        self, file_path: Path, silence_threshold: float, output_dir: Path
    ) -> list[Path]:
        """Split audio file by silence detection."""
        pass

    @abstractmethod
    def split_by_size(
        self, file_path: Path, max_size_mb: float, output_dir: Path
    ) -> list[Path]:
        """Split audio file by maximum file size."""
        pass


class AudioPreprocessingService(ABC):
    @abstractmethod
    def reduce_noise(
        self, input_path: Path, output_path: Path, strength: float = 0.5
    ) -> bool:
        """Apply noise reduction to audio file."""
        pass

    @abstractmethod
    def normalize_volume(
        self, input_path: Path, output_path: Path, target_db: float = -20.0
    ) -> bool:
        """Normalize audio volume to target decibel level."""
        pass

    @abstractmethod
    def trim_silence(
        self, input_path: Path, output_path: Path, threshold: float = 0.01
    ) -> bool:
        """Trim silence from beginning and end of audio."""
        pass

    @abstractmethod
    def apply_filters(
        self, input_path: Path, output_path: Path, filters: list[str]
    ) -> bool:
        """Apply custom FFmpeg audio filters."""
        pass


class BatchProcessingService(ABC):
    @abstractmethod
    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        config: WhisperConfig,
        max_workers: int = 4,
        file_patterns: list[str] = None,
    ) -> list[dict[str, Any]]:
        """Process all audio files in a directory."""
        pass

    @abstractmethod
    def get_processable_files(
        self, directory: Path, patterns: list[str] = None
    ) -> list[Path]:
        """Get list of processable audio files in directory."""
        pass

    @abstractmethod
    def estimate_batch_time(self, files: list[Path]) -> float:
        """Estimate total processing time for batch."""
        pass


class TextInputService(ABC):
    """Abstract interface for getting text input from various sources"""

    @abstractmethod
    def get_clipboard_text(self) -> str:
        """Get text from system clipboard"""
        pass

    @abstractmethod
    def get_selected_text(self) -> str:
        """Get currently selected text under mouse cursor"""
        pass

    @abstractmethod
    def start_monitoring(self, callback: callable) -> None:
        """Start monitoring for text selection changes"""
        pass

    @abstractmethod
    def stop_monitoring(self) -> None:
        """Stop text monitoring"""
        pass


class TTSService(ABC):
    """Abstract interface for text-to-speech functionality"""

    @abstractmethod
    def generate_speech(
        self, text: str, voice_samples: list[str], config: TTSConfig
    ) -> TTSResult:
        """Generate speech from text (non-streaming)"""
        pass

    @abstractmethod
    def generate_speech_streaming(
        self, text: str, voice_samples: list[str], config: TTSConfig
    ) -> Iterator[bytes]:
        """Generate speech with streaming output"""
        pass

    @abstractmethod
    def load_voice_samples(self, voices_dir: str) -> dict[str, VoiceInfo]:
        """Load available voice samples from directory"""
        pass

    @abstractmethod
    def stop_generation(self) -> None:
        """Stop current generation if running"""
        pass

    @abstractmethod
    def is_generating(self) -> bool:
        """Check if currently generating speech"""
        pass


class AudioPlaybackService(ABC):
    """Abstract interface for audio playback"""

    @abstractmethod
    def play_audio(self, audio_data: bytes, sample_rate: int) -> None:
        """Play audio data"""
        pass

    @abstractmethod
    def save_audio(self, audio_data: bytes, sample_rate: int, filepath: str) -> None:
        """Save audio data to file"""
        pass

    @abstractmethod
    def stop_playback(self) -> None:
        """Stop current audio playback"""
        pass

    @abstractmethod
    def is_playing(self) -> bool:
        """Check if currently playing audio"""
        pass
