import platform
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class OperationMode(Enum):
    TOGGLE = "toggle"
    PUSH_TO_TALK = "push_to_talk"


class PlatformType(Enum):
    WINDOWS = "windows"
    MACOS = "darwin"
    LINUX = "linux"


class GPUType(Enum):
    NONE = "none"
    CUDA = "cuda"
    METAL = "metal"
    OPENCL = "opencl"


class OutputFormat(Enum):
    JSON = "json"
    SRT = "srt"
    VTT = "vtt"
    PLAIN_TEXT = "txt"
    CSV = "csv"


class TimestampMode(Enum):
    WORD_LEVEL = "word"
    SENTENCE_LEVEL = "sentence"
    PARAGRAPH_LEVEL = "paragraph"


class TTSMode(Enum):
    CLIPBOARD = "clipboard"
    MOUSE = "mouse"
    MANUAL = "manual"


class TTSStreamingMode(Enum):
    NON_STREAMING = "non_streaming"
    STREAMING = "streaming"


class TTSOutputMode(Enum):
    PLAY_AUDIO = "play"
    SAVE_FILE = "save"
    BOTH = "both"


@dataclass
class WhisperConfig:
    model_name: str = "medium"
    language: str | None = None
    initial_prompt: str | None = None
    temperature: float = 0.0
    mode: OperationMode = OperationMode.TOGGLE
    key: str = "ctrl+f2"
    start_key: str = "ctrl+f2"
    stop_key: str = "f10"
    quit_key: str = "esc"
    paste_stream: bool = False
    copy_stream: bool = False
    paste_final: bool = False
    copy_final: bool = True
    debug: bool = False

    # GPU Settings
    use_gpu: bool = True
    gpu_device: str | None = None
    force_cpu: bool = False

    # Memory Optimization
    chunk_size: int = 30
    max_memory_mb: int = 0  # 0 = auto-detect based on available system memory

    # Resume Settings
    enable_resume: bool = True
    session_name: str | None = None

    # Export Settings
    export_format: OutputFormat = OutputFormat.PLAIN_TEXT
    timestamp_mode: TimestampMode = TimestampMode.SENTENCE_LEVEL
    include_confidence_scores: bool = True

    # Translation Settings
    enable_translation: bool = False
    translation_target_language: str | None = None
    auto_detect_language: bool = True

    # Speaker Detection Settings
    enable_speaker_detection: bool = False
    max_speakers: int = 4

    # TTS Settings
    tts_enabled: bool = False
    tts_config: "TTSConfig" = field(default_factory=lambda: TTSConfig())

    def to_dict(self) -> dict[str, Any]:
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif hasattr(value, "to_dict"):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WhisperConfig":
        if "mode" in data and isinstance(data["mode"], str):
            data["mode"] = OperationMode(data["mode"])
        if "export_format" in data and isinstance(data["export_format"], str):
            data["export_format"] = OutputFormat(data["export_format"])
        if "timestamp_mode" in data and isinstance(data["timestamp_mode"], str):
            data["timestamp_mode"] = TimestampMode(data["timestamp_mode"])
        if "tts_config" in data and isinstance(data["tts_config"], dict):
            data["tts_config"] = TTSConfig.from_dict(data["tts_config"])
        return cls(**data)


@dataclass
class TTSConfig:
    # Core TTS settings
    model_path: str = "aoi-ot/VibeVoice-7B"
    voice_samples_dir: str = "voices"
    default_voice: str = "en-Alice_woman"
    cfg_scale: float = 1.3
    inference_steps: int = 10

    # Input/Output modes
    tts_mode: TTSMode = TTSMode.CLIPBOARD
    streaming_mode: TTSStreamingMode = TTSStreamingMode.NON_STREAMING
    output_mode: TTSOutputMode = TTSOutputMode.PLAY_AUDIO

    # Hotkey configuration
    tts_toggle_key: str = "f11"
    tts_generate_key: str = "f2"
    tts_stop_key: str = "ctrl+alt+s"

    # Audio settings
    sample_rate: int = 24000
    output_file_path: str | None = None
    auto_play: bool = True

    # Performance settings
    use_gpu: bool = True
    max_text_length: int = 2000
    chunk_text_threshold: int = 500

    def to_dict(self) -> dict[str, Any]:
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TTSConfig":
        if "tts_mode" in data and isinstance(data["tts_mode"], str):
            data["tts_mode"] = TTSMode(data["tts_mode"])
        if "streaming_mode" in data and isinstance(data["streaming_mode"], str):
            data["streaming_mode"] = TTSStreamingMode(data["streaming_mode"])
        if "output_mode" in data and isinstance(data["output_mode"], str):
            data["output_mode"] = TTSOutputMode(data["output_mode"])
        return cls(**data)


@dataclass
class TTSResult:
    audio_data: bytes
    sample_rate: int
    text: str
    generation_time: float
    voice_used: str | None = None
    streaming_mode: bool = False


@dataclass
class VoiceInfo:
    name: str
    file_path: str
    display_name: str | None = None
    language: str | None = None
    gender: str | None = None


@dataclass
class TranscriptionSegment:
    text: str
    start_time: float
    end_time: float
    confidence: float | None = None
    speaker_id: int | None = None


@dataclass
class SpeakerInfo:
    speaker_id: int
    name: str | None = None
    confidence: float | None = None
    total_speaking_time: float = 0.0


@dataclass
class ExportConfig:
    format: OutputFormat
    timestamp_mode: TimestampMode = TimestampMode.SENTENCE_LEVEL
    include_confidence: bool = True
    include_speaker_info: bool = True
    translation_target: str | None = None
    output_file: str | None = None


@dataclass
class TranscriptionResult:
    text: str
    segments: list[TranscriptionSegment] = field(default_factory=list)
    speakers: list[SpeakerInfo] = field(default_factory=list)
    confidence: float | None = None
    language: str | None = None
    duration: float | None = None
    detected_language_probability: float | None = None


@dataclass
class AudioDeviceInfo:
    name: str
    device_id: str
    platform: PlatformType


@dataclass
class SystemInfo:
    platform: PlatformType

    @classmethod
    def current(cls) -> "SystemInfo":
        system = platform.system().lower()
        if system == "windows":
            return cls(PlatformType.WINDOWS)
        elif system == "darwin":
            return cls(PlatformType.MACOS)
        else:
            return cls(PlatformType.LINUX)


@dataclass
class GPUInfo:
    gpu_type: GPUType
    device_name: str
    memory_total: int = 0  # MB
    memory_available: int = 0  # MB
    compute_capability: str | None = None


@dataclass
class SessionInfo:
    session_id: str
    session_name: str | None
    created_at: datetime
    audio_file: str
    progress_seconds: float = 0.0
    total_duration: float = 0.0
    transcribed_text: str = ""
    is_completed: bool = False


@dataclass
class PerformanceMetrics:
    operation: str
    duration: float
    details: dict[str, Any] = field(default_factory=dict)

    # Extended metrics
    memory_used_mb: float | None = None
    gpu_used: bool = False
    gpu_memory_mb: float | None = None
    model_load_time: float | None = None
    processing_speed_ratio: float | None = None  # audio duration / processing time


class EventType(Enum):
    TRANSCRIPTION_START = "transcription_start"
    TRANSCRIPTION_COMPLETE = "transcription_complete"
    TRANSCRIPTION_ERROR = "transcription_error"
    MODEL_LOADED = "model_loaded"
    PROGRESS_UPDATE = "progress_update"


class RetryStrategy(Enum):
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR = "linear"
    FIXED_INTERVAL = "fixed_interval"


@dataclass
class VocabularyConfig:
    custom_words: list[str] = field(default_factory=list)
    domain_terms: dict[str, list[str]] = field(default_factory=dict)
    proper_nouns: list[str] = field(default_factory=list)
    technical_jargon: list[str] = field(default_factory=list)
    phonetic_mappings: dict[str, str] = field(default_factory=dict)
    boost_factor: float = 2.0
    enable_fuzzy_matching: bool = True


@dataclass
class PostProcessingConfig:
    enable_punctuation_cleanup: bool = True
    enable_capitalization: bool = True
    enable_profanity_filter: bool = False
    custom_replacements: dict[str, str] = field(default_factory=dict)
    sentence_segmentation: bool = True
    text_normalization: bool = True
    remove_filler_words: bool = True
    filler_words: list[str] = field(
        default_factory=lambda: ["um", "uh", "like", "you know"]
    )


@dataclass
class IntegrationConfig:
    webhook_urls: list[str] = field(default_factory=list)
    api_endpoints: list[str] = field(default_factory=list)
    event_types: list[EventType] = field(
        default_factory=lambda: [EventType.TRANSCRIPTION_COMPLETE]
    )
    authentication: dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 30
    retry_attempts: int = 3
    enable_async_delivery: bool = True


@dataclass
class RetryConfig:
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    retryable_errors: list[str] = field(
        default_factory=lambda: ["ConnectionError", "TimeoutError", "HTTPError"]
    )
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 300.0
    circuit_breaker_enabled: bool = True


@dataclass
class ProgressTracker:
    operation_id: str
    operation_type: str
    start_time: datetime
    current_progress: float = 0.0
    estimated_total_duration: float | None = None
    status: str = "running"
    current_step: str = ""
    steps_completed: int = 0
    total_steps: int = 0
    eta_seconds: float | None = None

    def calculate_eta(self) -> float | None:
        if self.current_progress > 0 and self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            total_estimated = elapsed / self.current_progress
            return total_estimated - elapsed
        return None


@dataclass
class WebhookEvent:
    event_type: EventType
    timestamp: datetime
    operation_id: str
    data: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "operation_id": self.operation_id,
            "data": self.data,
            "session_id": self.session_id,
        }


@dataclass
class CircuitBreakerState:
    failure_count: int = 0
    last_failure_time: datetime | None = None
    state: str = "closed"  # closed, open, half_open
    next_attempt_time: datetime | None = None
