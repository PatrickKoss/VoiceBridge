import time
from collections.abc import Iterator

from voicebridge.domain.models import PerformanceMetrics, WhisperConfig
from voicebridge.ports.interfaces import (
    AudioRecorder,
    ClipboardService,
    Logger,
    TranscriptionService,
)


class WhisperTranscriptionOrchestrator:
    def __init__(
        self,
        audio_recorder: AudioRecorder,
        transcription_service: TranscriptionService,
        clipboard_service: ClipboardService,
        logger: Logger,
    ):
        self.audio_recorder = audio_recorder
        self.transcription_service = transcription_service
        self.clipboard_service = clipboard_service
        self.logger = logger

    def transcribe_single_recording(self, config: WhisperConfig) -> str:
        start_time = time.time()

        try:
            # Record audio
            self.logger.info("Starting audio recording...")
            audio_data = b""

            for chunk in self.audio_recorder.record_stream():
                audio_data += chunk
                # In single recording mode, we collect all audio first
                # Break condition would be handled by the caller

            if not audio_data:
                return ""

            # Transcribe
            self.logger.info("Starting transcription...")
            transcribe_start = time.time()

            result = self.transcription_service.transcribe(audio_data, config)

            transcribe_time = time.time() - transcribe_start
            self.logger.log_performance(
                PerformanceMetrics(
                    operation="transcription",
                    duration=transcribe_time,
                    details={
                        "text_length": len(result.text),
                        "confidence": result.confidence,
                    },
                )
            )

            # Handle output
            text = result.text.strip()
            if text:
                self._handle_output(text, config)

            total_time = time.time() - start_time
            self.logger.log_performance(
                PerformanceMetrics(
                    operation="full_transcription_cycle",
                    duration=total_time,
                    details={"text_length": len(text)},
                )
            )

            return text

        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            return ""

    def transcribe_streaming(self, config: WhisperConfig) -> Iterator[str]:
        self.logger.info("Starting streaming transcription...")

        try:
            audio_stream = self.audio_recorder.record_stream()
            transcription_stream = self.transcription_service.transcribe_stream(
                audio_stream, config
            )

            for result in transcription_stream:
                text = result.text.strip()
                if text:
                    if config.copy_stream or config.paste_stream:
                        self._handle_output(text, config)
                    yield text

        except Exception as e:
            self.logger.error(f"Streaming transcription failed: {e}")

    def _handle_output(self, text: str, config: WhisperConfig) -> None:
        try:
            if config.copy_final or config.copy_stream:
                success = self.clipboard_service.copy_text(text)
                if success:
                    self.logger.debug(f"Copied text to clipboard: {text[:50]}...")
                else:
                    self.logger.error("Failed to copy text to clipboard")

            if config.paste_final or config.paste_stream:
                success = self.clipboard_service.type_text(text)
                if success:
                    self.logger.debug(f"Typed text: {text[:50]}...")
                else:
                    self.logger.error("Failed to type text")

        except Exception as e:
            self.logger.error(f"Output handling failed: {e}")
