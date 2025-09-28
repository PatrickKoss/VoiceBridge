import gc
import io
import time
from collections.abc import Iterator
from typing import Any

from voicebridge.domain.models import (
    GPUInfo,
    GPUType,
    TranscriptionResult,
    WhisperConfig,
)
from voicebridge.ports.interfaces import (
    PerformanceService,
    SystemService,
    TranscriptionService,
)

try:
    import torch
    import whisper
except ImportError:
    whisper = None
    torch = None


class WhisperTranscriptionService(TranscriptionService):
    def __init__(
        self,
        system_service: SystemService | None = None,
        performance_service: PerformanceService | None = None,
    ):
        if whisper is None:
            raise RuntimeError("Whisper library not available")
        self._model = None
        self._current_model_name = None
        self._current_device = None
        self._gpu_devices: list[GPUInfo] = []
        self._system_service = system_service
        self._performance_service = performance_service

        # Initialize GPU detection
        if self._system_service:
            self._gpu_devices = self._system_service.detect_gpu_devices()

    def _select_device(self, config: WhisperConfig) -> str:
        """Select the best device based on configuration and availability."""
        if config.force_cpu:
            return "cpu"

        if not config.use_gpu or not torch:
            return "cpu"

        # Find best GPU device
        gpu_devices = [gpu for gpu in self._gpu_devices if gpu.gpu_type != GPUType.NONE]
        if not gpu_devices:
            return "cpu"

        # Prioritize CUDA over Metal for performance
        cuda_devices = [gpu for gpu in gpu_devices if gpu.gpu_type == GPUType.CUDA]
        if cuda_devices and torch.cuda.is_available():
            return f"cuda:{config.gpu_device if config.gpu_device else '0'}"

        metal_devices = [gpu for gpu in gpu_devices if gpu.gpu_type == GPUType.METAL]
        if metal_devices and torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    def _load_model(self, model_name: str, config: WhisperConfig):
        """Load model with device selection and performance monitoring."""
        device = self._select_device(config)

        if (
            self._model is None
            or self._current_model_name != model_name
            or self._current_device != device
        ):
            # Clear previous model from memory
            if self._model is not None:
                del self._model
                if torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            # Start timing if performance service available
            timing_id = None
            if self._performance_service:
                timing_id = self._performance_service.start_timing("model_load")

            # Load new model
            self._model = whisper.load_model(model_name, device=device)
            self._current_model_name = model_name
            self._current_device = device

            # End timing
            if timing_id and self._performance_service:
                self._performance_service.end_timing(
                    timing_id,
                    model_name=model_name,
                    device=device,
                    gpu_used=(device != "cpu"),
                )

    def transcribe(
        self, audio_data: bytes, config: WhisperConfig
    ) -> TranscriptionResult:
        """Transcribe audio with performance monitoring and memory optimization."""
        # Start performance monitoring
        timing_id = None
        initial_memory = None
        if self._performance_service:
            timing_id = self._performance_service.start_timing("transcription")
            if self._system_service:
                memory_info = self._system_service.get_memory_usage()
                initial_memory = memory_info["used_mb"]

        self._load_model(config.model_name, config)

        # Memory check before processing
        self._check_memory_usage(config.max_memory_mb, len(audio_data))

        # Convert raw PCM bytes to proper WAV format for whisper
        import os
        import tempfile
        import wave

        # Convert raw PCM data to WAV format
        sample_rate = 16000  # FFmpeg outputs 16kHz mono
        channels = 1
        sample_width = 2  # 16-bit audio

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file_path = temp_file.name

        # Write proper WAV file with headers
        with wave.open(temp_file_path, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)

        try:
            options = {
                "language": config.language,
                "initial_prompt": config.initial_prompt,
                "temperature": config.temperature,
            }

            # Remove None values
            options = {k: v for k, v in options.items() if v is not None}

            result = self._model.transcribe(temp_file_path, **options)
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

        # Calculate performance metrics
        transcription_result = TranscriptionResult(
            text=result["text"].strip(),
            language=result.get("language"),
            confidence=self._calculate_confidence(result),
            duration=result.get("duration"),
        )

        # End performance monitoring
        if timing_id and self._performance_service:
            final_memory = None
            if self._system_service and initial_memory:
                memory_info = self._system_service.get_memory_usage()
                final_memory = memory_info["used_mb"] - initial_memory

            gpu_memory = (
                self._get_gpu_memory_usage() if self._current_device != "cpu" else None
            )

            # Calculate processing speed ratio
            processing_speed = None
            if transcription_result.duration:
                processing_speed = transcription_result.duration / (
                    time.time() - timing_id
                )

            self._performance_service.end_timing(
                timing_id,
                memory_used_mb=final_memory,
                gpu_used=(self._current_device != "cpu"),
                gpu_memory_mb=gpu_memory,
                processing_speed_ratio=processing_speed,
                audio_duration=transcription_result.duration,
                text_length=len(transcription_result.text),
            )

        return transcription_result

    def transcribe_stream(
        self, audio_stream: Iterator[bytes], config: WhisperConfig
    ) -> Iterator[TranscriptionResult]:
        """Stream transcription with memory optimization and chunked processing."""
        self._load_model(config.model_name, config)

        audio_buffer = io.BytesIO()
        chunk_size_bytes = config.chunk_size * 16000  # Assume 16kHz sample rate

        for chunk in audio_stream:
            audio_buffer.write(chunk)

            # Process when we have enough audio data (configurable chunk size)
            if audio_buffer.tell() > chunk_size_bytes:
                # Memory check before processing
                self._check_memory_usage(config.max_memory_mb, audio_buffer.tell())

                audio_buffer.seek(0)
                audio_data = audio_buffer.read()
                audio_buffer = io.BytesIO()  # Reset buffer

                try:
                    result = self.transcribe(audio_data, config)
                    if result.text.strip():
                        yield result

                    # Trigger garbage collection after each chunk
                    gc.collect()
                    if torch and torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception:
                    # Skip failed transcriptions in streaming mode
                    continue

    def transcribe_chunked_file(
        self, audio_data: bytes, config: WhisperConfig
    ) -> Iterator[TranscriptionResult]:
        """Process large audio files in chunks to optimize memory usage."""
        chunk_size_bytes = (
            config.chunk_size * 16000 * 2
        )  # Assume 16kHz, 2 bytes per sample
        total_size = len(audio_data)

        for offset in range(0, total_size, chunk_size_bytes):
            # Memory check before processing each chunk
            self._check_memory_usage(config.max_memory_mb, chunk_size_bytes)

            chunk = audio_data[offset : offset + chunk_size_bytes]
            if len(chunk) < 1000:  # Skip very small chunks
                continue

            try:
                result = self.transcribe(chunk, config)
                if result.text.strip():
                    yield result

                # Cleanup after each chunk
                gc.collect()
                if torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                # Log error but continue with next chunk
                if config.debug:
                    print(f"Chunk transcription failed: {e}")
                continue

    def _calculate_confidence(self, whisper_result: dict) -> float | None:
        # Whisper doesn't directly provide confidence scores
        # This is a simplified heuristic based on available data
        segments = whisper_result.get("segments", [])
        if not segments:
            return None

        # Average the "no_speech_prob" across segments (lower is better)
        no_speech_probs = [seg.get("no_speech_prob", 0.5) for seg in segments]
        avg_no_speech = sum(no_speech_probs) / len(no_speech_probs)
        return 1.0 - avg_no_speech  # Convert to confidence score

    def _check_memory_usage(
        self, max_memory_mb: int, additional_bytes: int = 0
    ) -> None:
        """Check if memory usage would exceed limits."""
        if not self._system_service:
            return

        memory_info = self._system_service.get_memory_usage()
        additional_mb = additional_bytes / (1024 * 1024)
        projected_usage = memory_info["used_mb"] + additional_mb

        # Auto-detect memory limit if set to 0 (default)
        # Use actual system capabilities instead of arbitrary caps
        effective_limit = max_memory_mb
        if max_memory_mb <= 0:
            # Use 80% of available memory as safe limit
            # This allows for better utilization on high-end systems
            effective_limit = memory_info["available_mb"] * 0.8

            # Only enforce reasonable minimums, no maximum caps
            # Minimum: 512MB for small models
            # If less than 512MB available, use what we have (user's responsibility)
            effective_limit = max(512, effective_limit)

            # For very large systems (>64GB), be slightly more conservative (70%)
            total_memory = memory_info.get("total_mb", 8192)
            if total_memory > 65536:  # >64GB systems
                effective_limit = min(
                    effective_limit, memory_info["available_mb"] * 0.7
                )

        if projected_usage > effective_limit:
            # Force garbage collection
            gc.collect()
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Re-check after cleanup
            memory_info = self._system_service.get_memory_usage()
            if memory_info["used_mb"] + additional_mb > effective_limit:
                limit_type = "auto-detected" if max_memory_mb <= 0 else "configured"
                raise RuntimeError(
                    f"Memory limit exceeded: {memory_info['used_mb']:.0f}MB + {additional_mb:.0f}MB > {effective_limit:.0f}MB ({limit_type}). "
                    f"Try using a smaller model or set --max-memory to a higher value."
                )

    def _get_gpu_memory_usage(self) -> float | None:
        """Get current GPU memory usage in MB."""
        if not torch:
            return None

        if self._current_device and self._current_device.startswith("cuda"):
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 * 1024)
        elif self._current_device == "mps":
            if torch.backends.mps.is_available():
                # MPS doesn't have direct memory query, estimate based on model
                return None

        return None

    def get_device_info(self) -> dict[str, Any]:
        """Get information about the current device and model."""
        return {
            "current_device": self._current_device or "none",
            "current_model": self._current_model_name or "none",
            "gpu_devices": [
                {
                    "type": gpu.gpu_type.value,
                    "name": gpu.device_name,
                    "memory_total": gpu.memory_total,
                    "memory_available": gpu.memory_available,
                }
                for gpu in self._gpu_devices
            ],
        }
