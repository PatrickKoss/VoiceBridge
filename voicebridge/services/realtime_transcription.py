import queue
import threading
import time
from collections.abc import Iterator

from voicebridge.domain.models import TranscriptionResult, WhisperConfig
from voicebridge.ports.interfaces import AudioRecorder, Logger, TranscriptionService


class RealtimeTranscriptionService:
    """Enhanced real-time transcription with configurable chunking and VAD."""

    def __init__(
        self,
        audio_recorder: AudioRecorder,
        transcription_service: TranscriptionService,
        logger: Logger,
    ):
        self.audio_recorder = audio_recorder
        self.transcription_service = transcription_service
        self.logger = logger
        self.is_running = False

    def transcribe_realtime(
        self,
        config: WhisperConfig,
        chunk_duration: float = 2.0,
        overlap_duration: float = 0.5,
        vad_threshold: float = 0.01,
        output_format: str = "live",
    ) -> Iterator[dict]:
        """
        Real-time transcription with configurable parameters.

        Args:
            config: Whisper configuration
            chunk_duration: Duration of each audio chunk in seconds
            overlap_duration: Overlap between chunks to avoid word cuts
            vad_threshold: Voice activity detection threshold (0.0-1.0)
            output_format: Format for output ("live", "segments", "complete")
        """
        self.is_running = True
        self.logger.info("Starting realtime transcription service")

        try:
            # Audio buffer management
            audio_buffer = queue.Queue()
            sample_rate = 16000
            chunk_size = int(sample_rate * chunk_duration)
            overlap_size = int(sample_rate * overlap_duration)

            # Start audio recording thread
            self.logger.info("Starting audio capture thread")
            audio_thread = threading.Thread(
                target=self._audio_capture_worker,
                args=(audio_buffer, sample_rate),
                daemon=True,
            )
            audio_thread.start()
            self.logger.info("Audio capture thread started")

            # Processing state
            audio_accumulator = b""
            last_transcription = ""
            segment_counter = 0

            startup_time = time.time()
            max_startup_wait = 5.0  # 5 seconds max wait for first audio
            first_chunk_received = False

            while self.is_running:
                try:
                    # Get audio chunk with timeout
                    chunk = audio_buffer.get(timeout=1.0)
                    if chunk is None:  # End signal
                        break

                    if not first_chunk_received:
                        self.logger.info("First audio chunk received")
                        first_chunk_received = True

                    audio_accumulator += chunk

                    # Process when we have enough audio
                    if len(audio_accumulator) >= chunk_size * 2:  # 2 bytes per sample
                        # Extract chunk with overlap
                        current_chunk = audio_accumulator[: chunk_size * 2]

                        # Keep overlap for next iteration
                        audio_accumulator = audio_accumulator[
                            chunk_size * 2 - overlap_size * 2 :
                        ]

                        # Voice Activity Detection (simple energy-based)
                        if self._has_voice_activity(current_chunk, vad_threshold):
                            # Transcribe chunk
                            result = self.transcription_service.transcribe(
                                current_chunk, config
                            )

                            if result.text.strip():
                                segment_counter += 1

                                # Format output based on requested format
                                output = self._format_output(
                                    result,
                                    output_format,
                                    segment_counter,
                                    last_transcription,
                                )

                                if output:
                                    yield output
                                    last_transcription = result.text.strip()

                except queue.Empty:
                    # Check if we're still waiting for the first chunk
                    if not first_chunk_received:
                        elapsed = time.time() - startup_time
                        if elapsed > max_startup_wait:
                            self.logger.error(
                                "No audio input received after 5 seconds. Audio device may not be available."
                            )
                            break
                    continue
                except Exception as e:
                    self.logger.error(f"Realtime transcription error: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Realtime transcription failed: {e}")
        finally:
            self.is_running = False

    def transcribe_file_streaming(
        self, file_path: str, config: WhisperConfig, chunk_duration: float = 30.0
    ) -> Iterator[dict]:
        """Stream transcription of an audio file in chunks."""
        try:
            import subprocess
            from pathlib import Path

            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Audio file not found: {file_path}")

            sample_rate = 16000
            chunk_samples = int(sample_rate * chunk_duration)
            chunk_size = chunk_samples * 2  # 16-bit audio

            # Use FFmpeg to stream audio data
            cmd = [
                "ffmpeg",
                "-i",
                str(file_path),
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                "-f",
                "s16le",
                "-",
            ]

            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )

            chunk_counter = 0

            while True:
                # Read chunk from FFmpeg
                audio_chunk = process.stdout.read(chunk_size)
                if not audio_chunk:
                    break

                chunk_counter += 1

                try:
                    # Transcribe chunk
                    result = self.transcription_service.transcribe(audio_chunk, config)

                    if result.text.strip():
                        yield {
                            "chunk_id": chunk_counter,
                            "text": result.text,
                            "confidence": result.confidence,
                            "timestamp": chunk_counter * chunk_duration,
                            "type": "chunk",
                        }

                except Exception as e:
                    self.logger.error(f"Error processing chunk {chunk_counter}: {e}")
                    continue

            # Clean up
            process.terminate()
            process.wait()

        except Exception as e:
            self.logger.error(f"File streaming transcription failed: {e}")

    def stop(self):
        """Stop the real-time transcription."""
        self.is_running = False

    def _audio_capture_worker(self, audio_buffer: queue.Queue, sample_rate: int):
        """Worker thread for continuous audio capture."""
        try:
            self.logger.info("Starting audio recorder stream")
            for chunk in self.audio_recorder.record_stream(sample_rate):
                if not self.is_running:
                    break
                audio_buffer.put(chunk)
                self.logger.debug(f"Audio chunk received: {len(chunk)} bytes")
        except Exception as e:
            self.logger.error(f"Audio capture error: {e}")
        finally:
            audio_buffer.put(None)  # End signal
            self.logger.info("Audio capture worker finished")

    def _has_voice_activity(self, audio_chunk: bytes, threshold: float) -> bool:
        """Simple energy-based voice activity detection."""
        if not audio_chunk:
            return False

        try:
            import struct

            # Convert bytes to 16-bit integers
            samples = struct.unpack(f"<{len(audio_chunk) // 2}h", audio_chunk)

            # Calculate RMS energy
            rms = (sum(x * x for x in samples) / len(samples)) ** 0.5

            # Normalize to 0-1 range (rough approximation)
            normalized_energy = min(rms / 10000.0, 1.0)

            return normalized_energy > threshold

        except Exception:
            # If VAD fails, assume there's voice activity
            return True

    def _format_output(
        self,
        result: TranscriptionResult,
        output_format: str,
        segment_id: int,
        last_text: str,
    ) -> dict | None:
        """Format transcription output based on requested format."""

        if output_format == "live":
            # Live format shows incremental text
            return {
                "text": result.text,
                "confidence": result.confidence,
                "type": "live",
                "timestamp": time.time(),
            }

        elif output_format == "segments":
            # Segment format shows distinct segments
            if result.text.strip() != last_text:
                return {
                    "segment_id": segment_id,
                    "text": result.text,
                    "confidence": result.confidence,
                    "type": "segment",
                    "timestamp": time.time(),
                }

        elif output_format == "complete":
            # Complete format accumulates all text
            return {
                "complete_text": result.text,
                "confidence": result.confidence,
                "type": "complete",
                "segments_processed": segment_id,
                "timestamp": time.time(),
            }

        return None
