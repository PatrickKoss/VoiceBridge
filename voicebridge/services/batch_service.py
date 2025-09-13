import concurrent.futures
from pathlib import Path
from typing import Any

from voicebridge.domain.models import WhisperConfig
from voicebridge.ports.interfaces import (
    AudioFormatService,
    BatchProcessingService,
    Logger,
    TranscriptionService,
)


class WhisperBatchProcessingService(BatchProcessingService):
    """Batch processing service for multiple audio files."""

    def __init__(
        self,
        transcription_service: TranscriptionService,
        audio_format_service: AudioFormatService,
        logger: Logger,
    ):
        self.transcription_service = transcription_service
        self.audio_format_service = audio_format_service
        self.logger = logger

        # Default file patterns for audio files
        self.default_patterns = [
            "*.mp3",
            "*.wav",
            "*.m4a",
            "*.aac",
            "*.flac",
            "*.ogg",
            "*.wma",
            "*.aiff",
            "*.au",
            "*.webm",
        ]

    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        config: WhisperConfig,
        max_workers: int = 4,
        file_patterns: list[str] = None,
    ) -> list[dict[str, Any]]:
        """Process all audio files in a directory with parallel workers."""
        if not input_dir.exists() or not input_dir.is_dir():
            raise ValueError(f"Input directory does not exist: {input_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Get list of files to process
        patterns = file_patterns or self.default_patterns
        files_to_process = self.get_processable_files(input_dir, patterns)

        if not files_to_process:
            self.logger.info(f"No processable audio files found in {input_dir}")
            return []

        self.logger.info(f"Found {len(files_to_process)} files to process")

        # Process files using thread pool for I/O bound operations
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(
                    self._process_single_file, file_path, output_dir, config
                ): file_path
                for file_path in files_to_process
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)

                    status = "success" if result.get("success") else "failed"
                    self.logger.info(f"Processed {file_path.name}: {status}")

                except Exception as e:
                    error_result = {
                        "file_path": str(file_path),
                        "success": False,
                        "error": str(e),
                        "transcription": None,
                        "processing_time": 0,
                    }
                    results.append(error_result)
                    self.logger.error(f"Failed to process {file_path.name}: {e}")

        return results

    def get_processable_files(
        self, directory: Path, patterns: list[str] = None
    ) -> list[Path]:
        """Get list of processable audio files in directory."""
        if not directory.exists() or not directory.is_dir():
            return []

        patterns = patterns or self.default_patterns
        files = []

        for pattern in patterns:
            files.extend(directory.glob(pattern))
            # Also search subdirectories
            files.extend(directory.glob(f"**/{pattern}"))

        # Filter for supported formats and remove duplicates
        supported_files = []
        seen_files = set()

        for file_path in files:
            if file_path.is_file() and file_path not in seen_files:
                if self.audio_format_service.is_supported_format(file_path):
                    supported_files.append(file_path)
                    seen_files.add(file_path)

        return sorted(supported_files)

    def estimate_batch_time(self, files: list[Path]) -> float:
        """Estimate total processing time for batch based on file durations."""
        total_duration = 0.0

        for file_path in files:
            try:
                audio_info = self.audio_format_service.get_audio_info(file_path)
                total_duration += audio_info.get("duration", 0)
            except Exception:
                # Assume 3 minutes if we can't get duration
                total_duration += 180

        # Rough estimate: transcription takes about 1/10 of audio duration
        # Add overhead for I/O and processing
        estimated_time = (total_duration * 0.1) + (len(files) * 2)

        return estimated_time

    def _process_single_file(
        self, file_path: Path, output_dir: Path, config: WhisperConfig
    ) -> dict[str, Any]:
        """Process a single audio file for transcription."""
        import time

        start_time = time.time()

        try:
            # Convert to WAV if needed
            temp_wav_path = None
            if file_path.suffix.lower() != ".wav":
                temp_wav_path = output_dir / f"temp_{file_path.stem}.wav"
                conversion_success = self.audio_format_service.convert_to_wav(
                    file_path, temp_wav_path
                )
                if not conversion_success:
                    raise RuntimeError(f"Failed to convert {file_path} to WAV")

                audio_path = temp_wav_path
            else:
                audio_path = file_path

            # Read audio data
            with open(audio_path, "rb") as f:
                audio_data = f.read()

            # Transcribe
            transcription_result = self.transcription_service.transcribe(
                audio_data, config
            )

            # Save transcription result
            output_file = output_dir / f"{file_path.stem}_transcription.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(transcription_result.text)

            # Clean up temporary file
            if temp_wav_path and temp_wav_path.exists():
                temp_wav_path.unlink()

            processing_time = time.time() - start_time

            return {
                "file_path": str(file_path),
                "output_path": str(output_file),
                "success": True,
                "transcription": transcription_result.text,
                "confidence": transcription_result.confidence,
                "processing_time": processing_time,
                "error": None,
            }

        except Exception as e:
            # Clean up on error
            if "temp_wav_path" in locals() and temp_wav_path and temp_wav_path.exists():
                temp_wav_path.unlink()

            processing_time = time.time() - start_time

            return {
                "file_path": str(file_path),
                "output_path": None,
                "success": False,
                "transcription": None,
                "confidence": 0.0,
                "processing_time": processing_time,
                "error": str(e),
            }
