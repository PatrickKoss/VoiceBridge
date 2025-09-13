import subprocess
from pathlib import Path

from voicebridge.ports.interfaces import AudioFormatService, AudioSplittingService


class FFmpegAudioSplittingAdapter(AudioSplittingService):
    """FFmpeg-based audio splitting service."""

    def __init__(self, audio_format_service: AudioFormatService):
        self.audio_format_service = audio_format_service

    def split_by_duration(
        self, file_path: Path, chunk_duration: int, output_dir: Path
    ) -> list[Path]:
        """Split audio file into chunks of specified duration (seconds)."""
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Get audio info to calculate number of chunks
        audio_info = self.audio_format_service.get_audio_info(file_path)
        total_duration = audio_info.get("duration", 0)

        if total_duration == 0:
            return []

        chunks = []
        chunk_count = int(total_duration // chunk_duration) + (
            1 if total_duration % chunk_duration else 0
        )

        base_name = file_path.stem
        extension = file_path.suffix

        for i in range(chunk_count):
            start_time = i * chunk_duration
            chunk_path = output_dir / f"{base_name}_chunk_{i:03d}{extension}"

            cmd = [
                "ffmpeg",
                "-i",
                str(file_path),
                "-ss",
                str(start_time),
                "-t",
                str(chunk_duration),
                "-c",
                "copy",  # Copy without re-encoding when possible
                "-y",
                str(chunk_path),
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0 and chunk_path.exists():
                    chunks.append(chunk_path)
            except subprocess.SubprocessError:
                continue

        return chunks

    def split_by_silence(
        self, file_path: Path, silence_threshold: float, output_dir: Path
    ) -> list[Path]:
        """Split audio file by silence detection."""
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Use FFmpeg's silencedetect filter to find silence
        base_name = file_path.stem
        extension = file_path.suffix

        try:
            # First pass: detect silence
            detect_cmd = [
                "ffmpeg",
                "-i",
                str(file_path),
                "-af",
                f"silencedetect=noise={silence_threshold}:duration=0.5",
                "-f",
                "null",
                "-",
            ]

            result = subprocess.run(detect_cmd, capture_output=True, text=True)

            # Parse silence periods from output
            silence_times = self._parse_silence_times(result.stderr)

            if not silence_times:
                # If no silence detected, return original file
                return [file_path]

            # Second pass: split at silence points
            chunks = []
            start_time = 0

            for i, silence_start in enumerate(silence_times):
                if silence_start > start_time:
                    chunk_path = output_dir / f"{base_name}_segment_{i:03d}{extension}"

                    split_cmd = [
                        "ffmpeg",
                        "-i",
                        str(file_path),
                        "-ss",
                        str(start_time),
                        "-to",
                        str(silence_start),
                        "-c",
                        "copy",
                        "-y",
                        str(chunk_path),
                    ]

                    split_result = subprocess.run(
                        split_cmd, capture_output=True, text=True
                    )
                    if split_result.returncode == 0 and chunk_path.exists():
                        chunks.append(chunk_path)

                start_time = silence_start + 0.5  # Skip silence duration

            # Add final segment if there's remaining audio
            audio_info = self.audio_format_service.get_audio_info(file_path)
            total_duration = audio_info.get("duration", 0)

            if start_time < total_duration:
                final_chunk = (
                    output_dir / f"{base_name}_segment_{len(chunks):03d}{extension}"
                )
                final_cmd = [
                    "ffmpeg",
                    "-i",
                    str(file_path),
                    "-ss",
                    str(start_time),
                    "-c",
                    "copy",
                    "-y",
                    str(final_chunk),
                ]

                final_result = subprocess.run(final_cmd, capture_output=True, text=True)
                if final_result.returncode == 0 and final_chunk.exists():
                    chunks.append(final_chunk)

            return chunks

        except subprocess.SubprocessError:
            return []

    def split_by_size(
        self, file_path: Path, max_size_mb: float, output_dir: Path
    ) -> list[Path]:
        """Split audio file by maximum file size in MB."""
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        # Get current file size
        current_size_mb = file_path.stat().st_size / (1024 * 1024)

        if current_size_mb <= max_size_mb:
            return [file_path]

        # Estimate chunk duration based on file size ratio
        audio_info = self.audio_format_service.get_audio_info(file_path)
        total_duration = audio_info.get("duration", 0)

        if total_duration == 0:
            return []

        # Calculate chunk duration to achieve target size
        size_ratio = max_size_mb / current_size_mb
        chunk_duration = int(
            total_duration * size_ratio * 0.9
        )  # 90% to ensure under limit

        if chunk_duration < 1:
            chunk_duration = 1

        return self.split_by_duration(file_path, chunk_duration, output_dir)

    def _parse_silence_times(self, ffmpeg_output: str) -> list[float]:
        """Parse silence detection output to extract silence start times."""
        silence_times = []

        for line in ffmpeg_output.split("\n"):
            if "silence_start:" in line:
                try:
                    # Extract time value after 'silence_start: '
                    time_part = line.split("silence_start:")[1].strip().split()[0]
                    silence_time = float(time_part)
                    silence_times.append(silence_time)
                except (IndexError, ValueError):
                    continue

        return sorted(silence_times)
