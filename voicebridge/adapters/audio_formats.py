import json
import subprocess
from pathlib import Path
from typing import Any

from voicebridge.ports.interfaces import AudioFormatService


class FFmpegAudioFormatAdapter(AudioFormatService):
    """FFmpeg-based audio format conversion and information service."""

    SUPPORTED_FORMATS = [
        "mp3",
        "wav",
        "m4a",
        "aac",
        "flac",
        "ogg",
        "wma",
        "aiff",
        "au",
        "webm",
    ]

    def __init__(self):
        self._ensure_ffmpeg_available()

    def convert_to_wav(
        self, input_path: Path, output_path: Path, sample_rate: int = 16000
    ) -> bool:
        """Convert audio file to WAV format with specified sample rate."""
        try:
            cmd = [
                "ffmpeg",
                "-i",
                str(input_path),
                "-ar",
                str(sample_rate),  # Sample rate
                "-ac",
                "1",  # Mono
                "-f",
                "wav",  # Force WAV format
                "-y",  # Overwrite output
                str(output_path),
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            return result.returncode == 0

        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            return False

    def get_audio_info(self, file_path: Path) -> dict[str, Any]:
        """Get comprehensive audio file information using ffprobe."""
        try:
            cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(file_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return {}

            data = json.loads(result.stdout)

            # Extract audio stream info
            audio_stream = None
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "audio":
                    audio_stream = stream
                    break

            if not audio_stream:
                return {}

            format_info = data.get("format", {})

            return {
                "duration": float(format_info.get("duration", 0)),
                "format": format_info.get("format_name", "").split(",")[0],
                "size": int(format_info.get("size", 0)),
                "bitrate": int(format_info.get("bit_rate", 0)),
                "sample_rate": int(audio_stream.get("sample_rate", 0)),
                "channels": int(audio_stream.get("channels", 0)),
                "codec": audio_stream.get("codec_name", ""),
                "bit_depth": audio_stream.get("bits_per_sample", 0),
            }

        except (subprocess.SubprocessError, json.JSONDecodeError, ValueError):
            return {}

    def get_supported_formats(self) -> list[str]:
        """Return list of supported audio formats."""
        return self.SUPPORTED_FORMATS.copy()

    def is_supported_format(self, file_path: Path) -> bool:
        """Check if the file format is supported."""
        if not file_path.exists():
            return False

        suffix = file_path.suffix.lower().lstrip(".")
        return suffix in self.SUPPORTED_FORMATS

    def _ensure_ffmpeg_available(self) -> None:
        """Ensure ffmpeg and ffprobe are available."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"], capture_output=True, check=False
            )
            if result.returncode != 0:
                raise FileNotFoundError("ffmpeg not found")

            result = subprocess.run(
                ["ffprobe", "-version"], capture_output=True, check=False
            )
            if result.returncode != 0:
                raise FileNotFoundError("ffprobe not found")

        except (subprocess.SubprocessError, FileNotFoundError, ValueError) as e:
            raise RuntimeError(
                "FFmpeg is required but not found. Please install FFmpeg:\n"
                "- Ubuntu/Debian: sudo apt install ffmpeg\n"
                "- macOS: brew install ffmpeg\n"
                "- Windows: Download from https://ffmpeg.org/"
            ) from e
