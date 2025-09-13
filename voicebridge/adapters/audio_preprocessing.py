import subprocess
from pathlib import Path

from voicebridge.ports.interfaces import AudioPreprocessingService


class FFmpegAudioPreprocessingAdapter(AudioPreprocessingService):
    """FFmpeg-based audio preprocessing service."""

    def reduce_noise(
        self, input_path: Path, output_path: Path, strength: float = 0.5
    ) -> bool:
        """Apply noise reduction using FFmpeg's afftdn filter."""
        if not input_path.exists():
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Use afftdn (FFT Denoiser) for noise reduction
            # strength: 0.0 (no reduction) to 1.0 (maximum reduction)
            noise_reduction = min(max(strength, 0.0), 1.0) * 95  # Convert to 0-95 range

            cmd = [
                "ffmpeg",
                "-i",
                str(input_path),
                "-af",
                f"afftdn=nr={noise_reduction}:nf=-25",
                "-y",
                str(output_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0

        except subprocess.SubprocessError:
            return False

    def normalize_volume(
        self, input_path: Path, output_path: Path, target_db: float = -20.0
    ) -> bool:
        """Normalize audio volume to target decibel level."""
        if not input_path.exists():
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Use loudnorm filter for EBU R128 loudness normalization
            cmd = [
                "ffmpeg",
                "-i",
                str(input_path),
                "-af",
                f"loudnorm=I={target_db}:dual_mono=true",
                "-y",
                str(output_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0

        except subprocess.SubprocessError:
            return False

    def trim_silence(
        self, input_path: Path, output_path: Path, threshold: float = 0.01
    ) -> bool:
        """Trim silence from beginning and end of audio."""
        if not input_path.exists():
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Use silenceremove filter to trim silence
            # threshold: 0.0 (silence) to 1.0 (loudest)
            silence_db = -40 + (threshold * 40)  # Convert to dB range (-40 to 0)

            cmd = [
                "ffmpeg",
                "-i",
                str(input_path),
                "-af",
                f"silenceremove=start_periods=1:start_silence=0.1:start_threshold={silence_db}dB:stop_periods=1:stop_silence=0.1:stop_threshold={silence_db}dB",
                "-y",
                str(output_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0

        except subprocess.SubprocessError:
            return False

    def apply_filters(
        self, input_path: Path, output_path: Path, filters: list[str]
    ) -> bool:
        """Apply custom FFmpeg audio filters."""
        if not input_path.exists() or not filters:
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Join filters with comma for FFmpeg audio filter chain
            filter_chain = ",".join(filters)

            cmd = [
                "ffmpeg",
                "-i",
                str(input_path),
                "-af",
                filter_chain,
                "-y",
                str(output_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0

        except subprocess.SubprocessError:
            return False

    def enhance_speech(self, input_path: Path, output_path: Path) -> bool:
        """Apply speech enhancement filters optimized for transcription."""
        speech_filters = [
            "highpass=f=80",  # Remove low-frequency noise
            "lowpass=f=8000",  # Remove high-frequency noise above speech
            "afftdn=nr=20:nf=-25",  # Light noise reduction
            "loudnorm=I=-23:LRA=7",  # Normalize for consistent levels
            "compand=0.1,0.3:-90,-90,-30,-15,-20,-5,0,0:0.1:0.0:-90:0.1",  # Light compression
        ]

        return self.apply_filters(input_path, output_path, speech_filters)
