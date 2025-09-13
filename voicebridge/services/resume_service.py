import os
import wave
from collections.abc import Iterator

from voicebridge.domain.models import TranscriptionResult, WhisperConfig
from voicebridge.ports.interfaces import SessionService, TranscriptionService


class TranscriptionResumeService:
    def __init__(
        self,
        transcription_service: TranscriptionService,
        session_service: SessionService,
    ):
        self.transcription_service = transcription_service
        self.session_service = session_service

    def resume_transcription(
        self, session_id: str, config: WhisperConfig
    ) -> Iterator[TranscriptionResult]:
        """Resume a transcription from where it left off."""
        session = self.session_service.load_session(session_id)

        if session.is_completed:
            raise ValueError(f"Session {session_id} is already completed")

        if not os.path.exists(session.audio_file):
            raise FileNotFoundError(f"Audio file not found: {session.audio_file}")

        # Calculate starting position in bytes
        start_seconds = session.progress_seconds

        # Read audio file and skip to resume position
        audio_data = self._read_audio_file(session.audio_file)
        audio_duration = self._get_audio_duration(session.audio_file)

        # Update session with total duration if not set
        if session.total_duration == 0.0:
            session.total_duration = audio_duration
            self.session_service.save_session(session)

        # Calculate byte offset for resuming
        bytes_per_second = len(audio_data) / audio_duration
        start_offset = int(start_seconds * bytes_per_second)

        remaining_audio = audio_data[start_offset:]

        # Use chunked transcription for large files
        if hasattr(self.transcription_service, "transcribe_chunked_file"):
            chunk_results = self.transcription_service.transcribe_chunked_file(
                remaining_audio, config
            )
        else:
            # Fallback to single transcription
            result = self.transcription_service.transcribe(remaining_audio, config)
            chunk_results = [result] if result.text.strip() else []

        current_progress = start_seconds
        accumulated_text = session.transcribed_text

        for result in chunk_results:
            # Update session progress
            if result.duration:
                current_progress += result.duration
            else:
                # Estimate progress based on chunk size
                chunk_duration = config.chunk_size
                current_progress += chunk_duration

            # Accumulate transcribed text
            accumulated_text += " " + result.text.strip()

            # Update session
            session.progress_seconds = min(current_progress, session.total_duration)
            session.transcribed_text = accumulated_text.strip()
            session.is_completed = (
                session.progress_seconds >= session.total_duration * 0.95
            )  # 95% threshold

            self.session_service.save_session(session)

            yield result

        # Mark as completed if we've processed the entire file
        if session.progress_seconds >= session.total_duration * 0.95:
            session.is_completed = True
            self.session_service.save_session(session)

    def create_resumable_transcription(
        self, audio_file: str, config: WhisperConfig, session_name: str | None = None
    ) -> Iterator[TranscriptionResult]:
        """Create a new resumable transcription session."""
        # Create new session
        session = self.session_service.create_session(audio_file, session_name)

        try:
            # Get audio duration
            audio_duration = self._get_audio_duration(audio_file)
            session.total_duration = audio_duration
            self.session_service.save_session(session)

            # Start transcription
            audio_data = self._read_audio_file(audio_file)

            # Use chunked transcription for better resume capabilities
            if hasattr(self.transcription_service, "transcribe_chunked_file"):
                chunk_results = self.transcription_service.transcribe_chunked_file(
                    audio_data, config
                )
            else:
                # Fallback to single transcription
                result = self.transcription_service.transcribe(audio_data, config)
                chunk_results = [result] if result.text.strip() else []

            current_progress = 0.0
            accumulated_text = ""

            for result in chunk_results:
                # Update progress
                if result.duration:
                    current_progress += result.duration
                else:
                    # Estimate based on chunk size
                    current_progress += config.chunk_size

                # Accumulate text
                accumulated_text += " " + result.text.strip()

                # Update session
                session.progress_seconds = min(current_progress, session.total_duration)
                session.transcribed_text = accumulated_text.strip()
                session.is_completed = (
                    session.progress_seconds >= session.total_duration * 0.95
                )

                self.session_service.save_session(session)

                yield result

            # Final completion check
            if session.progress_seconds >= session.total_duration * 0.95:
                session.is_completed = True
                self.session_service.save_session(session)

        except Exception as e:
            # Save error state in session
            session.transcribed_text += f"\n[ERROR: {str(e)}]"
            self.session_service.save_session(session)
            raise

    def get_session_progress(self, session_id: str) -> dict:
        """Get detailed progress information for a session."""
        session = self.session_service.load_session(session_id)

        progress_percentage = 0.0
        if session.total_duration > 0:
            progress_percentage = (
                session.progress_seconds / session.total_duration
            ) * 100

        return {
            "session_id": session.session_id,
            "session_name": session.session_name,
            "progress_seconds": session.progress_seconds,
            "total_duration": session.total_duration,
            "progress_percentage": progress_percentage,
            "is_completed": session.is_completed,
            "transcribed_text_length": len(session.transcribed_text),
            "can_resume": not session.is_completed,
            "audio_file_exists": os.path.exists(session.audio_file),
        }

    def _read_audio_file(self, file_path: str) -> bytes:
        """Read audio file as bytes."""
        with open(file_path, "rb") as f:
            return f.read()

    def _get_audio_duration(self, file_path: str) -> float:
        """Get audio file duration in seconds."""
        try:
            # Try to read as WAV file first
            with wave.open(file_path, "rb") as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                return frames / sample_rate
        except (wave.Error, Exception):
            # Fallback: estimate based on file size and assume 16kHz mono
            file_size = os.path.getsize(file_path)
            # Assume 16-bit samples at 16kHz
            estimated_duration = file_size / (16000 * 2)  # 2 bytes per sample
            return estimated_duration
