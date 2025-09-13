import tempfile
import wave
from pathlib import Path

from voicebridge.domain.models import SpeakerInfo, TranscriptionSegment
from voicebridge.ports.interfaces import SpeakerDiarizationService


class MockSpeakerDiarizationService(SpeakerDiarizationService):
    """Mock speaker diarization for testing - assigns speakers based on timing patterns"""

    def __init__(self, max_speakers: int = 4):
        self.max_speakers = max_speakers

    def identify_speakers(
        self, audio_data: bytes, segments: list[TranscriptionSegment]
    ) -> list[TranscriptionSegment]:
        """Mock speaker identification based on timing patterns"""
        if not segments:
            return segments

        # Simple mock: assign speakers based on audio patterns and pauses
        identified_segments = []
        current_speaker = 1
        last_end_time = 0.0

        for segment in segments:
            # Switch speaker if there's a long pause (> 3 seconds)
            if segment.start_time - last_end_time > 3.0:
                current_speaker = (current_speaker % self.max_speakers) + 1

            # Create new segment with speaker ID
            identified_segment = TranscriptionSegment(
                text=segment.text,
                start_time=segment.start_time,
                end_time=segment.end_time,
                confidence=segment.confidence,
                speaker_id=current_speaker,
            )
            identified_segments.append(identified_segment)
            last_end_time = segment.end_time

        return identified_segments

    def get_speaker_info(
        self, segments: list[TranscriptionSegment]
    ) -> list[SpeakerInfo]:
        """Generate speaker statistics from segments"""
        if not segments:
            return []

        speaker_stats = {}

        for segment in segments:
            if segment.speaker_id is None:
                continue

            speaker_id = segment.speaker_id
            duration = segment.end_time - segment.start_time

            if speaker_id not in speaker_stats:
                speaker_stats[speaker_id] = {
                    "total_time": 0.0,
                    "segment_count": 0,
                    "confidence_sum": 0.0,
                    "confidence_count": 0,
                }

            speaker_stats[speaker_id]["total_time"] += duration
            speaker_stats[speaker_id]["segment_count"] += 1

            if segment.confidence is not None:
                speaker_stats[speaker_id]["confidence_sum"] += segment.confidence
                speaker_stats[speaker_id]["confidence_count"] += 1

        speakers = []
        for speaker_id, stats in speaker_stats.items():
            avg_confidence = None
            if stats["confidence_count"] > 0:
                avg_confidence = stats["confidence_sum"] / stats["confidence_count"]

            speaker = SpeakerInfo(
                speaker_id=speaker_id,
                name=f"Speaker {speaker_id}",
                total_speaking_time=stats["total_time"],
                confidence=avg_confidence,
            )
            speakers.append(speaker)

        # Sort by total speaking time (descending)
        speakers.sort(key=lambda s: s.total_speaking_time, reverse=True)
        return speakers


class PyAnnoteSpeakerDiarizationService(SpeakerDiarizationService):
    """Speaker diarization using pyannote.audio (requires installation)"""

    def __init__(self, max_speakers: int = 4, use_auth_token: str | None = None):
        self.max_speakers = max_speakers
        self.use_auth_token = use_auth_token
        self._pipeline = None

    def _load_pipeline(self):
        """Lazy load the pyannote pipeline"""
        if self._pipeline is not None:
            return self._pipeline

        try:
            from pyannote.audio import Pipeline

            # Load the speaker diarization pipeline
            if self.use_auth_token:
                self._pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.use_auth_token,
                )
            else:
                # Try to load without auth token (may fail for some models)
                self._pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1"
                )

            return self._pipeline

        except ImportError as e:
            raise ImportError(
                "pyannote.audio not installed. Install with: pip install pyannote.audio"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load pyannote pipeline: {e}") from e

    def identify_speakers(
        self, audio_data: bytes, segments: list[TranscriptionSegment]
    ) -> list[TranscriptionSegment]:
        """Identify speakers using pyannote.audio"""
        if not segments:
            return segments

        try:
            pipeline = self._load_pipeline()
        except Exception as e:
            print(f"Failed to load speaker diarization: {e}")
            # Fallback to mock service
            mock_service = MockSpeakerDiarizationService(self.max_speakers)
            return mock_service.identify_speakers(audio_data, segments)

        # Save audio data to temporary file for pyannote
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name

            # Write audio data as WAV file
            with wave.open(tmp_path, "wb") as wav_file:
                # Assume 16kHz, 16-bit, mono for now
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(audio_data)

        try:
            # Run speaker diarization
            diarization = pipeline(tmp_path)

            # Map segments to speakers
            identified_segments = []

            for segment in segments:
                # Find the most overlapping speaker for this segment
                speaker_id = self._find_dominant_speaker(
                    segment.start_time, segment.end_time, diarization
                )

                identified_segment = TranscriptionSegment(
                    text=segment.text,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    confidence=segment.confidence,
                    speaker_id=speaker_id,
                )
                identified_segments.append(identified_segment)

            return identified_segments

        except Exception as e:
            print(f"Speaker diarization failed: {e}")
            # Fallback to mock service
            mock_service = MockSpeakerDiarizationService(self.max_speakers)
            return mock_service.identify_speakers(audio_data, segments)

        finally:
            # Clean up temporary file
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass

    def _find_dominant_speaker(
        self, start_time: float, end_time: float, diarization
    ) -> int:
        """Find the speaker who talks the most in the given time segment"""
        speaker_durations = {}

        # Check overlap with each speaker segment
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            seg_start = segment.start
            seg_end = segment.end

            # Calculate overlap
            overlap_start = max(start_time, seg_start)
            overlap_end = min(end_time, seg_end)
            overlap_duration = max(0, overlap_end - overlap_start)

            if overlap_duration > 0:
                if speaker not in speaker_durations:
                    speaker_durations[speaker] = 0
                speaker_durations[speaker] += overlap_duration

        if not speaker_durations:
            return 1  # Default speaker

        # Return speaker with most overlap time
        dominant_speaker = max(speaker_durations, key=speaker_durations.get)

        # Convert speaker label to integer ID
        try:
            return int(dominant_speaker.split("_")[-1]) + 1
        except (ValueError, AttributeError):
            return hash(str(dominant_speaker)) % self.max_speakers + 1

    def get_speaker_info(
        self, segments: list[TranscriptionSegment]
    ) -> list[SpeakerInfo]:
        """Generate speaker statistics from segments"""
        if not segments:
            return []

        speaker_stats = {}

        for segment in segments:
            if segment.speaker_id is None:
                continue

            speaker_id = segment.speaker_id
            duration = segment.end_time - segment.start_time

            if speaker_id not in speaker_stats:
                speaker_stats[speaker_id] = {
                    "total_time": 0.0,
                    "segment_count": 0,
                    "confidence_sum": 0.0,
                    "confidence_count": 0,
                }

            speaker_stats[speaker_id]["total_time"] += duration
            speaker_stats[speaker_id]["segment_count"] += 1

            if segment.confidence is not None:
                speaker_stats[speaker_id]["confidence_sum"] += segment.confidence
                speaker_stats[speaker_id]["confidence_count"] += 1

        speakers = []
        for speaker_id, stats in speaker_stats.items():
            avg_confidence = None
            if stats["confidence_count"] > 0:
                avg_confidence = stats["confidence_sum"] / stats["confidence_count"]

            speaker = SpeakerInfo(
                speaker_id=speaker_id,
                name=f"Speaker {speaker_id}",
                total_speaking_time=stats["total_time"],
                confidence=avg_confidence,
            )
            speakers.append(speaker)

        # Sort by total speaking time (descending)
        speakers.sort(key=lambda s: s.total_speaking_time, reverse=True)
        return speakers


class SimpleSpeakerDiarizationService(SpeakerDiarizationService):
    """Simple speaker diarization based on audio energy and silence detection"""

    def __init__(self, max_speakers: int = 4, silence_threshold: float = 2.0):
        self.max_speakers = max_speakers
        self.silence_threshold = silence_threshold  # seconds

    def identify_speakers(
        self, audio_data: bytes, segments: list[TranscriptionSegment]
    ) -> list[TranscriptionSegment]:
        """Simple speaker identification based on silence patterns"""
        if not segments:
            return segments

        identified_segments = []
        current_speaker = 1
        last_segment_end = 0.0
        speaker_history = []  # Track recent speakers

        for i, segment in enumerate(segments):
            silence_duration = segment.start_time - last_segment_end

            # Long silence indicates speaker change
            if silence_duration > self.silence_threshold:
                # Cycle to next speaker, avoiding recent speakers
                current_speaker = self._get_next_speaker(
                    current_speaker, speaker_history
                )
                speaker_history = []  # Reset history after long pause

            # Short silence with different characteristics might indicate speaker change
            elif silence_duration > 0.5 and i > 0:
                # Simple heuristic: if segment is much longer/shorter, might be different speaker
                prev_duration = segments[i - 1].end_time - segments[i - 1].start_time
                curr_duration = segment.end_time - segment.start_time

                duration_ratio = max(curr_duration, prev_duration) / min(
                    curr_duration, prev_duration
                )
                if duration_ratio > 2.0:  # Significant duration difference
                    current_speaker = self._get_next_speaker(
                        current_speaker, speaker_history
                    )

            # Track speaker history (last few speakers)
            speaker_history.append(current_speaker)
            if len(speaker_history) > 3:
                speaker_history.pop(0)

            identified_segment = TranscriptionSegment(
                text=segment.text,
                start_time=segment.start_time,
                end_time=segment.end_time,
                confidence=segment.confidence,
                speaker_id=current_speaker,
            )
            identified_segments.append(identified_segment)
            last_segment_end = segment.end_time

        return identified_segments

    def _get_next_speaker(
        self, current_speaker: int, recent_speakers: list[int]
    ) -> int:
        """Get next speaker, avoiding recently used speakers"""
        for candidate in range(1, self.max_speakers + 1):
            if candidate != current_speaker and candidate not in recent_speakers[-2:]:
                return candidate

        # If all speakers were recent, just cycle to next
        return (current_speaker % self.max_speakers) + 1

    def get_speaker_info(
        self, segments: list[TranscriptionSegment]
    ) -> list[SpeakerInfo]:
        """Generate speaker statistics from segments"""
        if not segments:
            return []

        speaker_stats = {}

        for segment in segments:
            if segment.speaker_id is None:
                continue

            speaker_id = segment.speaker_id
            duration = segment.end_time - segment.start_time

            if speaker_id not in speaker_stats:
                speaker_stats[speaker_id] = {
                    "total_time": 0.0,
                    "segment_count": 0,
                    "confidence_sum": 0.0,
                    "confidence_count": 0,
                }

            speaker_stats[speaker_id]["total_time"] += duration
            speaker_stats[speaker_id]["segment_count"] += 1

            if segment.confidence is not None:
                speaker_stats[speaker_id]["confidence_sum"] += segment.confidence
                speaker_stats[speaker_id]["confidence_count"] += 1

        speakers = []
        for speaker_id, stats in speaker_stats.items():
            avg_confidence = None
            if stats["confidence_count"] > 0:
                avg_confidence = stats["confidence_sum"] / stats["confidence_count"]

            speaker = SpeakerInfo(
                speaker_id=speaker_id,
                name=f"Speaker {speaker_id}",
                total_speaking_time=stats["total_time"],
                confidence=avg_confidence,
            )
            speakers.append(speaker)

        # Sort by total speaking time (descending)
        speakers.sort(key=lambda s: s.total_speaking_time, reverse=True)
        return speakers
