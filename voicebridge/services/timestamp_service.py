import re

from voicebridge.domain.models import TimestampMode, TranscriptionSegment
from voicebridge.ports.interfaces import TimestampService


class DefaultTimestampService(TimestampService):
    def process_segments(
        self, segments: list[TranscriptionSegment], mode: str
    ) -> list[TranscriptionSegment]:
        if mode == TimestampMode.WORD_LEVEL.value:
            return segments
        elif mode == TimestampMode.SENTENCE_LEVEL.value:
            return self.group_by_sentences(segments)
        elif mode == TimestampMode.PARAGRAPH_LEVEL.value:
            return self.group_by_paragraphs(segments)
        else:
            return segments

    def group_by_sentences(
        self, segments: list[TranscriptionSegment]
    ) -> list[TranscriptionSegment]:
        if not segments:
            return segments

        grouped = []
        current_sentence = []
        current_text = ""

        sentence_endings = re.compile(r"[.!?]+\s*")

        for segment in segments:
            current_sentence.append(segment)
            current_text += segment.text + " "

            if sentence_endings.search(segment.text) or self._is_long_pause(
                segment, segments
            ):
                if current_sentence:
                    grouped_segment = self._merge_segments(
                        current_sentence, current_text.strip()
                    )
                    grouped.append(grouped_segment)
                    current_sentence = []
                    current_text = ""

        if current_sentence:
            grouped_segment = self._merge_segments(
                current_sentence, current_text.strip()
            )
            grouped.append(grouped_segment)

        return grouped

    def group_by_paragraphs(
        self, segments: list[TranscriptionSegment]
    ) -> list[TranscriptionSegment]:
        if not segments:
            return segments

        sentence_grouped = self.group_by_sentences(segments)

        paragraphs = []
        current_paragraph = []
        current_text = ""

        for i, segment in enumerate(sentence_grouped):
            current_paragraph.append(segment)
            current_text += segment.text + " "

            next_segment = (
                sentence_grouped[i + 1] if i + 1 < len(sentence_grouped) else None
            )

            if (
                self._is_paragraph_break(segment, next_segment)
                or len(current_paragraph) >= 5
            ):  # Max 5 sentences per paragraph
                if current_paragraph:
                    paragraph_segment = self._merge_segments(
                        current_paragraph, current_text.strip()
                    )
                    paragraphs.append(paragraph_segment)
                    current_paragraph = []
                    current_text = ""

        if current_paragraph:
            paragraph_segment = self._merge_segments(
                current_paragraph, current_text.strip()
            )
            paragraphs.append(paragraph_segment)

        return paragraphs

    def _is_long_pause(
        self,
        current_segment: TranscriptionSegment,
        all_segments: list[TranscriptionSegment],
    ) -> bool:
        current_index = all_segments.index(current_segment)
        if current_index + 1 < len(all_segments):
            next_segment = all_segments[current_index + 1]
            pause_duration = next_segment.start_time - current_segment.end_time
            return pause_duration > 1.0  # 1 second pause threshold
        return False

    def _is_paragraph_break(
        self,
        current_segment: TranscriptionSegment,
        next_segment: TranscriptionSegment | None,
    ) -> bool:
        if not next_segment:
            return True

        pause_duration = next_segment.start_time - current_segment.end_time
        return pause_duration > 2.0  # 2 second pause for paragraph break

    def _merge_segments(
        self, segments: list[TranscriptionSegment], merged_text: str
    ) -> TranscriptionSegment:
        if not segments:
            raise ValueError("Cannot merge empty segments list")

        start_time = segments[0].start_time
        end_time = segments[-1].end_time

        avg_confidence = None
        if all(seg.confidence is not None for seg in segments):
            avg_confidence = sum(seg.confidence for seg in segments) / len(segments)

        speaker_id = self._determine_primary_speaker(segments)

        return TranscriptionSegment(
            text=merged_text,
            start_time=start_time,
            end_time=end_time,
            confidence=avg_confidence,
            speaker_id=speaker_id,
        )

    def _determine_primary_speaker(
        self, segments: list[TranscriptionSegment]
    ) -> int | None:
        if not segments:
            return None

        speaker_durations = {}
        for segment in segments:
            if segment.speaker_id is not None:
                duration = segment.end_time - segment.start_time
                speaker_durations[segment.speaker_id] = (
                    speaker_durations.get(segment.speaker_id, 0) + duration
                )

        if not speaker_durations:
            return None

        return max(speaker_durations, key=speaker_durations.get)
