from dataclasses import dataclass
from enum import Enum

from voicebridge.domain.models import TranscriptionResult, TranscriptionSegment


class ConfidenceLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class QualityFlag:
    segment_index: int
    issue_type: str
    description: str
    confidence_score: float
    severity: ConfidenceLevel


@dataclass
class ConfidenceAnalysis:
    overall_confidence: float
    confidence_level: ConfidenceLevel
    quality_flags: list[QualityFlag]
    segments_needing_review: list[int]
    audio_quality_score: float
    recommendations: list[str]


class ConfidenceAnalyzer:
    """Analyzes transcription confidence and identifies areas needing review"""

    def __init__(
        self,
        high_threshold: float = 0.9,
        medium_threshold: float = 0.7,
        low_threshold: float = 0.5,
        review_threshold: float = 0.6,
    ):
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.low_threshold = low_threshold
        self.review_threshold = review_threshold

    def analyze_confidence(self, result: TranscriptionResult) -> ConfidenceAnalysis:
        """Perform comprehensive confidence analysis"""
        if not result.segments:
            # Fallback for results without segments
            overall_confidence = result.confidence or 0.5
            confidence_level = self._get_confidence_level(overall_confidence)

            return ConfidenceAnalysis(
                overall_confidence=overall_confidence,
                confidence_level=confidence_level,
                quality_flags=[],
                segments_needing_review=[],
                audio_quality_score=0.5,
                recommendations=self._generate_recommendations([], confidence_level),
            )

        # Calculate overall confidence
        segment_confidences = [
            seg.confidence for seg in result.segments if seg.confidence is not None
        ]

        if segment_confidences:
            overall_confidence = sum(segment_confidences) / len(segment_confidences)
        else:
            overall_confidence = result.confidence or 0.5

        confidence_level = self._get_confidence_level(overall_confidence)

        # Identify quality issues
        quality_flags = self._identify_quality_issues(result.segments)

        # Find segments needing review
        segments_needing_review = [
            i
            for i, seg in enumerate(result.segments)
            if seg.confidence is not None and seg.confidence < self.review_threshold
        ]

        # Estimate audio quality
        audio_quality_score = self._estimate_audio_quality(result.segments)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            quality_flags, confidence_level
        )

        return ConfidenceAnalysis(
            overall_confidence=overall_confidence,
            confidence_level=confidence_level,
            quality_flags=quality_flags,
            segments_needing_review=segments_needing_review,
            audio_quality_score=audio_quality_score,
            recommendations=recommendations,
        )

    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert numeric confidence to categorical level"""
        if confidence >= self.high_threshold:
            return ConfidenceLevel.HIGH
        elif confidence >= self.medium_threshold:
            return ConfidenceLevel.MEDIUM
        elif confidence >= self.low_threshold:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def _identify_quality_issues(
        self, segments: list[TranscriptionSegment]
    ) -> list[QualityFlag]:
        """Identify potential quality issues in transcription"""
        flags = []

        for i, segment in enumerate(segments):
            # Low confidence segments
            if (
                segment.confidence is not None
                and segment.confidence < self.low_threshold
            ):
                flags.append(
                    QualityFlag(
                        segment_index=i,
                        issue_type="low_confidence",
                        description=f"Segment has low confidence ({segment.confidence:.2%})",
                        confidence_score=segment.confidence,
                        severity=ConfidenceLevel.LOW,
                    )
                )

            # Very short segments (might indicate noise)
            segment_duration = segment.end_time - segment.start_time
            if segment_duration < 0.5:
                flags.append(
                    QualityFlag(
                        segment_index=i,
                        issue_type="short_segment",
                        description=f"Very short segment ({segment_duration:.1f}s)",
                        confidence_score=segment.confidence or 0.5,
                        severity=ConfidenceLevel.MEDIUM,
                    )
                )

            # Repeated words (might indicate stuttering or processing issues)
            words = segment.text.lower().split()
            if len(words) > 1:
                repeated_count = sum(
                    1 for j in range(len(words) - 1) if words[j] == words[j + 1]
                )
                if repeated_count > 0:
                    flags.append(
                        QualityFlag(
                            segment_index=i,
                            issue_type="repeated_words",
                            description=f"Contains {repeated_count} repeated words",
                            confidence_score=segment.confidence or 0.5,
                            severity=ConfidenceLevel.MEDIUM,
                        )
                    )

            # Incomplete sentences (ends without punctuation)
            text = segment.text.strip()
            if text and text[-1] not in ".!?":
                flags.append(
                    QualityFlag(
                        segment_index=i,
                        issue_type="incomplete_sentence",
                        description="Segment may be incomplete (no ending punctuation)",
                        confidence_score=segment.confidence or 0.5,
                        severity=ConfidenceLevel.LOW,
                    )
                )

            # Unusual length (very long segments might have multiple speakers)
            if segment_duration > 30.0:
                flags.append(
                    QualityFlag(
                        segment_index=i,
                        issue_type="long_segment",
                        description=f"Very long segment ({segment_duration:.1f}s) - might contain multiple speakers",
                        confidence_score=segment.confidence or 0.5,
                        severity=ConfidenceLevel.MEDIUM,
                    )
                )

        return flags

    def _estimate_audio_quality(self, segments: list[TranscriptionSegment]) -> float:
        """Estimate audio quality based on transcription patterns"""
        if not segments:
            return 0.5

        quality_indicators = []

        # Average confidence
        confidences = [seg.confidence for seg in segments if seg.confidence is not None]
        if confidences:
            quality_indicators.append(sum(confidences) / len(confidences))

        # Segment length consistency (good audio has consistent segment lengths)
        durations = [seg.end_time - seg.start_time for seg in segments]
        if durations:
            avg_duration = sum(durations) / len(durations)
            duration_variance = sum((d - avg_duration) ** 2 for d in durations) / len(
                durations
            )
            consistency_score = max(0, 1.0 - (duration_variance / 100.0))  # Normalize
            quality_indicators.append(consistency_score)

        # Text quality (longer words indicate clearer speech)
        word_lengths = []
        for segment in segments:
            words = segment.text.split()
            word_lengths.extend([len(word) for word in words if word.isalpha()])

        if word_lengths:
            avg_word_length = sum(word_lengths) / len(word_lengths)
            word_quality = min(1.0, avg_word_length / 6.0)  # Normalize to 6 chars
            quality_indicators.append(word_quality)

        # Overall quality score
        if quality_indicators:
            return sum(quality_indicators) / len(quality_indicators)

        return 0.5

    def _generate_recommendations(
        self, flags: list[QualityFlag], confidence_level: ConfidenceLevel
    ) -> list[str]:
        """Generate recommendations based on analysis"""
        recommendations = []

        # General recommendations based on confidence level
        if confidence_level == ConfidenceLevel.VERY_LOW:
            recommendations.extend(
                [
                    "Consider re-recording with better audio quality",
                    "Check microphone placement and reduce background noise",
                    "Review all transcribed text manually",
                ]
            )
        elif confidence_level == ConfidenceLevel.LOW:
            recommendations.extend(
                [
                    "Review segments flagged for low confidence",
                    "Consider using a higher-quality audio source",
                    "Manual verification recommended for important content",
                ]
            )
        elif confidence_level == ConfidenceLevel.MEDIUM:
            recommendations.append("Review flagged segments for accuracy")

        # Specific recommendations based on flags
        flag_types = [flag.issue_type for flag in flags]

        if "low_confidence" in flag_types:
            recommendations.append(
                "Focus review on segments with low confidence scores"
            )

        if "repeated_words" in flag_types:
            recommendations.append("Check for stuttering or audio processing artifacts")

        if "short_segment" in flag_types:
            recommendations.append(
                "Very short segments may indicate background noise or processing issues"
            )

        if "long_segment" in flag_types:
            recommendations.append(
                "Long segments may contain multiple speakers - consider speaker detection"
            )

        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)

        return unique_recommendations

    def flag_uncertain_transcriptions(
        self, segments: list[TranscriptionSegment], threshold: float = 0.6
    ) -> list[TranscriptionSegment]:
        """Flag segments that need manual review based on confidence threshold"""
        flagged_segments = []

        for segment in segments:
            if segment.confidence is not None and segment.confidence < threshold:
                # Add a flag to the text
                flagged_text = (
                    f"[REVIEW NEEDED: {segment.confidence:.2%}] {segment.text}"
                )

                flagged_segment = TranscriptionSegment(
                    text=flagged_text,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    confidence=segment.confidence,
                    speaker_id=segment.speaker_id,
                )
                flagged_segments.append(flagged_segment)
            else:
                flagged_segments.append(segment)

        return flagged_segments

    def get_review_summary(self, analysis: ConfidenceAnalysis) -> str:
        """Generate a human-readable review summary"""
        lines = [
            "Confidence Analysis Summary",
            "=" * 30,
            f"Overall Confidence: {analysis.overall_confidence:.2%} ({analysis.confidence_level.value})",
            f"Audio Quality Score: {analysis.audio_quality_score:.2%}",
            f"Segments Needing Review: {len(analysis.segments_needing_review)}",
            f"Quality Issues Found: {len(analysis.quality_flags)}",
        ]

        if analysis.quality_flags:
            lines.append("\nIssues Found:")
            for flag in analysis.quality_flags[:5]:  # Show top 5 issues
                lines.append(f"  - Segment {flag.segment_index}: {flag.description}")

            if len(analysis.quality_flags) > 5:
                lines.append(f"  ... and {len(analysis.quality_flags) - 5} more issues")

        if analysis.recommendations:
            lines.append("\nRecommendations:")
            for rec in analysis.recommendations:
                lines.append(f"  • {rec}")

        return "\n".join(lines)
