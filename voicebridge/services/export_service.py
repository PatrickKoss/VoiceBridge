import csv
import json
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

from voicebridge.domain.models import (
    ExportConfig,
    OutputFormat,
    TranscriptionResult,
)
from voicebridge.ports.interfaces import ExportService


class DefaultExportService(ExportService):
    def get_supported_formats(self) -> list[OutputFormat]:
        return [
            OutputFormat.JSON,
            OutputFormat.PLAIN_TEXT,
            OutputFormat.CSV,
            OutputFormat.SRT,
            OutputFormat.VTT,
        ]

    def export_transcription(
        self, result: TranscriptionResult, config: ExportConfig
    ) -> str:
        format_handlers = {
            OutputFormat.JSON: self._export_to_json,
            OutputFormat.PLAIN_TEXT: self._export_to_plain_text,
            OutputFormat.CSV: self._export_to_csv,
            OutputFormat.SRT: self._export_to_srt,
            OutputFormat.VTT: self._export_to_vtt,
        }

        handler = format_handlers.get(config.format)
        if not handler:
            raise ValueError(f"Unsupported format: {config.format}")

        return handler(result, config)

    def export_to_file(self, result: TranscriptionResult, config: ExportConfig) -> bool:
        try:
            content = self.export_transcription(result, config)

            if config.output_file:
                output_path = Path(config.output_file)
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"transcription_{timestamp}.{config.format.value}"
                output_path = Path(filename)

            output_path.write_text(content, encoding="utf-8")
            return True
        except Exception:
            return False

    def _export_to_json(self, result: TranscriptionResult, config: ExportConfig) -> str:
        data = {
            "text": result.text,
            "language": result.language,
            "duration": result.duration,
            "timestamp": datetime.now().isoformat(),
        }

        if config.include_confidence and result.confidence:
            data["confidence"] = result.confidence

        if result.segments:
            segments_data = []
            for segment in result.segments:
                segment_data = {
                    "text": segment.text,
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                }

                if config.include_confidence and segment.confidence:
                    segment_data["confidence"] = segment.confidence

                if config.include_speaker_info and segment.speaker_id is not None:
                    segment_data["speaker_id"] = segment.speaker_id

                segments_data.append(segment_data)

            data["segments"] = segments_data

        if config.include_speaker_info and result.speakers:
            data["speakers"] = [
                {
                    "speaker_id": speaker.speaker_id,
                    "name": speaker.name,
                    "total_speaking_time": speaker.total_speaking_time,
                    "confidence": speaker.confidence,
                }
                for speaker in result.speakers
            ]

        return json.dumps(data, indent=2, ensure_ascii=False)

    def _export_to_plain_text(
        self, result: TranscriptionResult, config: ExportConfig
    ) -> str:
        lines = []

        if result.language:
            lines.append(f"Language: {result.language}")

        if config.include_confidence and result.confidence:
            lines.append(f"Overall Confidence: {result.confidence:.2%}")

        if result.duration:
            lines.append(f"Duration: {result.duration:.2f}s")

        lines.append("")

        if result.segments and len(result.segments) > 1:
            for segment in result.segments:
                timestamp = self._format_timestamp(segment.start_time)
                line = f"[{timestamp}]"

                if config.include_speaker_info and segment.speaker_id is not None:
                    line += f" Speaker {segment.speaker_id}:"

                line += f" {segment.text}"

                if config.include_confidence and segment.confidence:
                    line += f" (confidence: {segment.confidence:.2%})"

                lines.append(line)
        else:
            lines.append(result.text)

        return "\n".join(lines)

    def _export_to_csv(self, result: TranscriptionResult, config: ExportConfig) -> str:
        output = StringIO()

        fieldnames = ["start_time", "end_time", "text"]
        if config.include_speaker_info:
            fieldnames.append("speaker_id")
        if config.include_confidence:
            fieldnames.append("confidence")

        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        if result.segments:
            for segment in result.segments:
                row = {
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "text": segment.text,
                }

                if config.include_speaker_info:
                    row["speaker_id"] = segment.speaker_id

                if config.include_confidence:
                    row["confidence"] = segment.confidence

                writer.writerow(row)
        else:
            row = {
                "start_time": 0.0,
                "end_time": result.duration or 0.0,
                "text": result.text,
            }

            if config.include_speaker_info:
                row["speaker_id"] = None

            if config.include_confidence:
                row["confidence"] = result.confidence

            writer.writerow(row)

        return output.getvalue()

    def _export_to_srt(self, result: TranscriptionResult, config: ExportConfig) -> str:
        lines = []

        if not result.segments:
            lines.extend(
                [
                    "1",
                    f"00:00:00,000 --> {self._seconds_to_srt_timestamp(result.duration or 0)}",
                    result.text,
                    "",
                ]
            )
        else:
            for i, segment in enumerate(result.segments, 1):
                start_time = self._seconds_to_srt_timestamp(segment.start_time)
                end_time = self._seconds_to_srt_timestamp(segment.end_time)

                text = segment.text
                if config.include_speaker_info and segment.speaker_id is not None:
                    text = f"Speaker {segment.speaker_id}: {text}"

                lines.extend([str(i), f"{start_time} --> {end_time}", text, ""])

        return "\n".join(lines)

    def _export_to_vtt(self, result: TranscriptionResult, config: ExportConfig) -> str:
        lines = ["WEBVTT", ""]

        if not result.segments:
            lines.extend(
                [
                    f"00:00:00.000 --> {self._seconds_to_vtt_timestamp(result.duration or 0)}",
                    result.text,
                    "",
                ]
            )
        else:
            for segment in result.segments:
                start_time = self._seconds_to_vtt_timestamp(segment.start_time)
                end_time = self._seconds_to_vtt_timestamp(segment.end_time)

                text = segment.text
                if config.include_speaker_info and segment.speaker_id is not None:
                    text = f"<v Speaker {segment.speaker_id}>{text}"

                lines.extend([f"{start_time} --> {end_time}", text, ""])

        return "\n".join(lines)

    def _format_timestamp(self, seconds: float) -> str:
        minutes, secs = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _seconds_to_srt_timestamp(self, seconds: float) -> str:
        td = timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        secs = int(td.total_seconds() % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    def _seconds_to_vtt_timestamp(self, seconds: float) -> str:
        td = timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        secs = int(td.total_seconds() % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
