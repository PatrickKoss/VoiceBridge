from pathlib import Path

import typer

from voicebridge.cli.commands.base import BaseCommands
from voicebridge.cli.utils.command_helpers import (
    display_error,
    display_info,
    display_progress,
)
from voicebridge.domain.models import ExportConfig, OutputFormat, TimestampMode


class ExportCommands(BaseCommands):
    """Commands for exporting transcriptions and analysis."""

    def export_transcription(
        self,
        session_id: str,
        format: str = "txt",
        output_file: str | None = None,
        include_timestamps: bool = False,
        include_confidence: bool = False,
        timestamp_mode: str = "absolute",
    ):
        """Export a transcription session to various formats."""
        if not self.export_service:
            display_error("Export service not available")
            return

        try:
            # Validate format
            try:
                output_format = OutputFormat(format.lower())
            except ValueError:
                available = [f.value for f in OutputFormat]
                display_error(
                    f"Invalid format '{format}'. Available: {', '.join(available)}"
                )
                return

            # Validate timestamp mode
            try:
                ts_mode = TimestampMode(timestamp_mode.lower())
            except ValueError:
                available = [m.value for m in TimestampMode]
                display_error(
                    f"Invalid timestamp mode '{timestamp_mode}'. Available: {', '.join(available)}"
                )
                return

            # Create export config
            export_config = ExportConfig(
                format=output_format,
                include_timestamps=include_timestamps,
                include_confidence=include_confidence,
                timestamp_mode=ts_mode,
                output_file=output_file,
            )

            display_progress(f"Exporting session {session_id} as {format.upper()}...")

            result = self.export_service.export_session(session_id, export_config)

            if result and result.get("success"):
                output_path = result.get("output_path")
                display_progress(f"Export completed: {output_path}", finished=True)

                # Show file info
                if output_path and Path(output_path).exists():
                    file_size = Path(output_path).stat().st_size
                    typer.echo(f"  File size: {file_size:,} bytes")

                    if format in ["srt", "vtt"]:
                        # Show subtitle-specific info
                        lines = len(Path(output_path).read_text().splitlines())
                        typer.echo(f"  Lines: {lines}")
            else:
                error_msg = (
                    result.get("error", "Unknown error") if result else "Export failed"
                )
                display_error(f"Export failed: {error_msg}")

        except Exception as e:
            display_error(f"Error exporting transcription: {e}")

    def list_export_formats(self):
        """List available export formats."""
        typer.echo("Available Export Formats:")
        typer.echo()

        formats = {
            "txt": "Plain text transcription",
            "json": "JSON with metadata and timestamps",
            "srt": "SubRip subtitle format",
            "vtt": "WebVTT subtitle format",
            "csv": "Comma-separated values",
            "xml": "XML format with structure",
            "docx": "Microsoft Word document",
            "pdf": "PDF document",
        }

        for fmt, description in formats.items():
            typer.echo(f"  {fmt:<6} - {description}")

        typer.echo()
        typer.echo("Timestamp Modes:")
        typer.echo("  absolute  - Timestamps from start of audio")
        typer.echo("  relative  - Timestamps relative to each segment")
        typer.echo("  none      - No timestamps")

    def batch_export_sessions(
        self,
        output_dir: str = "exports",
        format: str = "txt",
        include_timestamps: bool = False,
        include_confidence: bool = False,
        session_filter: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ):
        """Export multiple sessions in batch."""
        if not self.export_service or not self.session_service:
            display_error("Required services not available")
            return

        try:
            # Validate format
            try:
                output_format = OutputFormat(format.lower())
            except ValueError:
                available = [f.value for f in OutputFormat]
                display_error(
                    f"Invalid format '{format}'. Available: {', '.join(available)}"
                )
                return

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Get sessions to export
            sessions = self.session_service.list_sessions(
                filter_pattern=session_filter,
                date_from=date_from,
                date_to=date_to,
            )

            if not sessions:
                display_info("No sessions found matching criteria")
                return

            display_info(f"Found {len(sessions)} sessions to export")
            display_progress(f"Exporting to {output_path}...")

            # Create export config
            export_config = ExportConfig(
                format=output_format,
                include_timestamps=include_timestamps,
                include_confidence=include_confidence,
                timestamp_mode=TimestampMode.ABSOLUTE,
            )

            successful = 0
            failed = 0

            for i, session in enumerate(sessions, 1):
                session_id = session.get("id")
                session_name = session.get("name", f"session_{session_id}")

                # Set output file for this session
                filename = f"{session_name}.{format}"
                export_config.output_file = str(output_path / filename)

                typer.echo(f"  [{i}/{len(sessions)}] Exporting {session_name}...")

                try:
                    result = self.export_service.export_session(
                        session_id, export_config
                    )
                    if result and result.get("success"):
                        successful += 1
                    else:
                        failed += 1
                        typer.echo(
                            f"    ❌ Failed: {result.get('error', 'Unknown error') if result else 'Export failed'}"
                        )
                except Exception as e:
                    failed += 1
                    typer.echo(f"    ❌ Error: {e}")

            # Summary
            typer.echo("\nBatch export completed:")
            typer.echo(f"  Successful: {successful}")
            typer.echo(f"  Failed: {failed}")
            typer.echo(f"  Output directory: {output_path}")

        except Exception as e:
            display_error(f"Error in batch export: {e}")

    def analyze_confidence(
        self,
        session_id: str,
        detailed: bool = False,
        threshold: float = 0.7,
    ):
        """Analyze transcription confidence for a session."""
        if not self.confidence_analyzer:
            display_error("Confidence analyzer not available")
            return

        try:
            display_progress(f"Analyzing confidence for session {session_id}...")

            analysis = self.confidence_analyzer.analyze_session(
                session_id, detailed=detailed, threshold=threshold
            )

            if analysis:
                typer.echo(f"\nConfidence Analysis: {session_id}")
                typer.echo("=" * 50)

                # Overall metrics
                overall = analysis.get("overall", {})
                typer.echo(f"Overall Confidence: {overall.get('average', 0):.1%}")
                typer.echo(f"Minimum Confidence: {overall.get('minimum', 0):.1%}")
                typer.echo(f"Maximum Confidence: {overall.get('maximum', 0):.1%}")
                typer.echo(f"Standard Deviation: {overall.get('std_dev', 0):.1%}")

                # Quality categories
                categories = analysis.get("categories", {})
                if categories:
                    typer.echo("\nQuality Distribution:")
                    typer.echo(
                        f"  High Confidence (>{threshold:.0%}): {categories.get('high', 0)} segments"
                    )
                    typer.echo(
                        f"  Medium Confidence: {categories.get('medium', 0)} segments"
                    )
                    typer.echo(
                        f"  Low Confidence (<{threshold:.0%}): {categories.get('low', 0)} segments"
                    )

                # Problematic segments
                low_confidence = analysis.get("low_confidence_segments", [])
                if low_confidence:
                    typer.echo(f"\nLow Confidence Segments ({len(low_confidence)}):")
                    for i, segment in enumerate(
                        low_confidence[:10], 1
                    ):  # Show first 10
                        timestamp = segment.get("timestamp", "Unknown")
                        confidence = segment.get("confidence", 0)
                        text_preview = segment.get("text", "")[:50]
                        typer.echo(
                            f"  {i:2d}. [{timestamp}] {confidence:.1%} - {text_preview}..."
                        )

                    if len(low_confidence) > 10:
                        typer.echo(f"  ... and {len(low_confidence) - 10} more")

                # Detailed analysis
                if detailed and "detailed" in analysis:
                    detailed_info = analysis["detailed"]

                    # Language detection confidence
                    if "language_confidence" in detailed_info:
                        lang = detailed_info["language_confidence"]
                        typer.echo("\nLanguage Detection:")
                        typer.echo(f"  Detected: {lang.get('language', 'Unknown')}")
                        typer.echo(f"  Confidence: {lang.get('confidence', 0):.1%}")

                    # Word-level analysis
                    if "word_analysis" in detailed_info:
                        words = detailed_info["word_analysis"]
                        typer.echo("\nWord-Level Analysis:")
                        typer.echo(f"  Total words: {words.get('total_words', 0)}")
                        typer.echo(
                            f"  Low confidence words: {words.get('low_confidence_words', 0)}"
                        )
                        typer.echo(
                            f"  Average word confidence: {words.get('average_confidence', 0):.1%}"
                        )

                # Recommendations
                recommendations = analysis.get("recommendations", [])
                if recommendations:
                    typer.echo("\nRecommendations:")
                    for rec in recommendations:
                        typer.echo(f"  💡 {rec}")

            else:
                display_error("Confidence analysis failed")

        except Exception as e:
            display_error(f"Error analyzing confidence: {e}")

    def analyze_all_sessions(self, detailed: bool = False, threshold: float = 0.7):
        """Analyze confidence for all sessions."""
        if not self.confidence_analyzer or not self.session_service:
            display_error("Required services not available")
            return

        try:
            sessions = self.session_service.list_sessions()

            if not sessions:
                display_info("No sessions found")
                return

            display_info(f"Analyzing {len(sessions)} sessions...")

            overall_stats = {
                "total_sessions": len(sessions),
                "high_quality": 0,
                "medium_quality": 0,
                "low_quality": 0,
                "total_confidence": 0,
            }

            session_results = []

            for i, session in enumerate(sessions, 1):
                session_id = session.get("id")
                session_name = session.get("name", f"session_{session_id}")

                typer.echo(f"  [{i}/{len(sessions)}] {session_name}...")

                try:
                    analysis = self.confidence_analyzer.analyze_session(
                        session_id, detailed=False, threshold=threshold
                    )

                    if analysis:
                        overall = analysis.get("overall", {})
                        avg_confidence = overall.get("average", 0)
                        overall_stats["total_confidence"] += avg_confidence

                        if avg_confidence >= 0.8:
                            overall_stats["high_quality"] += 1
                            quality = "High"
                        elif avg_confidence >= threshold:
                            overall_stats["medium_quality"] += 1
                            quality = "Medium"
                        else:
                            overall_stats["low_quality"] += 1
                            quality = "Low"

                        session_results.append(
                            {
                                "name": session_name,
                                "id": session_id,
                                "confidence": avg_confidence,
                                "quality": quality,
                            }
                        )

                except Exception as e:
                    typer.echo(f"    ❌ Error analyzing {session_name}: {e}")

            # Show summary
            avg_confidence = (
                overall_stats["total_confidence"] / len(sessions) if sessions else 0
            )

            typer.echo("\nOverall Analysis Summary:")
            typer.echo("=" * 50)
            typer.echo(f"Total Sessions: {overall_stats['total_sessions']}")
            typer.echo(f"Average Confidence: {avg_confidence:.1%}")
            typer.echo(f"High Quality (≥80%): {overall_stats['high_quality']}")
            typer.echo(
                f"Medium Quality (≥{threshold:.0%}): {overall_stats['medium_quality']}"
            )
            typer.echo(
                f"Low Quality (<{threshold:.0%}): {overall_stats['low_quality']}"
            )

            # Show problematic sessions
            low_quality_sessions = [s for s in session_results if s["quality"] == "Low"]
            if low_quality_sessions:
                typer.echo(f"\nSessions Needing Review ({len(low_quality_sessions)}):")
                for session in sorted(
                    low_quality_sessions, key=lambda x: x["confidence"]
                ):
                    typer.echo(f"  • {session['name']}: {session['confidence']:.1%}")

        except Exception as e:
            display_error(f"Error in batch analysis: {e}")

    def set_confidence_thresholds(
        self,
        low_threshold: float = 0.5,
        review_threshold: float = 0.7,
        high_threshold: float = 0.9,
    ):
        """Set confidence thresholds for quality assessment."""
        if not self.confidence_analyzer:
            display_error("Confidence analyzer not available")
            return

        try:
            # Validate thresholds
            if not (0 <= low_threshold <= review_threshold <= high_threshold <= 1):
                display_error(
                    "Thresholds must be in order: 0 ≤ low ≤ review ≤ high ≤ 1"
                )
                return

            success = self.confidence_analyzer.set_thresholds(
                low_threshold, review_threshold, high_threshold
            )

            if success:
                display_progress("Confidence thresholds updated", finished=True)
                typer.echo(f"  Low: {low_threshold:.1%}")
                typer.echo(f"  Review: {review_threshold:.1%}")
                typer.echo(f"  High: {high_threshold:.1%}")
            else:
                display_error("Failed to update thresholds")

        except Exception as e:
            display_error(f"Error setting thresholds: {e}")

    def export_confidence_report(
        self,
        session_id: str,
        output_file: str,
        format: str = "json",
        include_segments: bool = True,
    ):
        """Export detailed confidence analysis report."""
        if not self.confidence_analyzer:
            display_error("Confidence analyzer not available")
            return

        try:
            display_progress(f"Generating confidence report for {session_id}...")

            report = self.confidence_analyzer.generate_report(
                session_id, include_segments=include_segments, format=format
            )

            if report:
                output_path = Path(output_file)

                if format == "json":
                    import json

                    with open(output_path, "w") as f:
                        json.dump(report, f, indent=2)
                elif format == "csv":
                    import csv

                    # Convert report to CSV format
                    with open(output_path, "w", newline="") as f:
                        csv.writer(f)
                        # Write headers and data based on report structure
                        # This would need implementation based on report format
                        pass
                else:
                    display_error(f"Unsupported report format: {format}")
                    return

                display_progress(f"Report saved: {output_path}", finished=True)
            else:
                display_error("Failed to generate confidence report")

        except Exception as e:
            display_error(f"Error exporting confidence report: {e}")
