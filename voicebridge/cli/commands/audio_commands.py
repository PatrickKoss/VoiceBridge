from pathlib import Path

import typer

from voicebridge.cli.commands.base import BaseCommands
from voicebridge.cli.utils.command_helpers import (
    display_error,
    display_info,
    display_progress,
    format_duration,
    format_file_size,
    validate_directory_path,
    validate_file_path,
)


class AudioCommands(BaseCommands):
    """Commands for audio processing and management."""

    def audio_info(self, file_path: str):
        """Show information about an audio file."""
        if not self.audio_format_service:
            display_error("Audio format service not available")
            return

        input_file = validate_file_path(file_path, must_exist=True)

        try:
            info = self.audio_format_service.get_audio_info(input_file)

            if info:
                typer.echo(f"Audio File Information: {input_file.name}")
                typer.echo(f"  Format: {info.get('format', 'Unknown')}")
                typer.echo(f"  Duration: {format_duration(info.get('duration', 0))}")
                typer.echo(f"  Sample Rate: {info.get('sample_rate', 0)} Hz")
                typer.echo(f"  Channels: {info.get('channels', 0)}")
                typer.echo(f"  Bit Depth: {info.get('bit_depth', 0)} bits")
                typer.echo(f"  Bitrate: {info.get('bitrate', 0)} kbps")
                typer.echo(f"  File Size: {format_file_size(info.get('file_size', 0))}")

                # Show codec information if available
                if "codec" in info:
                    typer.echo(f"  Codec: {info['codec']}")

                # Show quality metrics if available
                if "quality_score" in info:
                    typer.echo(f"  Quality Score: {info['quality_score']:.1f}/10")
            else:
                display_error("Could not read audio file information")

        except Exception as e:
            display_error(f"Error reading audio info: {e}")

    def audio_formats(self):
        """List supported audio formats."""
        if not self.audio_format_service:
            display_error("Audio format service not available")
            return

        try:
            formats = self.audio_format_service.get_supported_formats()

            typer.echo("Supported Audio Formats:")

            # Formats is a list of strings
            if formats:
                for fmt in sorted(formats):
                    typer.echo(f"  .{fmt}")
            else:
                typer.echo("  No formats available")

        except Exception as e:
            display_error(f"Error getting supported formats: {e}")

    def audio_convert(
        self,
        input_file: str,
        output_file: str,
        format: str | None = None,
        sample_rate: int | None = None,
        channels: int | None = None,
        bitrate: int | None = None,
    ):
        """Convert audio file to different format."""
        if not self.audio_format_service:
            display_error("Audio format service not available")
            return

        input_path = validate_file_path(input_file, must_exist=True)
        output_path = Path(output_file)

        # Determine output format from file extension if not specified
        if not format:
            format = output_path.suffix.lstrip(".")

        try:
            display_progress(f"Converting {input_path.name} to {format.upper()}...")

            success = self.audio_format_service.convert_audio(
                input_path,
                output_path,
                output_format=format,
                sample_rate=sample_rate,
                channels=channels,
                bitrate=bitrate,
            )

            if success:
                display_progress(f"Conversion completed: {output_path}", finished=True)

                # Show output file info
                if output_path.exists():
                    size = format_file_size(output_path.stat().st_size)
                    typer.echo(f"  Output size: {size}")
            else:
                display_error("Audio conversion failed")

        except Exception as e:
            display_error(f"Error converting audio: {e}")

    def audio_preprocess(
        self,
        input_file: str,
        output_file: str,
        noise_reduction: float = 0.0,
        normalize: bool = False,
        trim_silence: bool = False,
        silence_threshold: float = 0.01,
        fade_in: float = 0.0,
        fade_out: float = 0.0,
    ):
        """Preprocess audio with noise reduction, normalization, etc."""
        if not self.audio_preprocessing_service:
            display_error("Audio preprocessing service not available")
            return

        input_path = validate_file_path(input_file, must_exist=True)
        output_path = Path(output_file)

        try:
            display_progress(f"Preprocessing {input_path.name}...")

            # Build preprocessing options
            options = {
                "noise_reduction": noise_reduction,
                "normalize": normalize,
                "trim_silence": trim_silence,
                "silence_threshold": silence_threshold,
                "fade_in": fade_in,
                "fade_out": fade_out,
            }

            success = self.audio_preprocessing_service.preprocess_audio(
                input_path, output_path, options
            )

            if success:
                display_progress(
                    f"Preprocessing completed: {output_path}", finished=True
                )

                # Show processing results
                if output_path.exists():
                    original_size = input_path.stat().st_size
                    processed_size = output_path.stat().st_size

                    typer.echo(f"  Original size: {format_file_size(original_size)}")
                    typer.echo(f"  Processed size: {format_file_size(processed_size)}")

                    if processed_size != original_size:
                        ratio = processed_size / original_size
                        typer.echo(f"  Size ratio: {ratio:.2f}x")
            else:
                display_error("Audio preprocessing failed")

        except Exception as e:
            display_error(f"Error preprocessing audio: {e}")

    def audio_split(
        self,
        file_path: str,
        output_dir: str = "split_audio",
        method: str = "duration",
        chunk_duration: int = 300,
        silence_threshold: float = 0.01,
        max_size_mb: float = 25.0,
    ):
        """Split audio file into chunks."""
        if not self.audio_splitting_service:
            display_error("Audio splitting service not available")
            return

        input_file = validate_file_path(file_path, must_exist=True)
        output_path = Path(output_dir)

        display_info(f"Splitting audio file: {input_file}")
        display_info(f"Method: {method}")
        display_info(f"Output directory: {output_path}")

        if method == "duration":
            display_info(f"Chunk duration: {chunk_duration}s")
        elif method == "silence":
            display_info(f"Silence threshold: {silence_threshold}")
        elif method == "size":
            display_info(f"Max size: {max_size_mb}MB")

        try:
            # Ensure output directory exists
            output_path.mkdir(parents=True, exist_ok=True)

            # Split the audio
            if method == "duration":
                chunks = self.audio_splitting_service.split_by_duration(
                    input_file, output_path, chunk_duration
                )
            elif method == "silence":
                chunks = self.audio_splitting_service.split_by_silence(
                    input_file, output_path, silence_threshold
                )
            elif method == "size":
                chunks = self.audio_splitting_service.split_by_size(
                    input_file, output_path, max_size_mb
                )
            else:
                display_error(f"Unknown splitting method: {method}")
                return

            display_progress(f"Audio split into {len(chunks)} chunks", finished=True)

            total_size = 0
            for i, chunk_path in enumerate(chunks, 1):
                chunk_size = chunk_path.stat().st_size
                total_size += chunk_size
                typer.echo(
                    f"  {i:02d}: {chunk_path.name} ({format_file_size(chunk_size)})"
                )

            typer.echo(f"\nTotal output size: {format_file_size(total_size)}")

        except Exception as e:
            display_error(f"Audio splitting failed: {e}")

    def audio_merge(
        self,
        input_dir: str,
        output_file: str,
        pattern: str = "*.wav",
        sort_by: str = "name",
    ):
        """Merge multiple audio files into one."""
        if not self.audio_format_service:
            display_error("Audio format service not available")
            return

        input_path = validate_directory_path(input_dir, must_exist=True)
        output_path = Path(output_file)

        try:
            # Find audio files
            if sort_by == "name":
                audio_files = sorted(input_path.glob(pattern))
            elif sort_by == "date":
                audio_files = sorted(
                    input_path.glob(pattern), key=lambda p: p.stat().st_mtime
                )
            else:
                audio_files = list(input_path.glob(pattern))

            if not audio_files:
                display_error(f"No audio files found matching pattern: {pattern}")
                return

            display_info(f"Found {len(audio_files)} files to merge")
            display_progress("Merging audio files...")

            success = self.audio_format_service.merge_audio_files(
                audio_files, output_path
            )

            if success:
                display_progress(f"Audio merged: {output_path}", finished=True)

                if output_path.exists():
                    size = format_file_size(output_path.stat().st_size)
                    typer.echo(f"  Output size: {size}")
            else:
                display_error("Audio merge failed")

        except Exception as e:
            display_error(f"Error merging audio: {e}")

    def audio_enhance(
        self,
        input_file: str,
        output_file: str,
        enhance_speech: bool = True,
        remove_noise: bool = True,
        amplify: float = 1.0,
        equalize: bool = False,
    ):
        """Enhance audio quality for better transcription."""
        if not self.audio_preprocessing_service:
            display_error("Audio preprocessing service not available")
            return

        input_path = validate_file_path(input_file, must_exist=True)
        output_path = Path(output_file)

        try:
            display_progress(f"Enhancing audio: {input_path.name}")

            enhancement_options = {
                "enhance_speech": enhance_speech,
                "remove_noise": remove_noise,
                "amplify": amplify,
                "equalize": equalize,
            }

            success = self.audio_preprocessing_service.enhance_for_transcription(
                input_path, output_path, enhancement_options
            )

            if success:
                display_progress(f"Audio enhanced: {output_path}", finished=True)

                # Show quality comparison if available
                original_quality = self.audio_preprocessing_service.assess_quality(
                    input_path
                )
                enhanced_quality = self.audio_preprocessing_service.assess_quality(
                    output_path
                )

                if original_quality and enhanced_quality:
                    typer.echo(
                        f"  Quality improvement: {original_quality:.1f} -> {enhanced_quality:.1f}"
                    )
            else:
                display_error("Audio enhancement failed")

        except Exception as e:
            display_error(f"Error enhancing audio: {e}")

    def audio_analyze(self, file_path: str, detailed: bool = False):
        """Analyze audio file for quality and characteristics."""
        if not self.audio_preprocessing_service:
            display_error("Audio preprocessing service not available")
            return

        input_file = validate_file_path(file_path, must_exist=True)

        try:
            display_progress(f"Analyzing audio: {input_file.name}")

            analysis = self.audio_preprocessing_service.analyze_audio(
                input_file, detailed
            )

            if analysis:
                typer.echo(f"\nAudio Analysis: {input_file.name}")
                typer.echo("=" * 50)

                # Basic metrics
                typer.echo(f"Quality Score: {analysis.get('quality_score', 0):.1f}/10")
                typer.echo(
                    f"Transcription Suitability: {analysis.get('transcription_score', 0):.1f}/10"
                )

                # Audio characteristics
                if "characteristics" in analysis:
                    char = analysis["characteristics"]
                    typer.echo(f"Signal-to-Noise Ratio: {char.get('snr', 0):.1f} dB")
                    typer.echo(
                        f"Speech Percentage: {char.get('speech_percentage', 0):.1%}"
                    )
                    typer.echo(
                        f"Silence Percentage: {char.get('silence_percentage', 0):.1%}"
                    )
                    typer.echo(
                        f"Background Noise Level: {char.get('noise_level', 'Unknown')}"
                    )

                # Recommendations
                if "recommendations" in analysis:
                    typer.echo("\nRecommendations:")
                    for rec in analysis["recommendations"]:
                        typer.echo(f"  • {rec}")

                # Detailed analysis
                if detailed and "detailed" in analysis:
                    detailed_info = analysis["detailed"]
                    typer.echo("\nDetailed Analysis:")

                    if "frequency_analysis" in detailed_info:
                        freq = detailed_info["frequency_analysis"]
                        typer.echo(
                            f"  Frequency Range: {freq.get('min_freq', 0):.0f} - {freq.get('max_freq', 0):.0f} Hz"
                        )
                        typer.echo(
                            f"  Dominant Frequency: {freq.get('dominant_freq', 0):.0f} Hz"
                        )

                    if "temporal_analysis" in detailed_info:
                        temp = detailed_info["temporal_analysis"]
                        typer.echo(
                            f"  Speech Segments: {temp.get('speech_segments', 0)}"
                        )
                        typer.echo(
                            f"  Average Segment Length: {format_duration(temp.get('avg_segment_length', 0))}"
                        )
                        typer.echo(f"  Silence Gaps: {temp.get('silence_gaps', 0)}")

            else:
                display_error("Audio analysis failed")

        except Exception as e:
            display_error(f"Error analyzing audio: {e}")

    def audio_validate(self, file_path: str):
        """Validate audio file for transcription compatibility."""
        if not self.audio_format_service:
            display_error("Audio format service not available")
            return

        input_file = validate_file_path(file_path, must_exist=True)

        try:
            validation = self.audio_format_service.validate_for_transcription(
                input_file
            )

            typer.echo(f"Audio Validation: {input_file.name}")
            typer.echo("=" * 50)

            if validation.get("valid", False):
                display_progress(
                    "✓ Audio file is valid for transcription", finished=True
                )
            else:
                typer.echo("✗ Audio file has issues:")

            # Show validation details
            issues = validation.get("issues", [])
            warnings = validation.get("warnings", [])

            if issues:
                typer.echo("\nIssues (must be fixed):")
                for issue in issues:
                    typer.echo(f"  ❌ {issue}")

            if warnings:
                typer.echo("\nWarnings (may affect quality):")
                for warning in warnings:
                    typer.echo(f"  ⚠️ {warning}")

            # Show recommendations
            recommendations = validation.get("recommendations", [])
            if recommendations:
                typer.echo("\nRecommendations:")
                for rec in recommendations:
                    typer.echo(f"  💡 {rec}")

        except Exception as e:
            display_error(f"Error validating audio: {e}")
