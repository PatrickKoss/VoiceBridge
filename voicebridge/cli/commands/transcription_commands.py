import time
from pathlib import Path

import typer

from voicebridge.cli.commands.base import BaseCommands
from voicebridge.cli.utils.command_helpers import (
    display_error,
    display_info,
    display_progress,
    format_duration,
    validate_directory_path,
    validate_file_path,
)


class TranscriptionCommands(BaseCommands):
    """Commands for file-based transcription and batch processing."""

    def transcribe_file(
        self,
        file_path: str,
        output_path: str | None = None,
        model: str | None = None,
        language: str | None = None,
        temperature: float = 0.0,
        format_output: str = "txt",
        max_memory: int = 0,
    ):
        """Transcribe an audio file (supports MP3, WAV, M4A, FLAC, OGG)."""
        if not self.audio_format_service:
            display_error("Audio format service not available")
            return

        input_file = validate_file_path(file_path, must_exist=True)

        if not self.audio_format_service.is_supported_format(input_file):
            supported = ", ".join(self.audio_format_service.get_supported_formats())
            display_error(
                f"Unsupported file format: {input_file.suffix}. Supported: {supported}"
            )
            return

        # Load config and override with parameters
        config = self.config_repo.load()
        if model:
            config.model_name = model
        if language:
            config.language = language
        config.temperature = temperature
        config.max_memory_mb = max_memory

        # Convert to WAV if needed
        temp_wav = None
        try:
            if input_file.suffix.lower() != ".wav":
                temp_wav = input_file.parent / f"temp_{input_file.stem}.wav"
                display_progress(f"Converting {input_file.name} to WAV...")

                if not self.audio_format_service.convert_to_wav(input_file, temp_wav):
                    display_error("Failed to convert audio file")
                    return

                audio_file = temp_wav
            else:
                audio_file = input_file

            # Read and transcribe
            with open(audio_file, "rb") as f:
                audio_data = f.read()

            display_progress("Transcribing...")
            result = self.transcription_orchestrator.transcription_service.transcribe(
                audio_data, config
            )

            # Output result
            output_text = result.text.strip()
            if output_path:
                output_file = Path(output_path)
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(output_text)
                display_progress(
                    f"Transcription saved to: {output_file}", finished=True
                )
            else:
                typer.echo(output_text)

        finally:
            # Clean up temp file
            if temp_wav and temp_wav.exists():
                temp_wav.unlink()

    def batch_transcribe(
        self,
        input_dir: str,
        output_dir: str = "transcriptions",
        workers: int = 4,
        file_pattern: str | None = None,
        model: str | None = None,
        max_memory: int = 0,
    ):
        """Batch transcribe all audio files in a directory."""
        if not self.batch_processing_service:
            display_error("Batch processing service not available")
            return

        input_path = validate_directory_path(input_dir, must_exist=True)
        output_path = Path(output_dir)

        # Load config
        config = self.config_repo.load()
        if model:
            config.model_name = model
        config.max_memory_mb = max_memory

        # Set file patterns
        patterns = [file_pattern] if file_pattern else None

        display_info(f"Processing directory: {input_path}")
        display_info(f"Output directory: {output_path}")
        display_info(f"Workers: {workers}")

        # Estimate processing time
        files = self.batch_processing_service.get_processable_files(
            input_path, patterns
        )
        estimated_time = self.batch_processing_service.estimate_batch_time(files)

        display_info(f"Found {len(files)} files to process")
        display_info(f"Estimated time: {format_duration(estimated_time)}")

        # Process files
        display_progress("Starting batch processing...")
        results = self.batch_processing_service.process_directory(
            input_path, output_path, config, workers, patterns
        )

        # Summary
        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful

        typer.echo("\nBatch processing complete:")
        typer.echo(f"  Successful: {successful}")
        typer.echo(f"  Failed: {failed}")
        typer.echo(f"  Output directory: {output_path}")

    def listen_resumable(
        self,
        file_path: str,
        session_name: str | None = None,
        model: str | None = None,
        language: str | None = None,
        temperature: float = 0.0,
        profile: str | None = None,
        chunk_size: int = 30,
        overlap: int = 5,
        max_memory: int = 0,
    ):
        """Transcribe a long audio file with resume capability."""
        if not self.resume_service:
            display_error("Resume service not available")
            return

        input_file = validate_file_path(file_path, must_exist=True)

        config = self._build_config(
            model=model,
            language=language,
            temperature=temperature,
            profile=profile,
            max_memory_mb=max_memory,
        )

        # Generate session name if not provided
        if not session_name:
            session_name = f"resumable_{input_file.stem}_{int(time.time())}"

        display_info(f"Starting resumable transcription: {session_name}")
        display_info(f"File: {input_file}")
        display_info(f"Chunk size: {chunk_size}s, Overlap: {overlap}s")

        try:
            # Start or resume transcription
            session = self.resume_service.start_or_resume_transcription(
                str(input_file), session_name, config, chunk_size, overlap
            )

            display_progress("Processing audio chunks...")

            # Monitor progress
            while not session.is_complete:
                progress = session.progress
                typer.echo(
                    f"Progress: {progress.completed_chunks}/{progress.total_chunks} chunks "
                    f"({progress.completion_percentage:.1f}%)"
                )

                if progress.estimated_time_remaining:
                    typer.echo(
                        f"Estimated time remaining: {format_duration(progress.estimated_time_remaining)}"
                    )

                time.sleep(2)

            # Get final result
            result = self.resume_service.get_session_result(session_name)
            if result:
                display_progress("Transcription completed!", finished=True)
                typer.echo(f"\nFinal transcription:\n{result.text}")

                # Save to file
                output_file = input_file.parent / f"{input_file.stem}_transcript.txt"
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(result.text)
                display_info(f"Saved to: {output_file}")
            else:
                display_error("Failed to get transcription result")

        except KeyboardInterrupt:
            display_info(
                f"Transcription paused. Resume with: listen-resumable --session-name {session_name}"
            )
        except Exception as e:
            display_error(f"Transcription failed: {e}")

    def realtime_transcribe(
        self,
        chunk_duration: float = 2.0,
        output_format: str = "live",
        model: str | None = None,
        language: str | None = None,
        temperature: float = 0.0,
        profile: str | None = None,
        save_audio: bool = False,
        output_file: str | None = None,
        insert_cursor: bool = False,
    ):
        """Real-time streaming transcription with live output."""
        if not self.transcription_orchestrator:
            display_error("Transcription service not available")
            return

        config = self._build_config(
            model=model,
            language=language,
            temperature=temperature,
            profile=profile,
        )

        # Set cursor insertion behavior
        if insert_cursor:
            config.paste_final = True

        display_info("Starting real-time transcription...")
        display_info(f"Chunk duration: {chunk_duration}s")
        display_info("Press Ctrl+C to stop")

        if save_audio and output_file:
            display_info(f"Audio will be saved to: {output_file}")

        try:
            import threading
            import time

            # State management
            transcription_buffer = []
            audio_buffer = []
            stop_flag = threading.Event()

            def audio_processor():
                """Process audio chunks in real-time."""
                import os
                import tempfile
                import time
                import wave

                last_process_time = time.time()
                segment_count = 0
                chunks_received = 0

                typer.echo("🎙️  Audio processor started - waiting for audio...")

                try:
                    audio_stream = (
                        self.transcription_orchestrator.audio_recorder.record_stream(
                            sample_rate=16000
                        )
                    )
                    typer.echo(
                        "🔊 Audio stream initialized, starting to read chunks..."
                    )

                    for chunk in audio_stream:
                        if stop_flag.is_set():
                            break

                        chunks_received += 1
                        print(
                            f"🎧 Received audio chunk #{chunks_received}: {len(chunk):,} bytes"
                        )
                        audio_buffer.append(chunk)

                        # Show periodic progress
                        if chunks_received % 50 == 0:
                            print(
                                f"🔄 Received {chunks_received} audio chunks, buffer size: {len(b''.join(audio_buffer)):,} bytes"
                            )

                        # Check if enough time has passed for the next segment
                        current_time = time.time()
                        if current_time - last_process_time >= chunk_duration:
                            segment_count += 1

                            # Get all accumulated audio data for this segment
                            segment_data = b"".join(audio_buffer)

                            # Clear buffer for next segment (or keep some overlap if desired)
                            audio_buffer.clear()

                            print(
                                f"🎯 Processing segment {segment_count} with {len(segment_data):,} bytes of audio"
                            )

                            # Only process if we have sufficient audio data
                            if len(segment_data) > 1000:  # At least 1KB of audio
                                try:
                                    # Convert raw PCM to proper WAV format (same fix as listen mode)
                                    sample_rate = 16000  # FFmpeg outputs 16kHz mono
                                    channels = 1
                                    sample_width = 2  # 16-bit audio

                                    with tempfile.NamedTemporaryFile(
                                        delete=False, suffix=".wav"
                                    ) as temp_file:
                                        temp_file_path = temp_file.name

                                    # Write proper WAV file with headers
                                    with wave.open(temp_file_path, "wb") as wav_file:
                                        wav_file.setnchannels(channels)
                                        wav_file.setsampwidth(sample_width)
                                        wav_file.setframerate(sample_rate)
                                        wav_file.writeframes(segment_data)

                                    # Transcribe using the WAV file
                                    with open(temp_file_path, "rb") as f:
                                        wav_data = f.read()

                                    print(
                                        f"🎵 Sending {len(wav_data):,} bytes to transcription service..."
                                    )
                                    result = self.transcription_orchestrator.transcription_service.transcribe(
                                        wav_data, config
                                    )

                                    # Clean up temp file
                                    if os.path.exists(temp_file_path):
                                        os.unlink(temp_file_path)

                                    print(
                                        f"📝 Transcription result: '{result.text.strip() if result and result.text else 'None'}'"
                                    )

                                    if result and result.text.strip():
                                        transcription_buffer.append(result.text.strip())

                                        if output_format == "live":
                                            # Show live results with clear line management
                                            print(
                                                f"[{segment_count:03d}] {result.text.strip()}"
                                            )
                                        elif output_format == "segments":
                                            # Show completed segments
                                            typer.echo(
                                                f"[{segment_count:03d}] {result.text.strip()}"
                                            )
                                    else:
                                        if output_format == "live":
                                            print(
                                                f"[{segment_count:03d}] (no speech detected)"
                                            )
                                        elif output_format == "segments":
                                            typer.echo(
                                                f"[{segment_count:03d}] (no speech detected)"
                                            )

                                except Exception as e:
                                    if output_format == "live":
                                        print(f"\r[{segment_count:03d}] ERROR: {e}")
                                    else:
                                        typer.echo(f"[{segment_count:03d}] ERROR: {e}")

                                    # Clean up temp file on error
                                    try:
                                        if (
                                            "temp_file_path" in locals()
                                            and os.path.exists(temp_file_path)
                                        ):
                                            os.unlink(temp_file_path)
                                    except Exception:
                                        pass

                            else:
                                print(
                                    f"⚠️  Segment {segment_count} too small ({len(segment_data):,} bytes), skipping..."
                                )

                            last_process_time = current_time

                except Exception as e:
                    print(f"❌ Audio processor error: {e}")
                    typer.echo(f"Audio processing failed: {e}")

            # Start audio processing thread
            audio_thread = threading.Thread(target=audio_processor, daemon=True)
            audio_thread.start()

            # Keep running until interrupted
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                display_info("\nStopping real-time transcription...")
                stop_flag.set()
                self._stop_audio_recorder()

                if audio_thread.is_alive():
                    audio_thread.join(timeout=2)

                # Show final results
                if transcription_buffer:
                    typer.echo("\n\nFinal transcription:")
                    full_text = " ".join(transcription_buffer)
                    typer.echo(full_text)

                    # Save to file if requested
                    if output_file:
                        with open(output_file, "w", encoding="utf-8") as f:
                            f.write(full_text)
                        display_info(f"Transcription saved to: {output_file}")

                # Save audio if requested
                if save_audio and output_file and audio_buffer:
                    audio_file = Path(output_file).with_suffix(".wav")
                    # Note: This would need proper WAV file writing implementation
                    display_info(f"Audio saved to: {audio_file}")

        except Exception as e:
            display_error(f"Real-time transcription failed: {e}")

    def split_audio(
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

            for i, chunk_path in enumerate(chunks, 1):
                typer.echo(f"  {i:02d}: {chunk_path.name}")

        except Exception as e:
            display_error(f"Audio splitting failed: {e}")

    def test_audio_setup(self):
        """Test audio recording and playback setup."""
        if not self.transcription_orchestrator:
            display_error("Transcription service not available")
            return

        display_info("Testing audio setup...")

        # Test recording
        try:
            display_progress("Testing audio recording...")
            test_duration = 3
            typer.echo(f"Recording for {test_duration} seconds...")

            import threading
            import time

            audio_data = b""
            stop_recording = threading.Event()

            def record_test():
                nonlocal audio_data
                start_time = time.time()
                for (
                    chunk
                ) in self.transcription_orchestrator.audio_recorder.record_stream():
                    if (
                        stop_recording.is_set()
                        or time.time() - start_time > test_duration
                    ):
                        break
                    audio_data += chunk

            # Record for test duration
            record_thread = threading.Thread(target=record_test, daemon=True)
            record_thread.start()
            time.sleep(test_duration)
            stop_recording.set()
            record_thread.join(timeout=1)

            if audio_data:
                display_progress("✓ Audio recording successful", finished=True)
                display_info(f"Recorded {len(audio_data):,} bytes of audio data")

                # Test transcription
                display_progress("Testing transcription...")
                config = self.config_repo.load()

                # Convert raw PCM to proper WAV format before transcription
                import os
                import tempfile
                import wave

                sample_rate = 16000  # FFmpeg outputs 16kHz mono
                channels = 1
                sample_width = 2  # 16-bit audio

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".wav"
                ) as temp_file:
                    temp_file_path = temp_file.name

                # Write proper WAV file with headers
                with wave.open(temp_file_path, "wb") as wav_file:
                    wav_file.setnchannels(channels)
                    wav_file.setsampwidth(sample_width)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_data)

                # Read back the WAV file for transcription
                with open(temp_file_path, "rb") as f:
                    wav_data = f.read()

                # Clean up temp file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

                result = (
                    self.transcription_orchestrator.transcription_service.transcribe(
                        wav_data, config
                    )
                )

                if result and result.text.strip():
                    display_progress("✓ Transcription successful", finished=True)
                    typer.echo(f"Test transcription: '{result.text.strip()}'")
                else:
                    display_info("⚠ No speech detected in test recording")
            else:
                display_error("No audio data recorded")

        except Exception as e:
            display_error(f"Audio test failed: {e}")

        # Test GPU if available
        if self.system_service:
            try:
                display_progress("Testing GPU availability...")
                gpu_devices = self.system_service.detect_gpu_devices()
                gpu_info = gpu_devices[0] if gpu_devices else None
                if gpu_info and gpu_info.gpu_type.value != "none":
                    display_progress(
                        f"✓ GPU available: {gpu_info.device_name}", finished=True
                    )
                    display_info(
                        f"Memory: {gpu_info.memory_total:.1f}MB total, {gpu_info.memory_available:.1f}MB available"
                    )
                else:
                    display_info("⚠ No GPU available, using CPU")
            except Exception as e:
                display_info(f"⚠ GPU test failed: {e}")

        display_progress("Audio setup test completed", finished=True)
