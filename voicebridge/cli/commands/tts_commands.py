import os
import time

import typer

from voicebridge.cli.commands.base import BaseCommands
from voicebridge.cli.utils.command_helpers import (
    display_error,
    display_info,
    display_progress,
)
from voicebridge.domain.models import TTSMode, TTSOutputMode, TTSStreamingMode


class TTSCommands(BaseCommands):
    """Commands for Text-to-Speech functionality."""

    def tts_generate(
        self,
        text: str,
        voice: str | None = None,
        streaming: bool = False,
        output_file: str | None = None,
        auto_play: bool = True,
        cfg_scale: float | None = None,
        inference_steps: int | None = None,
        sample_rate: int | None = None,
        use_gpu: bool | None = None,
    ):
        """Generate TTS from provided text."""
        if not self.tts_orchestrator:
            display_error("TTS service not available")
            return

        config = self.config_repo.load()
        tts_config = config.tts_config

        # Override config with command parameters
        if voice:
            tts_config.default_voice = voice
        if output_file:
            tts_config.output_file_path = output_file
            tts_config.output_mode = (
                TTSOutputMode.SAVE_FILE if not auto_play else TTSOutputMode.BOTH
            )
        if cfg_scale is not None:
            tts_config.cfg_scale = cfg_scale
        if inference_steps is not None:
            tts_config.inference_steps = inference_steps
        if sample_rate is not None:
            tts_config.sample_rate = sample_rate
        if use_gpu is not None:
            tts_config.use_gpu = use_gpu

        tts_config.streaming_mode = (
            TTSStreamingMode.STREAMING if streaming else TTSStreamingMode.NON_STREAMING
        )
        tts_config.auto_play = auto_play

        preview_text = text[:100] + ("..." if len(text) > 100 else "")
        display_progress(f"Generating TTS for: {preview_text}")

        try:
            success = self.tts_orchestrator.generate_tts_from_text(text, tts_config)
            if success:
                display_progress("TTS generation completed", finished=True)
            else:
                display_error("TTS generation failed")
        except Exception as e:
            display_error(f"TTS generation error: {e}")

    def tts_generate_file(
        self,
        input_file: str,
        voice: str | None = None,
        streaming: bool = False,
        output_file: str | None = None,
        auto_play: bool = False,
        cfg_scale: float | None = None,
        inference_steps: int | None = None,
        sample_rate: int | None = None,
        use_gpu: bool | None = None,
    ):
        """Generate TTS from a text file."""
        if not self.tts_orchestrator:
            display_error("TTS service not available")
            return

        # Read the input file
        try:
            from pathlib import Path

            input_path = Path(input_file)
            if not input_path.exists():
                display_error(f"Input file not found: {input_file}")
                return

            if not input_path.is_file():
                display_error(f"Input path is not a file: {input_file}")
                return

            # Read file content with encoding detection
            try:
                text = input_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                # Try with different encodings
                for encoding in ["latin-1", "cp1252", "iso-8859-1"]:
                    try:
                        text = input_path.read_text(encoding=encoding)
                        display_info(f"File read with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    display_error("Could not read file with any known encoding")
                    return

            if not text.strip():
                display_error("Input file is empty")
                return

            display_info(f"Read {len(text)} characters from {input_file}")

        except Exception as e:
            display_error(f"Error reading input file: {e}")
            return

        # Set default output file if not provided
        if not output_file:
            output_file = str(input_path.with_suffix(".wav"))
            display_info(f"Output file not specified, using: {output_file}")

        # Load config
        config = self.config_repo.load()
        tts_config = config.tts_config

        # Override config with command parameters
        if voice:
            tts_config.default_voice = voice
        tts_config.output_file_path = output_file
        tts_config.output_mode = (
            TTSOutputMode.SAVE_FILE if not auto_play else TTSOutputMode.BOTH
        )
        if cfg_scale is not None:
            tts_config.cfg_scale = cfg_scale
        if inference_steps is not None:
            tts_config.inference_steps = inference_steps
        if sample_rate is not None:
            tts_config.sample_rate = sample_rate
        if use_gpu is not None:
            tts_config.use_gpu = use_gpu

        tts_config.streaming_mode = (
            TTSStreamingMode.STREAMING if streaming else TTSStreamingMode.NON_STREAMING
        )
        tts_config.auto_play = auto_play

        preview_text = text[:100] + ("..." if len(text) > 100 else "")
        display_progress(f"Generating TTS for: {preview_text}")
        display_info(f"Total text length: {len(text)} characters")

        try:
            success = self.tts_orchestrator.generate_tts_from_text(text, tts_config)
            if success:
                display_progress("TTS generation completed", finished=True)
                display_info(f"Audio saved to: {output_file}")
            else:
                display_error("TTS generation failed")
        except Exception as e:
            display_error(f"TTS generation error: {e}")

    def tts_listen_clipboard(
        self,
        voice: str | None = None,
        streaming: bool = False,
        auto_play: bool = True,
        output_file: str | None = None,
    ):
        """Listen to clipboard changes and generate TTS."""
        if not self.tts_orchestrator:
            display_error("TTS service not available")
            return

        config = self.config_repo.load()
        tts_config = config.tts_config

        # Override config with command parameters
        tts_config.tts_mode = TTSMode.CLIPBOARD
        if voice:
            tts_config.default_voice = voice
        if output_file:
            tts_config.output_file_path = output_file
            tts_config.output_mode = (
                TTSOutputMode.SAVE_FILE if not auto_play else TTSOutputMode.BOTH
            )
        tts_config.streaming_mode = (
            TTSStreamingMode.STREAMING if streaming else TTSStreamingMode.NON_STREAMING
        )
        tts_config.auto_play = auto_play

        display_info("Starting TTS clipboard monitoring...")
        display_info(f"Voice: {tts_config.default_voice}")
        display_info(f"Mode: {'Streaming' if streaming else 'Non-streaming'}")
        typer.echo("Copy text to clipboard to generate TTS. Press Ctrl+C to stop.")

        try:
            self.tts_orchestrator.start_tts_mode(tts_config)

            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            typer.echo("\nStopping clipboard monitoring...")
        finally:
            self.tts_orchestrator.stop_tts()

    def tts_listen_selection(
        self,
        voice: str | None = None,
        streaming: bool = False,
        auto_play: bool = True,
    ):
        """Listen for text selections and generate TTS via hotkey."""
        if not self.tts_orchestrator:
            display_error("TTS service not available")
            return

        config = self.config_repo.load()
        tts_config = config.tts_config

        # Override config with command parameters
        tts_config.tts_mode = TTSMode.MOUSE
        if voice:
            tts_config.default_voice = voice
        tts_config.streaming_mode = (
            TTSStreamingMode.STREAMING if streaming else TTSStreamingMode.NON_STREAMING
        )
        tts_config.auto_play = auto_play

        display_info("Starting TTS selection mode...")
        display_info(f"Voice: {tts_config.default_voice}")
        display_info(f"Generate hotkey: {tts_config.tts_generate_key}")
        display_info(f"Stop hotkey: {tts_config.tts_stop_key}")
        typer.echo(
            "Select text and press hotkey to generate TTS. Press Ctrl+C to stop."
        )

        try:
            self.tts_orchestrator.start_tts_mode(tts_config)

            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            typer.echo("\nStopping selection monitoring...")
        finally:
            self.tts_orchestrator.stop_tts()

    def tts_daemon_start(
        self,
        voice: str | None = None,
        mode: str = "clipboard",
        streaming: bool = False,
        auto_play: bool = True,
        background: bool = False,
    ):
        """Start TTS daemon with hotkey support."""
        if not self.tts_daemon_service:
            display_error("TTS daemon service not available")
            return

        # Check if daemon is already running
        if self.tts_daemon_service.is_daemon_running():
            display_error("TTS daemon is already running")
            return

        config = self.config_repo.load()
        tts_config = config.tts_config

        # Set mode
        if mode == "clipboard":
            tts_config.tts_mode = TTSMode.CLIPBOARD
        elif mode == "selection":
            tts_config.tts_mode = TTSMode.MOUSE
        else:
            display_error(f"Invalid mode: {mode}. Use 'clipboard' or 'selection'")
            return

        # Override config
        if voice:
            tts_config.default_voice = voice
        tts_config.streaming_mode = (
            TTSStreamingMode.STREAMING if streaming else TTSStreamingMode.NON_STREAMING
        )
        tts_config.auto_play = auto_play

        # Check if we're being called recursively in background mode
        if background and os.environ.get("VOICEBRIDGE_NO_BACKGROUND"):
            background = False

        if background:
            # Start daemon in background
            import subprocess
            import sys

            cmd = [
                sys.executable,
                "-m",
                "voicebridge",
                "tts",
                "daemon",
                "start",
                "--mode",
                mode,
                "--voice",
                voice or tts_config.default_voice,
            ]

            if streaming:
                cmd.append("--streaming")
            if not auto_play:
                cmd.append("--no-auto-play")

            # Set environment variable to prevent recursive background calls
            env = os.environ.copy()
            env["VOICEBRIDGE_NO_BACKGROUND"] = "1"

            try:
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                display_progress(
                    f"TTS daemon started in background (PID: {process.pid})",
                    finished=True,
                )
                return
            except Exception as e:
                display_error(f"Failed to start daemon in background: {e}")
                return

        # Start daemon in foreground
        display_info(f"Starting TTS daemon in {mode} mode...")
        display_info(f"Voice: {tts_config.default_voice}")
        display_info(f"Streaming: {streaming}")
        display_info("Press Ctrl+C to stop the daemon")

        try:
            success = self.tts_daemon_service.start_daemon(tts_config)
            if success:
                display_progress("TTS daemon started successfully", finished=True)

                # Keep daemon running
                while self.tts_daemon_service.is_daemon_running():
                    time.sleep(1)
            else:
                display_error("Failed to start TTS daemon")

        except KeyboardInterrupt:
            typer.echo("\nStopping TTS daemon...")
        finally:
            self.tts_daemon_service.stop_daemon()

    def tts_daemon_stop(self):
        """Stop the TTS daemon."""
        if not self.tts_daemon_service:
            display_error("TTS daemon service not available")
            return

        if not self.tts_daemon_service.is_daemon_running():
            display_info("TTS daemon is not running")
            return

        try:
            success = self.tts_daemon_service.stop_daemon()
            if success:
                display_progress("TTS daemon stopped", finished=True)
            else:
                display_error("Failed to stop TTS daemon")
        except Exception as e:
            display_error(f"Error stopping daemon: {e}")

    def tts_daemon_status(self):
        """Show TTS daemon status."""
        if not self.tts_daemon_service:
            display_error("TTS daemon service not available")
            return

        try:
            status = self.tts_daemon_service.get_daemon_status()

            typer.echo("TTS Daemon Status:")
            typer.echo(f"  Running: {'Yes' if status.get('running', False) else 'No'}")

            if status.get("running"):
                typer.echo(f"  PID: {status.get('pid', 'Unknown')}")
                typer.echo(f"  Mode: {status.get('mode', 'Unknown')}")
                typer.echo(f"  Voice: {status.get('voice', 'Unknown')}")
                typer.echo(f"  Uptime: {status.get('uptime', 'Unknown')}")

                # Show performance stats if available
                if "stats" in status:
                    stats = status["stats"]
                    typer.echo(
                        f"  Requests processed: {stats.get('requests_processed', 0)}"
                    )
                    typer.echo(
                        f"  Audio generated: {stats.get('audio_generated_seconds', 0):.1f}s"
                    )
                    typer.echo(f"  Errors: {stats.get('errors', 0)}")

        except Exception as e:
            display_error(f"Error getting daemon status: {e}")

    def tts_list_voices(self):
        """List available TTS voices."""
        if not self.tts_orchestrator:
            display_error("TTS service not available")
            return

        try:
            # Get config first
            config = self.config_repo.load()
            tts_config = config.tts_config
            voices = self.tts_orchestrator.list_available_voices(tts_config)
            if voices:
                typer.echo("Available TTS voices:")
                for voice_id, info in voices.items():
                    typer.echo(f"  {voice_id}")
                    if info.display_name:
                        typer.echo(f"    Display Name: {info.display_name}")
                    if info.language:
                        typer.echo(f"    Language: {info.language}")
                    if info.gender:
                        typer.echo(f"    Gender: {info.gender}")
                    typer.echo(f"    Sample: {info.file_path}")
                    typer.echo()
            else:
                display_info(
                    "No TTS voices found. Please add voice samples to the voices directory."
                )

        except Exception as e:
            display_error(f"Error listing voices: {e}")

    def tts_config_show(self):
        """Show current TTS configuration."""
        try:
            config = self.config_repo.load()
            tts_config = config.tts_config

            typer.echo("TTS Configuration:")
            typer.echo(f"  Model path: {tts_config.model_path}")
            typer.echo(f"  Voice samples dir: {tts_config.voice_samples_dir}")
            typer.echo(f"  Default voice: {tts_config.default_voice}")
            typer.echo(f"  CFG scale: {tts_config.cfg_scale}")
            typer.echo(f"  Inference steps: {tts_config.inference_steps}")
            typer.echo(f"  TTS mode: {tts_config.tts_mode.value}")
            typer.echo(f"  Streaming mode: {tts_config.streaming_mode.value}")
            typer.echo(f"  Output mode: {tts_config.output_mode.value}")
            typer.echo(f"  Toggle hotkey: {tts_config.tts_toggle_key}")
            typer.echo(f"  Generate hotkey: {tts_config.tts_generate_key}")
            typer.echo(f"  Stop hotkey: {tts_config.tts_stop_key}")
            typer.echo(f"  Sample rate: {tts_config.sample_rate}")
            typer.echo(f"  Output file path: {tts_config.output_file_path or 'None'}")
            typer.echo(f"  Auto play: {tts_config.auto_play}")
            typer.echo(f"  Use GPU: {tts_config.use_gpu}")
            typer.echo(f"  Max text length: {tts_config.max_text_length}")
            typer.echo(f"  Chunk text threshold: {tts_config.chunk_text_threshold}")

        except Exception as e:
            display_error(f"Error loading TTS config: {e}")

    def tts_config_set(
        self,
        model_path: str | None = None,
        voice_samples_dir: str | None = None,
        default_voice: str | None = None,
        cfg_scale: float | None = None,
        inference_steps: int | None = None,
        tts_mode: str | None = None,
        streaming_mode: str | None = None,
        output_mode: str | None = None,
        toggle_key: str | None = None,
        generate_key: str | None = None,
        stop_key: str | None = None,
        sample_rate: int | None = None,
        output_file_path: str | None = None,
        auto_play: bool | None = None,
        use_gpu: bool | None = None,
        max_text_length: int | None = None,
        chunk_text_threshold: int | None = None,
    ):
        """Configure TTS settings."""
        try:
            config = self.config_repo.load()
            tts_config = config.tts_config

            updated = False

            if model_path is not None:
                tts_config.model_path = model_path
                updated = True

            if voice_samples_dir is not None:
                tts_config.voice_samples_dir = voice_samples_dir
                updated = True

            if default_voice is not None:
                tts_config.default_voice = default_voice
                updated = True

            if cfg_scale is not None:
                tts_config.cfg_scale = cfg_scale
                updated = True

            if inference_steps is not None:
                tts_config.inference_steps = inference_steps
                updated = True

            if tts_mode is not None:
                try:
                    from voicebridge.domain.models import TTSMode

                    tts_config.tts_mode = TTSMode(tts_mode)
                    updated = True
                except ValueError:
                    display_error(
                        f"Invalid TTS mode: {tts_mode}. Valid values: clipboard, mouse, manual"
                    )
                    return

            if streaming_mode is not None:
                try:
                    tts_config.streaming_mode = TTSStreamingMode(streaming_mode)
                    updated = True
                except ValueError:
                    display_error(
                        f"Invalid streaming mode: {streaming_mode}. Valid values: streaming, non_streaming"
                    )
                    return

            if output_mode is not None:
                try:
                    tts_config.output_mode = TTSOutputMode(output_mode)
                    updated = True
                except ValueError:
                    display_error(
                        f"Invalid output mode: {output_mode}. Valid values: play, save, both"
                    )
                    return

            if toggle_key is not None:
                tts_config.tts_toggle_key = toggle_key
                updated = True

            if generate_key is not None:
                tts_config.tts_generate_key = generate_key
                updated = True

            if stop_key is not None:
                tts_config.tts_stop_key = stop_key
                updated = True

            if sample_rate is not None:
                tts_config.sample_rate = sample_rate
                updated = True

            if output_file_path is not None:
                tts_config.output_file_path = output_file_path
                updated = True

            if auto_play is not None:
                tts_config.auto_play = auto_play
                updated = True

            if use_gpu is not None:
                tts_config.use_gpu = use_gpu
                updated = True

            if max_text_length is not None:
                tts_config.max_text_length = max_text_length
                updated = True

            if chunk_text_threshold is not None:
                tts_config.chunk_text_threshold = chunk_text_threshold
                updated = True

            if updated:
                self.config_repo.save(config)
                display_progress("TTS configuration updated", finished=True)
            else:
                display_info("No configuration changes specified")

        except Exception as e:
            display_error(f"Error updating TTS config: {e}")

    def tts_voices(self):
        """List available TTS voices (alias for tts_list_voices)."""
        self.tts_list_voices()
