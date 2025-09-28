import threading

import typer

from voicebridge.cli.commands.base import BaseCommands
from voicebridge.cli.utils.command_helpers import (
    display_error,
)


class SpeechCommands(BaseCommands):
    """Commands for real-time speech recognition and interaction."""

    def listen(
        self,
        model: str | None = None,
        language: str | None = None,
        initial_prompt: str | None = None,
        temperature: float = 0.0,
        profile: str | None = None,
        paste_stream: bool = False,
        copy_stream: bool = False,
        paste_final: bool = False,
        copy_final: bool = True,
        max_memory: int = 0,
        debug: bool = False,
    ):
        """Listen for speech and transcribe it."""
        print(
            f"[DEBUG] speech_commands.listen: paste_final={paste_final}, paste_stream={paste_stream}"
        )
        try:
            import time

            config = self._build_config(
                model=model,
                language=language,
                initial_prompt=initial_prompt,
                temperature=temperature,
                profile=profile,
                paste_stream=paste_stream,
                copy_stream=copy_stream,
                paste_final=paste_final,
                copy_final=copy_final,
                max_memory_mb=max_memory,
                debug=debug,
            )

            typer.echo("🎤 Press F2 to start/stop recording, or Ctrl+C to quit")

            # State management
            audio_data = b""
            recording = False
            recording_thread = None
            stop_recording = threading.Event()

            def record_audio():
                nonlocal audio_data
                audio_data = b""
                try:
                    import sys

                    chunk_count = 0
                    for (
                        chunk
                    ) in self.transcription_orchestrator.audio_recorder.record_stream():
                        if stop_recording.is_set():
                            break
                        audio_data += chunk
                        chunk_count += 1

                        # Show continuous progress updates
                        # Calculate audio level based on chunk size (simple volume indicator)
                        level = min(len(chunk) // 500, 10) if chunk else 0
                        bar = "█" * level + "░" * (10 - level)

                        # Clear the line and show updated progress
                        sys.stdout.write(
                            f"\r🎤 Recording... {bar} {len(audio_data):,} bytes"
                        )
                        sys.stdout.flush()
                except Exception as e:
                    typer.echo(f"\nRecording error: {e}")

            # Hotkey listener with debounce
            last_hotkey_time = [0]  # Use list to make it mutable

            def on_hotkey():
                nonlocal recording, recording_thread
                import time

                # Debounce: ignore rapid consecutive presses (within 300ms)
                current_time = time.time()
                if current_time - last_hotkey_time[0] < 0.3:
                    return
                last_hotkey_time[0] = current_time

                if not recording:
                    # Start recording
                    typer.echo("🎤 Starting recording... (Press F2 again to stop)")
                    recording = True
                    stop_recording.clear()
                    recording_thread = threading.Thread(target=record_audio)
                    recording_thread.start()
                else:
                    # Stop recording and transcribe
                    import sys

                    # Clear the progress line first
                    sys.stdout.write("\r" + " " * 80 + "\r")
                    sys.stdout.flush()

                    typer.echo("🛑 Stopping recording...")
                    recording = False
                    stop_recording.set()

                    if recording_thread:
                        recording_thread.join(timeout=2)

                    # Process transcription
                    if audio_data:
                        typer.echo("🔄 Transcribing audio...")
                        try:
                            result = self.transcription_orchestrator.transcription_service.transcribe(
                                audio_data, config
                            )
                            if result and result.text:
                                typer.echo(f"📝 Transcription: {result.text}")
                                typer.echo(f"🎯 Confidence: {result.confidence:.2f}")
                                typer.echo(f"🌍 Language: {result.language}")

                                # Handle clipboard and text insertion
                                if config.copy_final:
                                    success = self.transcription_orchestrator.clipboard_service.copy_text(
                                        result.text
                                    )
                                    if success:
                                        typer.echo("📋 Copied to clipboard")
                                    else:
                                        typer.echo("⚠️ Failed to copy to clipboard")

                                if config.paste_final:
                                    success = self.transcription_orchestrator.clipboard_service.type_text(
                                        result.text
                                    )
                                    if not success:
                                        typer.echo("⚠️ Failed to insert text at cursor")
                            else:
                                typer.echo(
                                    "No speech detected or transcription failed."
                                )
                        except Exception as e:
                            typer.echo(f"Transcription error: {e}")
                    else:
                        typer.echo("No audio recorded.")

                    typer.echo("💡 Press F2 to start recording again")

            # Setup hotkey listener with fallback for terminal issues
            try:
                import signal
                import sys

                from pynput import keyboard

                def on_press(key):
                    try:
                        # Handle F2 key with standard pynput detection
                        if key == keyboard.Key.f2:
                            on_hotkey()
                    except (AttributeError, ValueError):
                        pass

                listener = keyboard.Listener(on_press=on_press, suppress=False)

                # Add signal handler for proper Ctrl+C handling on Windows
                def signal_handler(signum, frame):
                    typer.echo("\n👋 Exiting...")
                    if recording:
                        stop_recording.set()
                        if recording_thread:
                            recording_thread.join(timeout=2)
                    listener.stop()
                    sys.exit(0)

                signal.signal(signal.SIGINT, signal_handler)
                listener.start()

                typer.echo("✅ Hotkey listener active. Press F2 to start recording.")
                typer.echo(
                    "💡 If F2 doesn't work, try pressing 'r' in the terminal or use Ctrl+C"
                )

                # Also listen for F2 and 'r' key in terminal as backup
                def check_terminal_input():
                    import platform
                    import sys
                    import time

                    # Platform-specific terminal input handling
                    if platform.system() == "Windows":
                        try:
                            import msvcrt

                            while True:
                                if msvcrt.kbhit():
                                    ch = msvcrt.getch()
                                    if (
                                        ch == b"\r" or ch == b"r" or ch == b"R"
                                    ):  # Enter or 'r'
                                        on_hotkey()
                                    elif ch == b"\x03":  # Ctrl+C
                                        break
                                    elif (
                                        ch == b"\x00" or ch == b"\xe0"
                                    ):  # Function key prefix
                                        extended = msvcrt.getch()
                                        if extended == b"<":  # F2 key code
                                            typer.echo("F2 detected via terminal!")
                                            on_hotkey()
                                time.sleep(0.05)
                        except ImportError:
                            # msvcrt not available, fallback to simple input
                            while True:
                                try:
                                    line = input()
                                    if line.lower().strip() in ["r", ""]:
                                        on_hotkey()
                                except (EOFError, KeyboardInterrupt):
                                    break
                    else:
                        # Unix/Linux/macOS terminal input handling
                        try:
                            import select
                            import termios
                            import tty

                            old_settings = None
                            try:
                                old_settings = termios.tcgetattr(sys.stdin)
                                tty.setraw(sys.stdin.fileno())
                                escape_sequence = ""
                                while True:
                                    if select.select([sys.stdin], [], [], 0.05) == (
                                        [sys.stdin],
                                        [],
                                        [],
                                    ):
                                        ch = sys.stdin.read(1)

                                        # Handle escape sequences for F2
                                        if ch == "\x1b":  # ESC
                                            escape_sequence = ch
                                            continue
                                        elif escape_sequence:
                                            escape_sequence += ch
                                            # F2 sequences: \x1b[OQ, \x1bOQ, or other variants
                                            if escape_sequence in [
                                                "\x1b[OQ",
                                                "\x1bOQ",
                                                "\x1b[12~",
                                            ]:
                                                typer.echo("F2 detected via terminal!")
                                                on_hotkey()
                                                escape_sequence = ""
                                                continue
                                            elif (
                                                len(escape_sequence) >= 4
                                            ):  # Reset if too long
                                                escape_sequence = ""

                                        # Handle regular keys
                                        if ch.lower() == "r":
                                            on_hotkey()
                                        elif ch == "\x03":  # Ctrl+C
                                            break

                                        # Reset escape sequence on regular key
                                        escape_sequence = ""
                            except (OSError, ValueError, termios.error):
                                pass
                            finally:
                                try:
                                    if old_settings is not None:
                                        termios.tcsetattr(
                                            sys.stdin, termios.TCSADRAIN, old_settings
                                        )
                                except (OSError, ValueError, termios.error):
                                    pass
                        except ImportError:
                            # termios/select not available, fallback to simple input
                            while True:
                                try:
                                    line = input()
                                    if line.lower().strip() in ["r", ""]:
                                        on_hotkey()
                                except (EOFError, KeyboardInterrupt):
                                    break

                # Start terminal input thread as backup
                terminal_thread = threading.Thread(
                    target=check_terminal_input, daemon=True
                )
                terminal_thread.start()

                try:
                    # Keep the main thread alive
                    while True:
                        time.sleep(0.1)
                except KeyboardInterrupt:
                    typer.echo("\n👋 Exiting...")
                    if recording:
                        stop_recording.set()
                        if recording_thread:
                            recording_thread.join(timeout=2)
                    listener.stop()

            except ImportError:
                typer.echo("⚠️ pynput not available - falling back to simple mode")
                typer.echo(
                    "🎤 Recording started... (Press Ctrl+C to stop and transcribe)"
                )

                # Fallback to old behavior
                recording = True
                recording_thread = threading.Thread(target=record_audio, daemon=True)
                recording_thread.start()

                try:
                    while recording:
                        time.sleep(0.1)
                except KeyboardInterrupt:
                    import sys

                    # Clear the progress line first
                    sys.stdout.write("\r" + " " * 80 + "\r")
                    sys.stdout.flush()

                    typer.echo("🛑 Recording stopped. Transcribing...")
                    recording = False
                    stop_recording.set()
                    time.sleep(0.2)

                if audio_data:
                    try:
                        typer.echo("🔄 Transcribing audio...")
                        result = self.transcription_orchestrator.transcription_service.transcribe(
                            audio_data, config
                        )
                        if result and result.text:
                            typer.echo(f"📝 Transcription: {result.text}")
                            typer.echo(f"🎯 Confidence: {result.confidence:.2f}")
                            typer.echo(f"🌍 Language: {result.language}")

                            # Handle clipboard and text insertion
                            if config.copy_final:
                                success = self.transcription_orchestrator.clipboard_service.copy_text(
                                    result.text
                                )
                                if success:
                                    typer.echo("📋 Copied to clipboard")
                                else:
                                    typer.echo("⚠️ Failed to copy to clipboard")

                            if config.paste_final:
                                success = self.transcription_orchestrator.clipboard_service.type_text(
                                    result.text
                                )
                                if not success:
                                    typer.echo("⚠️ Failed to insert text at cursor")
                        else:
                            typer.echo("No speech detected or transcription failed.")
                    except Exception as e:
                        typer.echo(f"Transcription error: {e}")
                else:
                    typer.echo("No audio recorded.")

        except Exception as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1) from None

    def interactive(
        self,
        model: str | None = None,
        language: str | None = None,
        initial_prompt: str | None = None,
        temperature: float = 0.0,
        profile: str | None = None,
        paste_stream: bool = False,
        copy_stream: bool = False,
        paste_final: bool = False,
        copy_final: bool = True,
        max_memory: int = 0,
        debug: bool = False,
    ):
        """Interactive mode for speech transcription with keyboard stopping."""
        try:
            import time

            config = self._build_config(
                model=model,
                language=language,
                initial_prompt=initial_prompt,
                temperature=temperature,
                profile=profile,
                paste_stream=paste_stream,
                copy_stream=copy_stream,
                paste_final=paste_final,
                copy_final=copy_final,
                max_memory_mb=max_memory,
                debug=debug,
            )

            typer.echo("🎯 Interactive Mode Started!")
            typer.echo(
                "💡 Commands: 's' = start recording, 't' = stop recording, 'q' = quit"
            )
            typer.echo()

            # Shared state between threads
            recording_state = {"is_recording": False}
            recording_thread = None
            audio_data = b""
            stop_recording = threading.Event()

            def record_audio():
                nonlocal audio_data
                audio_data = b""
                try:
                    import sys

                    for (
                        chunk
                    ) in self.transcription_orchestrator.audio_recorder.record_stream():
                        if stop_recording.is_set():
                            break
                        audio_data += chunk

                        # Show continuous progress with simple audio level indicator
                        level = min(len(chunk) // 500, 10) if chunk else 0
                        bar = "█" * level + "░" * (10 - level)

                        # Clear the line and show updated progress
                        sys.stdout.write(
                            f"\r🎤 Recording... {bar} {len(audio_data):,} bytes"
                        )
                        sys.stdout.flush()
                except Exception as e:
                    typer.echo(f"\nRecording error: {e}")

            should_exit = threading.Event()

            # Use same reliable terminal input handling as listen mode
            def check_terminal_input():
                import platform
                import sys
                import time

                nonlocal recording_thread

                def handle_key_command(key_char):
                    """Handle key commands - extracted for reuse between platforms"""
                    nonlocal recording_thread
                    if key_char.lower() == "s":
                        if not recording_state["is_recording"]:
                            # Start recording
                            typer.echo(
                                "🎤 Starting recording... Press 's' again to stop"
                            )
                            recording_state["is_recording"] = True
                            stop_recording.clear()
                            recording_thread = threading.Thread(target=record_audio)
                            recording_thread.start()
                        else:
                            # Stop recording and process
                            # Clear the progress line first
                            sys.stdout.write("\r" + " " * 80 + "\r")
                            sys.stdout.flush()

                            typer.echo("🛑 Stopping recording...")
                            stop_recording.set()
                            recording_state["is_recording"] = False

                            if recording_thread:
                                recording_thread.join(timeout=2)

                            # Process transcription - duplicate logic here for simplicity
                            if audio_data:
                                try:
                                    typer.echo("🔄 Processing audio...")
                                    result = self.transcription_orchestrator.transcription_service.transcribe(
                                        audio_data, config
                                    )
                                    if result and result.text:
                                        typer.echo(f"📝 Transcription: {result.text}")
                                        typer.echo(
                                            f"🎯 Confidence: {result.confidence:.2f}"
                                        )
                                        typer.echo(f"🌍 Language: {result.language}")

                                        # Handle clipboard and text insertion
                                        if config.copy_final:
                                            success = self.transcription_orchestrator.clipboard_service.copy_text(
                                                result.text
                                            )
                                            if success:
                                                typer.echo("📋 Copied to clipboard")
                                            else:
                                                typer.echo(
                                                    "⚠️ Failed to copy to clipboard"
                                                )

                                        if config.paste_final:
                                            success = self.transcription_orchestrator.clipboard_service.type_text(
                                                result.text
                                            )
                                            if not success:
                                                typer.echo(
                                                    "⚠️ Failed to insert text at cursor"
                                                )
                                    else:
                                        typer.echo(
                                            "No speech detected or transcription failed."
                                        )
                                except Exception as e:
                                    typer.echo(f"Transcription error: {e}")
                            else:
                                typer.echo("No audio recorded.")

                            typer.echo(
                                "💡 Press 's' to start recording again, 'q' to quit"
                            )

                    elif key_char.lower() == "t" and recording_state["is_recording"]:
                        # Stop recording only
                        # Clear the progress line first
                        sys.stdout.write("\r" + " " * 80 + "\r")
                        sys.stdout.flush()

                        typer.echo("🛑 Stopping recording...")
                        stop_recording.set()
                        recording_state["is_recording"] = False

                        if recording_thread:
                            recording_thread.join(timeout=2)

                        # Process transcription
                        if audio_data:
                            try:
                                typer.echo("🔄 Processing audio...")
                                result = self.transcription_orchestrator.transcription_service.transcribe(
                                    audio_data, config
                                )
                                if result and result.text:
                                    typer.echo(f"📝 Transcription: {result.text}")
                                    typer.echo(
                                        f"🎯 Confidence: {result.confidence:.2f}"
                                    )
                                    typer.echo(f"🌍 Language: {result.language}")

                                    # Handle clipboard and text insertion
                                    if config.copy_final:
                                        success = self.transcription_orchestrator.clipboard_service.copy_text(
                                            result.text
                                        )
                                        if success:
                                            typer.echo("📋 Copied to clipboard")
                                        else:
                                            typer.echo("⚠️ Failed to copy to clipboard")

                                    if config.paste_final:
                                        success = self.transcription_orchestrator.clipboard_service.type_text(
                                            result.text
                                        )
                                        if not success:
                                            typer.echo(
                                                "⚠️ Failed to insert text at cursor"
                                            )
                                else:
                                    typer.echo(
                                        "No speech detected or transcription failed."
                                    )
                            except Exception as e:
                                typer.echo(f"Transcription error: {e}")
                        else:
                            typer.echo("No audio recorded.")

                        typer.echo("💡 Press 's' to start recording again, 'q' to quit")

                    elif key_char.lower() == "q":
                        should_exit.set()
                        return True  # Exit signal

                    return False  # Continue signal

                # Platform-specific terminal input handling
                if platform.system() == "Windows":
                    try:
                        import msvcrt

                        while not should_exit.is_set():
                            if msvcrt.kbhit():
                                ch = msvcrt.getch()
                                if ch == b"\x03":  # Ctrl+C
                                    should_exit.set()
                                    break
                                try:
                                    key_char = ch.decode("utf-8")
                                    if handle_key_command(key_char):
                                        break
                                except UnicodeDecodeError:
                                    continue
                            time.sleep(0.05)
                    except ImportError:
                        # msvcrt not available, fallback to simple input
                        while not should_exit.is_set():
                            try:
                                line = input().strip()
                                if line:
                                    if handle_key_command(line[0]):
                                        break
                            except (EOFError, KeyboardInterrupt):
                                should_exit.set()
                                break
                else:
                    # Unix/Linux/macOS terminal input handling
                    try:
                        import select
                        import termios
                        import tty

                        old_settings = None
                        try:
                            old_settings = termios.tcgetattr(sys.stdin)
                            tty.setraw(sys.stdin.fileno())

                            while not should_exit.is_set():
                                if select.select([sys.stdin], [], [], 0.05) == (
                                    [sys.stdin],
                                    [],
                                    [],
                                ):
                                    ch = sys.stdin.read(1)

                                    if ch == "\x03":  # Ctrl+C
                                        should_exit.set()
                                        break

                                    if handle_key_command(ch):
                                        break
                        except (OSError, ValueError, termios.error):
                            pass
                        finally:
                            try:
                                if old_settings is not None:
                                    termios.tcsetattr(
                                        sys.stdin, termios.TCSADRAIN, old_settings
                                    )
                            except (OSError, ValueError, termios.error):
                                pass
                    except ImportError:
                        # termios/select not available, fallback to simple input
                        while not should_exit.is_set():
                            try:
                                line = input().strip()
                                if line:
                                    if handle_key_command(line[0]):
                                        break
                            except (EOFError, KeyboardInterrupt):
                                should_exit.set()
                                break

            # Start terminal input handler
            terminal_thread = threading.Thread(target=check_terminal_input, daemon=True)
            terminal_thread.start()

            typer.echo(
                "💡 Press 's' to start/stop recording, 't' to stop only, 'q' to quit"
            )

            try:
                # Keep main thread alive
                while not should_exit.is_set():
                    time.sleep(0.1)
            except KeyboardInterrupt:
                should_exit.set()
                typer.echo("\n👋 Stopped by Ctrl+C")

            # Final cleanup
            if recording_state["is_recording"]:
                stop_recording.set()
                if recording_thread:
                    recording_thread.join(timeout=2)
            should_exit.set()
            typer.echo("👋 Goodbye!")

        except Exception as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1) from None

    def hotkey(
        self,
        key: str = "f9",
        mode: str = "toggle",
        model: str | None = None,
        language: str | None = None,
        initial_prompt: str | None = None,
        temperature: float = 0.0,
        profile: str | None = None,
        paste_stream: bool = False,
        copy_stream: bool = False,
        paste_final: bool = False,
        copy_final: bool = True,
        max_memory: int = 0,
        debug: bool = False,
    ):
        """Global hotkey listener for speech recognition."""
        if not self.transcription_orchestrator:
            display_error("Transcription service not available")
            return

        try:
            config = self._build_config(
                model=model,
                language=language,
                initial_prompt=initial_prompt,
                temperature=temperature,
                profile=profile,
                paste_stream=paste_stream,
                copy_stream=copy_stream,
                paste_final=paste_final,
                copy_final=copy_final,
                max_memory_mb=max_memory,
                debug=debug,
            )

            typer.echo(f"🎯 Global hotkey mode: {key.upper()} ({mode})")
            typer.echo("Press Ctrl+C to stop the hotkey listener")

            # State management
            audio_data = b""
            recording = False
            recording_thread = None
            stop_recording = threading.Event()

            def start_recording():
                nonlocal recording, recording_thread, audio_data
                if recording:
                    return

                typer.echo("🎤 Recording started...")
                recording = True
                audio_data = b""
                stop_recording.clear()
                recording_thread = threading.Thread(
                    target=self._record_audio_thread,
                    args=(audio_data, stop_recording),
                    daemon=True,
                )
                recording_thread.start()

            def stop_recording_func():
                nonlocal recording, recording_thread
                if not recording:
                    return

                typer.echo("🛑 Stopping recording...")
                recording = False
                stop_recording.set()

                if recording_thread:
                    recording_thread.join(timeout=2)

                self._process_transcription(audio_data, config)

            def toggle_recording():
                if recording:
                    stop_recording_func()
                else:
                    start_recording()

            def on_hotkey():
                if mode == "toggle":
                    toggle_recording()
                elif mode == "hold":
                    start_recording()

            self._setup_global_hotkey_listener(
                key, on_hotkey, mode, stop_recording_func, debug
            )

        except KeyboardInterrupt:
            typer.echo("\n👋 Hotkey listener stopped!")
        except Exception as e:
            display_error(f"Hotkey command failed: {e}")
