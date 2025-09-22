import os
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
        debug: bool = False,
    ):
        """Listen for speech and transcribe it with hotkey control."""
        if not self.transcription_orchestrator:
            display_error("Transcription service not available")
            return

        try:
            config = self._build_config(
                model, language, initial_prompt, temperature, profile,
                paste_stream=paste_stream, copy_stream=copy_stream,
                paste_final=paste_final, copy_final=copy_final, debug=debug
            )

            hotkey = getattr(config, 'key', 'ctrl+f2')
            typer.echo(f"🎤 Press {hotkey.upper()} to start/stop recording, or Ctrl+C to quit")

            # State management
            audio_data = b""
            recording = False
            recording_thread = None
            stop_recording = threading.Event()

            def record_audio():
                nonlocal audio_data, recording
                audio_data = b""
                try:
                    chunk_count = 0
                    for chunk in self.transcription_orchestrator.audio_recorder.record_stream():
                        if stop_recording.is_set():
                            break
                        audio_data += chunk
                        chunk_count += 1

                        if chunk_count % 20 == 0:
                            level = min(len(chunk) // 1000, 10)
                            bar = "█" * level + "░" * (10 - level)
                            print(f"🎤 Recording... {bar} {len(audio_data):,} bytes")
                except Exception as e:
                    print(f"\nRecording error: {e}")
                finally:
                    recording = False

            def on_hotkey():
                nonlocal recording, recording_thread
                if debug:
                    typer.echo(f"🔥 DEBUG: on_hotkey() called! Recording state: {recording}")

                if not recording:
                    typer.echo("🎤 Starting recording... (Press hotkey again to stop)")
                    recording = True
                    stop_recording.clear()
                    recording_thread = threading.Thread(target=record_audio, daemon=True)
                    recording_thread.start()
                else:
                    typer.echo("\n🛑 Stopping recording...")
                    recording = False
                    stop_recording.set()
                    self._stop_audio_recorder()

                    if recording_thread:
                        recording_thread.join(timeout=2)

                    self._process_transcription(audio_data, config)
                    typer.echo(f"💡 Press {hotkey.upper()} to start recording again")

            self._setup_hotkey_listener(hotkey, on_hotkey, debug)

        except KeyboardInterrupt:
            typer.echo("\n👋 Goodbye!")
        except Exception as e:
            display_error(f"Listen command failed: {e}")

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
        debug: bool = False,
    ):
        """Interactive mode with press-and-hold 'r' to record."""
        if not self.transcription_orchestrator:
            display_error("Transcription service not available")
            return

        try:
            config = self._build_config(
                model, language, initial_prompt, temperature, profile,
                paste_stream=paste_stream, copy_stream=copy_stream,
                paste_final=paste_final, copy_final=copy_final, debug=debug
            )

            typer.echo("🎤 Interactive mode: Hold 'r' to record, release to transcribe, 'q' to quit")

            # State management
            audio_data = b""
            recording = False
            stop_recording = threading.Event()

            def record_audio():
                nonlocal audio_data, recording
                audio_data = b""
                try:
                    chunk_count = 0
                    for chunk in self.transcription_orchestrator.audio_recorder.record_stream():
                        if stop_recording.is_set():
                            break
                        audio_data += chunk
                        chunk_count += 1

                        if chunk_count % 20 == 0:
                            level = min(len(chunk) // 1000, 10)
                            bar = "█" * level + "░" * (10 - level)
                            print(f"\r🎤 Recording... {bar} {len(audio_data):,} bytes", end="", flush=True)
                except Exception as e:
                    print(f"\nRecording error: {e}")
                finally:
                    recording = False

            def check_terminal_input():
                import select
                import sys
                import termios
                import tty

                old_settings = termios.tcgetattr(sys.stdin)
                try:
                    tty.setraw(sys.stdin.fileno())
                    while True:
                        if select.select([sys.stdin], [], [], 0.1) == ([sys.stdin], [], []):
                            char = sys.stdin.read(1)
                            if char.lower() == 'q':
                                break
                            elif char.lower() == 'r' and not recording:
                                # Start recording
                                print("\n🎤 Recording started... (Release 'r' to stop)")
                                recording = True
                                stop_recording.clear()
                                recording_thread = threading.Thread(target=record_audio, daemon=True)
                                recording_thread.start()
                            elif recording:
                                # Stop recording
                                print("\n🛑 Stopping recording...")
                                recording = False
                                stop_recording.set()
                                self._stop_audio_recorder()

                                if recording_thread:
                                    recording_thread.join(timeout=2)

                                self._process_transcription(audio_data, config)
                                print("💡 Hold 'r' to record again, 'q' to quit")
                finally:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

            check_terminal_input()

        except KeyboardInterrupt:
            typer.echo("\n👋 Goodbye!")
        except Exception as e:
            display_error(f"Interactive command failed: {e}")

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
        debug: bool = False,
    ):
        """Global hotkey listener for speech recognition."""
        if not self.transcription_orchestrator:
            display_error("Transcription service not available")
            return

        try:
            config = self._build_config(
                model, language, initial_prompt, temperature, profile,
                paste_stream=paste_stream, copy_stream=copy_stream,
                paste_final=paste_final, copy_final=copy_final, debug=debug
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
                recording_thread = threading.Thread(target=self._record_audio_thread,
                                                 args=(audio_data, stop_recording), daemon=True)
                recording_thread.start()

            def stop_recording_func():
                nonlocal recording, recording_thread
                if not recording:
                    return

                typer.echo("🛑 Stopping recording...")
                recording = False
                stop_recording.set()
                self._stop_audio_recorder()

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

            self._setup_global_hotkey_listener(key, on_hotkey, mode, stop_recording_func, debug)

        except KeyboardInterrupt:
            typer.echo("\n👋 Hotkey listener stopped!")
        except Exception as e:
            display_error(f"Hotkey command failed: {e}")

    def _process_transcription(self, audio_data: bytes, config):
        """Process audio data and handle transcription output."""
        if not audio_data:
            typer.echo("No audio recorded.")
            return

        typer.echo("🔄 Transcribing audio...")
        try:
            result = self.transcription_orchestrator.transcription_service.transcribe(audio_data, config)
            if result and result.text:
                typer.echo(f"📝 Transcription: {result.text}")
                typer.echo(f"🎯 Confidence: {result.confidence:.2f}")
                typer.echo(f"🌍 Language: {result.language}")

                # Handle clipboard operations
                if config.copy_final:
                    success = self.transcription_orchestrator.clipboard_service.copy_text(result.text)
                    if success:
                        typer.echo("📋 Copied to clipboard")
                    else:
                        typer.echo("⚠️ Failed to copy to clipboard")
            else:
                typer.echo("No speech detected or transcription failed.")
        except Exception as e:
            typer.echo(f"Transcription error: {e}")

    def _record_audio_thread(self, audio_data: bytes, stop_event: threading.Event):
        """Audio recording thread function."""
        try:
            chunk_count = 0
            for chunk in self.transcription_orchestrator.audio_recorder.record_stream():
                if stop_event.is_set():
                    break
                audio_data += chunk
                chunk_count += 1

                if chunk_count % 20 == 0:
                    level = min(len(chunk) // 1000, 10)
                    bar = "█" * level + "░" * (10 - level)
                    print(f"\r🎤 Recording... {bar} {len(audio_data):,} bytes", end="", flush=True)
        except Exception as e:
            print(f"\nRecording error: {e}")

    def _setup_hotkey_listener(self, hotkey: str, callback, debug: bool = False):
        """Setup hotkey listener with fallback support."""
        try:
            from pynput import keyboard

            # Check for WSL environment
            is_wsl = self._detect_wsl()
            if is_wsl:
                typer.echo("⚠️ WSL detected. If hotkeys don't work, use interactive mode instead.")

            if debug:
                typer.echo(f"🔧 Setting up hotkey: {hotkey}")

            # Try pynput's built-in hotkey first
            try:
                formatted_hotkey = self._format_hotkey_for_pynput(hotkey)
                listener = keyboard.GlobalHotKeys({formatted_hotkey: callback})
                listener.start()

                typer.echo(f"✅ Hotkey {hotkey.upper()} is active")
                listener.join()
            except Exception as e:
                if debug:
                    typer.echo(f"Built-in hotkey failed: {e}")
                self._setup_manual_hotkey_detection(hotkey, callback, debug)

        except ImportError:
            display_error("pynput library not available. Please install it with: pip install pynput")
        except Exception as e:
            if debug:
                typer.echo(f"Hotkey setup error: {e}")
            display_error(f"Failed to setup hotkey: {e}")

    def _setup_global_hotkey_listener(self, key: str, on_press_callback, mode: str, on_release_callback, debug: bool):
        """Setup global hotkey listener for hotkey command."""
        try:
            from pynput import keyboard

            if debug:
                typer.echo(f"🔧 Setting up global hotkey: {key} (mode: {mode})")

            if mode == "hold":
                # For hold mode, we need both press and release events
                def on_press(pressed_key):
                    if self._key_matches_target(pressed_key, key):
                        on_press_callback()

                def on_release(pressed_key):
                    if self._key_matches_target(pressed_key, key):
                        on_release_callback()

                listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            else:
                # For toggle mode, use GlobalHotKeys
                formatted_key = self._format_single_key_for_pynput(key)
                listener = keyboard.GlobalHotKeys({formatted_key: on_press_callback})

            listener.start()
            typer.echo(f"✅ Global hotkey {key.upper()} is active ({mode} mode)")
            listener.join()

        except ImportError:
            display_error("pynput library not available. Please install it with: pip install pynput")
        except Exception as e:
            display_error(f"Failed to setup global hotkey: {e}")

    def _setup_manual_hotkey_detection(self, hotkey: str, callback, debug: bool):
        """Manual hotkey detection as fallback."""
        try:
            from pynput import keyboard

            required_mods, modifiers_active, key_matches = self._build_manual_hotkey_detector(hotkey)

            current_modifiers = {
                "ctrl": False, "alt": False, "shift": False, "cmd": False, "super": False
            }

            def on_press(key):
                # Update modifier states
                if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
                    current_modifiers["ctrl"] = True
                elif key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
                    current_modifiers["alt"] = True
                elif key == keyboard.Key.shift_l or key == keyboard.Key.shift_r:
                    current_modifiers["shift"] = True
                elif key == keyboard.Key.cmd:
                    current_modifiers["cmd"] = True

                # Check if hotkey matches
                if key_matches(key) and modifiers_active(**current_modifiers):
                    if debug:
                        typer.echo(f"🔥 Manual hotkey detected: {hotkey}")
                    callback()

            def on_release(key):
                # Update modifier states
                if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
                    current_modifiers["ctrl"] = False
                elif key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
                    current_modifiers["alt"] = False
                elif key == keyboard.Key.shift_l or key == keyboard.Key.shift_r:
                    current_modifiers["shift"] = False
                elif key == keyboard.Key.cmd:
                    current_modifiers["cmd"] = False

            listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            listener.start()

            if debug:
                typer.echo(f"✅ Manual hotkey detection active for: {hotkey}")

            listener.join()

        except Exception as e:
            display_error(f"Manual hotkey detection failed: {e}")

    def _detect_wsl(self) -> bool:
        """Detect if running in WSL environment."""
        if os.path.exists("/proc/version"):
            try:
                with open("/proc/version") as f:
                    content = f.read().lower()
                    return "microsoft" in content or "wsl" in content
            except OSError:
                pass
        return False

    def _format_hotkey_for_pynput(self, hotkey: str) -> str:
        """Format hotkey string for pynput."""
        parts = [part.strip() for part in hotkey.split('+') if part.strip()]
        formatted = []
        for part in parts:
            part_lower = part.lower()
            if part_lower in {"ctrl", "alt", "shift", "cmd", "super"}:
                formatted.append(f"<{part_lower}>")
            elif part_lower.startswith('f') and part_lower[1:].isdigit():
                formatted.append(f"<{part_lower}>")
            elif part_lower in {"space", "enter", "return", "tab", "esc", "escape", "backspace"}:
                formatted.append(f"<{part_lower}>")
            else:
                formatted.append(part_lower)
        return "+".join(formatted)

    def _format_single_key_for_pynput(self, key: str) -> str:
        """Format single key for pynput."""
        key_lower = key.lower()
        if key_lower.startswith('f') and key_lower[1:].isdigit():
            return f"<{key_lower}>"
        elif key_lower in {"space", "enter", "return", "tab", "esc", "escape", "backspace"}:
            return f"<{key_lower}>"
        else:
            return key_lower

    def _key_matches_target(self, pressed_key, target_key: str) -> bool:
        """Check if pressed key matches target key."""
        try:
            from pynput import keyboard

            target_lower = target_key.lower()

            # Function keys
            if target_lower.startswith('f') and target_lower[1:].isdigit():
                target_attr = getattr(keyboard.Key, target_lower, None)
                return pressed_key == target_attr

            # Special keys
            special_keys = {
                "space": keyboard.Key.space,
                "enter": keyboard.Key.enter,
                "return": keyboard.Key.enter,
                "tab": keyboard.Key.tab,
                "esc": keyboard.Key.esc,
                "escape": keyboard.Key.esc,
                "backspace": keyboard.Key.backspace,
            }

            if target_lower in special_keys:
                return pressed_key == special_keys[target_lower]

            # Character keys
            if hasattr(pressed_key, "char") and pressed_key.char:
                return pressed_key.char.lower() == target_lower

            return False
        except (AttributeError, TypeError, ImportError):
            return False

    def _build_manual_hotkey_detector(self, hotkey: str):
        """Build manual hotkey detection components."""
        from pynput import keyboard

        parts = [part.strip() for part in hotkey.split('+') if part.strip()]
        modifiers = {"ctrl", "alt", "shift", "cmd", "super"}
        required_mods = {part for part in parts if part in modifiers}
        base_key = None

        for part in reversed(parts):
            if part not in modifiers:
                base_key = part
                break

        if base_key is None:
            base_key = "f2"

        base_key = base_key.lower()

        special_keys = {
            "space": keyboard.Key.space,
            "enter": keyboard.Key.enter,
            "return": keyboard.Key.enter,
            "tab": keyboard.Key.tab,
            "esc": keyboard.Key.esc,
            "escape": keyboard.Key.esc,
            "backspace": keyboard.Key.backspace,
        }

        def modifiers_active(ctrl: bool, alt: bool, shift: bool, cmd: bool, super_key: bool) -> bool:
            checks = []
            if "ctrl" in required_mods:
                checks.append(ctrl)
            if "alt" in required_mods:
                checks.append(alt)
            if "shift" in required_mods:
                checks.append(shift)
            if "cmd" in required_mods:
                checks.append(cmd)
            if "super" in required_mods:
                checks.append(super_key)
            return all(checks) if checks else True

        def key_matches(key: object) -> bool:
            if base_key in special_keys:
                return key == special_keys[base_key]
            if base_key.startswith('f') and base_key[1:].isdigit():
                target = getattr(keyboard.Key, base_key, None)
                return key == target
            if hasattr(key, "char") and key.char:
                return key.char.lower() == base_key
            return False

        return required_mods, modifiers_active, key_matches
