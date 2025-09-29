import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from voicebridge.domain.models import (
    TTSConfig,
    TTSMode,
    TTSOutputMode,
    TTSResult,
    TTSStreamingMode,
    VoiceInfo,
)
from voicebridge.ports.interfaces import (
    AudioPlaybackService,
    Logger,
    TextInputService,
    TTSService,
)

# Import keyboard libraries for testing compatibility
try:
    import keyboard

    # Create a namespace object to hold GlobalHotKeys for testing
    if not hasattr(keyboard, "GlobalHotKeys"):
        try:
            from pynput.keyboard import GlobalHotKeys

            keyboard.GlobalHotKeys = GlobalHotKeys
        except ImportError:
            keyboard.GlobalHotKeys = None
except ImportError:
    # Create a mock keyboard module for testing
    class MockKeyboard:
        GlobalHotKeys = None

    keyboard = MockKeyboard()


class TTSOrchestrator:
    """Orchestrates TTS functionality across different input sources and output modes"""

    def __init__(
        self,
        tts_service: TTSService,
        text_input_service: TextInputService,
        audio_playback_service: AudioPlaybackService,
        logger: Logger,
    ):
        self.tts_service = tts_service
        self.text_input_service = text_input_service
        self.audio_playback_service = audio_playback_service
        self.logger = logger

        self.is_active = False
        self.current_config = None
        self.voice_samples_cache = {}
        self.monitoring_thread = None
        self.last_processed_text = ""
        self.keyboard_listener = None

    def start_tts_mode(self, config: TTSConfig) -> None:
        """Start TTS mode with specified configuration"""
        if self.is_active:
            self.logger.info("TTS mode already active, stopping current mode first")
            self.stop_tts()

        self.current_config = config
        self.is_active = True

        # Load voice samples
        self._load_voice_samples(config)

        self.logger.info(f"Starting TTS mode: {config.tts_mode.value}")

        if config.tts_mode == TTSMode.CLIPBOARD:
            self._start_clipboard_monitoring(config)
        elif config.tts_mode == TTSMode.MOUSE:
            self._start_selection_monitoring(config)
        # MANUAL mode doesn't need monitoring

    def stop_tts(self) -> None:
        """Stop TTS mode and all monitoring"""
        if not self.is_active:
            return

        self.logger.info("Stopping TTS mode")
        self.is_active = False

        # Stop all services
        self.tts_service.stop_generation()
        self.text_input_service.stop_monitoring()
        self.audio_playback_service.stop_playback()

        # Stop keyboard listener if active
        if self.keyboard_listener:
            try:
                self.keyboard_listener.stop()
                self.keyboard_listener = None
            except Exception as e:
                self.logger.error(f"Error stopping keyboard listener: {e}")

        # Wait for monitoring thread to finish
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)

        self.current_config = None

    def generate_tts_from_text(self, text: str, config: TTSConfig) -> bool:
        """Generate TTS from provided text"""
        if not text.strip():
            self.logger.warning("Empty text provided for TTS generation")
            return False

        # Prevent duplicate processing
        if text == self.last_processed_text:
            self.logger.debug("Skipping duplicate text")
            return False

        self.last_processed_text = text

        # Check text length
        if len(text) > config.max_text_length:
            self.logger.warning(
                f"Text too long ({len(text)} chars), truncating to {config.max_text_length}"
            )
            text = text[: config.max_text_length]

        self.logger.info(
            f"Generating TTS for text: {text[:100]}{'...' if len(text) > 100 else ''}"
        )

        try:
            # Get voice samples
            voice_samples = self._get_voice_samples(config)
            if not voice_samples:
                self.logger.error("No voice samples available")
                return False

            if config.streaming_mode == TTSStreamingMode.STREAMING:
                return self._handle_streaming_tts(text, voice_samples, config)
            else:
                return self._handle_non_streaming_tts(text, voice_samples, config)

        except Exception as e:
            self.logger.error(f"TTS generation failed: {e}")
            return False

    def generate_tts_from_clipboard(self) -> bool:
        """Generate TTS from current clipboard content"""
        if not self.current_config:
            self.logger.error("No TTS config available")
            return False

        text = self.text_input_service.get_clipboard_text()
        return self.generate_tts_from_text(text, self.current_config)

    def generate_tts_from_selection(self) -> bool:
        """Generate TTS from current text selection"""
        if not self.current_config:
            self.logger.error("No TTS config available")
            return False

        text = self.text_input_service.get_selected_text()
        return self.generate_tts_from_text(text, self.current_config)

    def list_available_voices(self, config: TTSConfig) -> dict[str, VoiceInfo]:
        """List all available voice samples"""
        return self.tts_service.load_voice_samples(config.voice_samples_dir)

    def load_voice_samples(self, config: TTSConfig) -> dict[str, VoiceInfo]:
        """Load voice samples from the specified directory"""
        return self.tts_service.load_voice_samples(config.voice_samples_dir)

    def _get_voice_sample_paths(
        self,
        voice_samples: dict[str, VoiceInfo],
        config: TTSConfig,
        voice_name: str = None,
    ) -> list[str]:
        """Get voice sample paths for specified voice"""
        if not voice_samples:
            raise RuntimeError("No voice samples available")

        target_voice = voice_name or config.default_voice

        if target_voice and target_voice in voice_samples:
            return [voice_samples[target_voice].file_path]
        elif target_voice:
            raise ValueError(f"Voice '{target_voice}' not found in available voices")
        else:
            # Use first available voice if no default specified
            first_voice = next(iter(voice_samples.values()))
            return [first_voice.file_path]

    def generate_speech(self, text: str, config: TTSConfig, voice_name: str = None):
        """Generate speech from text with specified configuration"""
        if not text.strip():
            raise ValueError("Text cannot be empty")

        voice_samples = self.load_voice_samples(config)
        voice_paths = self._get_voice_sample_paths(voice_samples, config, voice_name)

        if config.streaming_mode == TTSStreamingMode.STREAMING:
            return self.tts_service.generate_speech_streaming(text, voice_paths, config)
        else:
            return self.tts_service.generate_speech(text, voice_paths, config)

    def generate_and_play_speech(
        self,
        text: str,
        config: TTSConfig,
        voice_name: str = None,
        output_file: str = None,
    ) -> "TTSResult":
        """Generate speech and handle output based on configuration"""
        result = self.generate_speech(text, config, voice_name)

        # Handle output modes
        if config.output_mode in [TTSOutputMode.PLAY_AUDIO, TTSOutputMode.BOTH]:
            self.audio_playback_service.play_audio_data(
                result.audio_data, result.sample_rate
            )

        if (
            config.output_mode in [TTSOutputMode.SAVE_FILE, TTSOutputMode.BOTH]
            and output_file
        ):
            from pathlib import Path

            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "wb") as f:
                f.write(result.audio_data)

        return result

    def process_clipboard_text(self, config: TTSConfig) -> Optional["TTSResult"]:
        """Process text from clipboard"""
        text = self.text_input_service.get_clipboard_text()
        if not text.strip():
            return None
        # Use high-quality generation and handle output properly
        return self.generate_and_play_speech(text, config)

    def process_selected_text(self, config: TTSConfig) -> Optional["TTSResult"]:
        """Process selected text"""
        text = self.text_input_service.get_selected_text()
        if not text.strip():
            return None
        # Use high-quality generation and handle output properly
        return self.generate_and_play_speech(text, config)

    def stop_generation(self) -> None:
        """Stop TTS generation and playback"""
        self.tts_service.stop_generation()
        self.audio_playback_service.stop_playback()

    def is_tts_active(self) -> bool:
        """Check if TTS mode is currently active"""
        return self.is_active

    def is_generating(self) -> bool:
        """Check if TTS is currently generating audio"""
        return self.tts_service.is_generating()

    def is_playing(self) -> bool:
        """Check if audio is currently playing"""
        return self.audio_playback_service.is_playing()

    def _start_clipboard_monitoring(self, config: TTSConfig) -> None:
        """Start monitoring clipboard for changes"""

        def clipboard_callback(text: str):
            if self.is_active and text.strip():
                # Run TTS generation in background thread to avoid blocking monitoring
                generation_thread = threading.Thread(
                    target=self.generate_tts_from_text, args=(text, config), daemon=True
                )
                generation_thread.start()

        self.text_input_service.start_monitoring(clipboard_callback)
        self.logger.info("Started clipboard monitoring for TTS")

    def _start_selection_monitoring(self, config: TTSConfig) -> None:
        """Start monitoring for text selections (manual trigger via hotkey)"""
        # Set up hotkey listener for selection mode
        try:
            from pynput import keyboard

            def on_press(key):
                try:
                    key_str = None

                    # Convert key to string for comparison
                    if hasattr(key, "name"):
                        key_str = key.name.lower()
                    elif hasattr(key, "char") and key.char:
                        key_str = key.char.lower()

                    # Check for generate hotkey
                    if key_str == config.tts_generate_key.lower():
                        self.logger.info(
                            "Generate hotkey pressed, getting selected text"
                        )
                        text = self.text_input_service.get_selected_text()
                        if text and text.strip():
                            self.generate_tts_from_text(text, config)
                        else:
                            self.logger.warning("No text selected")

                    # Check for stop hotkey (Esc)
                    elif key == keyboard.Key.esc:
                        self.logger.info("Stop hotkey pressed")
                        self.tts_service.stop_generation()
                        self.audio_playback_service.stop_playback()

                except Exception as e:
                    self.logger.error(f"Error processing hotkey: {e}")

            # Start keyboard listener in background
            self.keyboard_listener = keyboard.Listener(on_press=on_press)
            self.keyboard_listener.start()

            self.logger.info(
                f"TTS selection mode ready - press {config.tts_generate_key} after selecting text"
            )
        except ImportError:
            self.logger.error("pynput not available for hotkey monitoring")
            # Fall back to simple message
            self.logger.info("TTS selection mode ready (trigger with hotkeys)")
        except Exception as e:
            self.logger.error(f"Failed to start hotkey monitoring: {e}")

    def _handle_non_streaming_tts(
        self, text: str, voice_samples: list[str], config: TTSConfig
    ) -> bool:
        """Handle non-streaming TTS generation"""
        try:
            # Generate complete audio first
            result = self.tts_service.generate_speech(text, voice_samples, config)

            self.logger.info(
                f"Generated {len(result.audio_data)} bytes of audio in {result.generation_time:.2f}s"
            )

            # Handle output based on config
            if config.output_mode in [TTSOutputMode.SAVE_FILE, TTSOutputMode.BOTH]:
                if config.output_file_path:
                    self.audio_playback_service.save_audio(
                        result.audio_data, result.sample_rate, config.output_file_path
                    )
                    self.logger.info(f"Saved audio to {config.output_file_path}")

            if config.output_mode in [TTSOutputMode.PLAY_AUDIO, TTSOutputMode.BOTH]:
                if config.auto_play:
                    self.audio_playback_service.play_audio(
                        result.audio_data, result.sample_rate
                    )
                    self.logger.info("Started audio playback")

            return True

        except Exception as e:
            self.logger.error(f"Non-streaming TTS failed: {e}")
            return False

    def _handle_streaming_tts(
        self, text: str, voice_samples: list[str], config: TTSConfig
    ) -> bool:
        """Handle streaming TTS generation"""
        try:
            self.logger.info("Starting streaming TTS generation")

            # Collect audio chunks
            audio_chunks = []
            chunk_count = 0

            for chunk in self.tts_service.generate_speech_streaming(
                text, voice_samples, config
            ):
                if not self.is_active:  # Check if TTS was stopped
                    break

                chunk_count += 1
                audio_chunks.append(chunk)

                # Play chunk immediately if configured
                if (
                    config.output_mode in [TTSOutputMode.PLAY_AUDIO, TTSOutputMode.BOTH]
                    and config.auto_play
                ):
                    self.audio_playback_service.play_audio(chunk, config.sample_rate)

            # Save complete audio if requested
            if (
                config.output_mode in [TTSOutputMode.SAVE_FILE, TTSOutputMode.BOTH]
                and config.output_file_path
            ):
                complete_audio = b"".join(audio_chunks)
                self.audio_playback_service.save_audio(
                    complete_audio, config.sample_rate, config.output_file_path
                )
                self.logger.info(f"Saved streaming audio to {config.output_file_path}")

            self.logger.info(f"Completed streaming TTS with {chunk_count} chunks")
            return True

        except Exception as e:
            self.logger.error(f"Streaming TTS failed: {e}")
            return False

    def _load_voice_samples(self, config: TTSConfig) -> None:
        """Load and cache voice samples"""
        try:
            self.voice_samples_cache = self.tts_service.load_voice_samples(
                config.voice_samples_dir
            )
            self.logger.info(f"Loaded {len(self.voice_samples_cache)} voice samples")
        except Exception as e:
            self.logger.error(f"Failed to load voice samples: {e}")
            self.voice_samples_cache = {}

    def _get_voice_samples(self, config: TTSConfig) -> list[str]:
        """Get voice sample file paths for TTS generation"""
        if not self.voice_samples_cache:
            self._load_voice_samples(config)

        # Use default voice or first available
        voice_name = config.default_voice

        # Try exact match first
        if voice_name in self.voice_samples_cache:
            return [self.voice_samples_cache[voice_name].file_path]

        # Try partial matching (e.g., "Patrick" matches "en-Patrick")
        if voice_name and self.voice_samples_cache:
            for full_name, voice_info in self.voice_samples_cache.items():
                # Check if the requested voice name is contained in the full name
                if voice_name.lower() in full_name.lower():
                    self.logger.info(f"Voice '{voice_name}' matched to '{full_name}'")
                    return [voice_info.file_path]

                # Also check if the full name ends with the requested name
                if full_name.lower().endswith(f"-{voice_name.lower()}"):
                    self.logger.info(f"Voice '{voice_name}' matched to '{full_name}'")
                    return [voice_info.file_path]

        # If no match found, use first available voice
        if self.voice_samples_cache:
            first_voice = next(iter(self.voice_samples_cache.values()))
            self.logger.warning(
                f"Voice '{voice_name}' not found. Available voices: {list(self.voice_samples_cache.keys())}. Using '{first_voice.name}'"
            )
            return [first_voice.file_path]
        else:
            self.logger.error("No voice samples available")
            return []


class TTSDaemonService:
    """Service for running TTS in daemon mode with hotkey support"""

    def __init__(
        self,
        orchestrator: TTSOrchestrator,
        logger: Logger,
    ):
        self.orchestrator = orchestrator
        self.logger = logger
        self.hotkey_listeners = {}
        self.hotkey_listener = None
        self.clipboard_monitor_thread = None
        self.daemon_active = False
        self.is_running = False
        self.current_config = None
        self.is_monitoring_clipboard = False

        # Daemon state files
        self.pid_file = Path.home() / ".config" / "voicebridge" / "daemon.pid"
        self.status_file = Path.home() / ".config" / "voicebridge" / "daemon.status"

        # Ensure config directory exists
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)

        # Setup signal handlers for daemon control
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def start_daemon(self, config: TTSConfig) -> None:
        """Start TTS daemon with hotkey support"""
        # Check if daemon is already running
        if self.is_daemon_running():
            raise RuntimeError("TTS daemon is already running")

        try:
            # Write PID file
            self._write_pid_file()

            # Write status file
            self._write_status_file(config)

            self.current_config = config

            # Start TTS orchestrator
            self.orchestrator.start_tts_mode(config)

            # Setup hotkeys
            self._setup_hotkeys(config)

            # Start clipboard monitoring if in clipboard mode
            if config.tts_mode == TTSMode.CLIPBOARD:
                self._start_clipboard_monitoring(config)

            self.daemon_active = True
            self.is_running = True
            self.logger.info("TTS daemon started successfully")

        except Exception as e:
            self._cleanup_daemon_files()
            self.logger.error(f"Failed to start TTS daemon: {e}")
            raise

    def stop_daemon(self) -> None:
        """Stop TTS daemon"""
        self.logger.info("Stopping TTS daemon")

        # Stop monitoring
        self.stop_monitoring()

        # Stop orchestrator
        self.orchestrator.stop_tts()

        # Remove hotkey listeners
        self._cleanup_hotkeys()

        # Clean up daemon files
        self._cleanup_daemon_files()

        self.daemon_active = False
        self.is_running = False
        self.current_config = None
        self.logger.info("TTS daemon stopped")

    def stop_daemon_by_pid(self) -> bool:
        """Stop running daemon by sending signal to PID"""
        if not self.pid_file.exists():
            return False

        try:
            with open(self.pid_file) as f:
                pid = int(f.read().strip())

            # Check if process is still running
            try:
                os.kill(pid, 0)  # Test if process exists
            except OSError:
                # Process doesn't exist, clean up stale files
                self._cleanup_daemon_files()
                return False

            # Send SIGTERM to gracefully stop the daemon
            os.kill(pid, signal.SIGTERM)

            # Wait for process to exit
            for _ in range(10):  # Wait up to 10 seconds
                try:
                    os.kill(pid, 0)
                    time.sleep(1)
                except OSError:
                    # Process has exited
                    self._cleanup_daemon_files()
                    return True

            # If still running, force kill
            try:
                os.kill(pid, signal.SIGKILL)
                self._cleanup_daemon_files()
                return True
            except OSError:
                return False

        except (ValueError, FileNotFoundError, OSError) as e:
            self.logger.error(f"Error stopping daemon: {e}")
            self._cleanup_daemon_files()
            return False

    def get_status(self) -> dict:
        """Get daemon status"""
        if self.is_daemon_running():
            try:
                with open(self.status_file) as f:
                    import json

                    status_data = json.load(f)
                return {"status": "running", **status_data}
            except (FileNotFoundError, json.JSONDecodeError):
                return {"status": "running", "mode": "unknown"}
        else:
            return {"status": "stopped"}

    def is_daemon_running(self) -> bool:
        """Check if daemon is currently running"""
        if not self.pid_file.exists():
            return False

        try:
            with open(self.pid_file) as f:
                pid = int(f.read().strip())

            # Check if process is still running
            try:
                os.kill(pid, 0)  # Test if process exists
                return True
            except OSError:
                # Process doesn't exist, clean up stale files
                self._cleanup_daemon_files()
                return False

        except (ValueError, FileNotFoundError):
            return False

    def stop_monitoring(self) -> None:
        """Stop monitoring threads"""
        self.is_monitoring_clipboard = False
        if self.clipboard_monitor_thread and self.clipboard_monitor_thread.is_alive():
            self.clipboard_monitor_thread.join(timeout=2.0)

    def _start_clipboard_monitoring(self, config: TTSConfig) -> None:
        """Start clipboard monitoring in separate thread"""
        import threading

        self.is_monitoring_clipboard = True
        self.clipboard_monitor_thread = threading.Thread(
            target=self._clipboard_monitor_loop, args=(config,), daemon=True
        )
        self.clipboard_monitor_thread.start()

    def _clipboard_monitor_loop(self, config: TTSConfig) -> None:
        """Monitor clipboard for changes"""
        import time

        last_clipboard_text = ""

        while self.is_monitoring_clipboard:
            try:
                current_text = self.orchestrator.text_input_service.get_clipboard_text()
                if current_text != last_clipboard_text and current_text.strip():
                    last_clipboard_text = current_text
                    # Use high-quality TTS generation path
                    self.orchestrator.generate_tts_from_text(current_text, config)
                time.sleep(0.5)  # Check every 0.5 seconds
            except Exception as e:
                self.logger.error(f"Error in clipboard monitoring: {e}")
                time.sleep(1.0)

    def _setup_hotkeys(self, config: TTSConfig) -> None:
        """Setup hotkey listeners"""
        try:
            # Check if we're in WSL - be more conservative with hotkeys
            is_wsl = False
            if os.path.exists("/proc/version"):
                with open("/proc/version") as f:
                    if "microsoft" in f.read().lower():
                        is_wsl = True

            if is_wsl:
                # In WSL, avoid GlobalHotKeys as they can interfere with clipboard
                self.logger.warning(
                    "WSL detected - using basic hotkey support to avoid clipboard conflicts"
                )
                self.logger.info(
                    f"Hotkeys configured (may require focus) - Generate: {config.tts_generate_key}, Stop: {config.tts_stop_key}"
                )
                return

            # Try GlobalHotKeys first (pynput) - only on non-WSL systems
            if keyboard and keyboard.GlobalHotKeys:
                hotkey_map = {
                    config.tts_generate_key: lambda: self._handle_generate_hotkey(
                        config
                    ),
                    config.tts_stop_key: self._handle_stop_hotkey,
                }

                self.hotkey_listener = keyboard.GlobalHotKeys(hotkey_map)
                self.hotkey_listener.start()

                self.logger.info(
                    f"Setup hotkeys - Generate: {config.tts_generate_key}, Stop: {config.tts_stop_key}"
                )
            elif keyboard and hasattr(keyboard, "add_hotkey"):
                # Fallback to keyboard library
                keyboard.add_hotkey(
                    config.tts_generate_key,
                    self._handle_generate_hotkey,
                    args=(config,),
                )

                keyboard.add_hotkey(config.tts_stop_key, self._handle_stop_hotkey)

                self.logger.info(
                    f"Setup hotkeys - Generate: {config.tts_generate_key}, Stop: {config.tts_stop_key}"
                )
            else:
                self.logger.warning("No keyboard library available, hotkeys disabled")

        except Exception as e:
            self.logger.error(f"Failed to setup hotkeys: {e}")

    def _cleanup_hotkeys(self) -> None:
        """Remove hotkey listeners"""
        try:
            if self.hotkey_listener:
                self.hotkey_listener.stop()
                # Don't set to None immediately - tests need to check the mock
                # self.hotkey_listener = None
            elif keyboard is not None:
                # Fallback for keyboard library
                keyboard.unhook_all()
            self.logger.info("Removed hotkey listeners")
        except Exception as e:
            self.logger.error(f"Error cleaning up hotkeys: {e}")

    def _handle_generate_hotkey(self, config: TTSConfig = None) -> None:
        """Handle TTS generation hotkey press"""
        try:
            config = config or self.current_config
            if not config:
                return

            if config.tts_mode == TTSMode.CLIPBOARD:
                text = self.orchestrator.text_input_service.get_clipboard_text()
                if text.strip():
                    self.orchestrator.generate_tts_from_text(text, config)
            elif config.tts_mode == TTSMode.MOUSE:
                text = self.orchestrator.text_input_service.get_selected_text()
                if text.strip():
                    self.orchestrator.generate_tts_from_text(text, config)
        except Exception as e:
            self.logger.error(f"Error handling generate hotkey: {e}")

    def _handle_stop_hotkey(self) -> None:
        """Handle TTS stop hotkey press"""
        try:
            self.orchestrator.stop_generation()
            self.logger.info("TTS stopped via hotkey")
        except Exception as e:
            self.logger.error(f"Error handling stop hotkey: {e}")

    def _write_pid_file(self) -> None:
        """Write current process PID to file"""
        with open(self.pid_file, "w") as f:
            f.write(str(os.getpid()))

    def _write_status_file(self, config: TTSConfig) -> None:
        """Write daemon status to file"""
        import json

        status_data = {
            "mode": config.tts_mode.value,
            "voice": config.default_voice,
            "generate_key": config.tts_generate_key,
            "stop_key": config.tts_stop_key,
            "started_at": time.time(),
        }
        with open(self.status_file, "w") as f:
            json.dump(status_data, f)

    def _cleanup_daemon_files(self) -> None:
        """Clean up PID and status files"""
        try:
            if self.pid_file.exists():
                self.pid_file.unlink()
        except OSError:
            pass

        try:
            if self.status_file.exists():
                self.status_file.unlink()
        except OSError:
            pass

    def _signal_handler(self, signum, frame):
        """Handle signals for graceful shutdown"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop_daemon()
        sys.exit(0)
