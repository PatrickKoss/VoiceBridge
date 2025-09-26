import queue
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path

from voicebridge.domain.models import AudioDeviceInfo, PlatformType
from voicebridge.ports.interfaces import AudioRecorder


class WSLAudioRecorder(AudioRecorder):
    """Audio recorder that works in WSL by using Windows FFmpeg to access audio devices."""

    def __init__(self):
        self._state_lock = threading.Lock()
        self._current_process: subprocess.Popen | None = None
        self._current_stop_event: threading.Event | None = None
        self._current_audio_queue: queue.Queue | None = None
        self._current_reader_thread: threading.Thread | None = None
        self._windows_ffmpeg_path = self._find_windows_ffmpeg()

    def _find_windows_ffmpeg(self) -> str | None:
        """Find Windows FFmpeg executable through WSL."""
        possible_paths = [
            "/mnt/c/ProgramData/chocolatey/bin/ffmpeg.exe",
            "/mnt/c/Program Files/ffmpeg/bin/ffmpeg.exe",
            "/mnt/c/Program Files (x86)/ffmpeg/bin/ffmpeg.exe",
            "/mnt/c/ffmpeg/bin/ffmpeg.exe",
        ]

        for path in possible_paths:
            if Path(path).exists():
                return path

        # Try to find via PowerShell
        try:
            result = subprocess.run(
                [
                    "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
                    "-Command",
                    "Get-Command ffmpeg -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                windows_path = result.stdout.strip()
                # Convert Windows path to WSL path
                if windows_path.startswith("C:"):
                    wsl_path = "/mnt/c" + windows_path[2:].replace("\\", "/")
                    if Path(wsl_path).exists():
                        return wsl_path
        except Exception:
            pass

        return None

    def _is_wsl(self) -> bool:
        """Check if running in WSL."""
        try:
            with open("/proc/version") as f:
                return "microsoft" in f.read().lower()
        except Exception:
            return False

    def record_stream(self, sample_rate: int = 16000) -> Iterator[bytes]:
        """Record audio stream using Windows FFmpeg through WSL."""
        if not self._is_wsl():
            raise RuntimeError("WSL audio recorder can only be used in WSL environment")

        if not self._windows_ffmpeg_path:
            raise RuntimeError(
                "Windows FFmpeg not found. Please install FFmpeg on Windows. "
                "You can install it via chocolatey: choco install ffmpeg"
            )

        device = self._get_best_audio_device()
        if not device:
            raise RuntimeError(
                "No audio input device found on Windows. "
                "Please ensure a microphone is connected and enabled in Windows audio settings."
            )

        # Device selected for recording

        stop_event = threading.Event()
        with self._state_lock:
            if self._current_process is not None:
                raise RuntimeError("Audio recording already in progress")
            self._current_stop_event = stop_event

        cmd = self._build_windows_ffmpeg_command(device, sample_rate)

        try:
            # Start recording process that outputs to stdout
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
            )
        except Exception as e:
            with self._state_lock:
                self._current_stop_event = None
            raise RuntimeError(f"Failed to start Windows audio recording: {e}") from e

        with self._state_lock:
            self._current_process = proc

        # Wait a moment for the process to start and potentially fail
        import time

        time.sleep(1.0)

        # Check if process failed during startup
        if proc.poll() is not None:
            try:
                # Use communicate() to get stderr
                _, stderr = proc.communicate(timeout=5)
                stderr = (
                    stderr.decode("utf-8", errors="ignore")
                    if stderr
                    else "No error output available"
                )
            except subprocess.TimeoutExpired:
                proc.kill()
                _, stderr = proc.communicate()
                stderr = (
                    stderr.decode("utf-8", errors="ignore")
                    if stderr
                    else "Process killed due to timeout"
                )
            except Exception as e:
                stderr = f"Error reading process output: {e}"

            # Provide specific error messages for common issues
            if (
                "Could not enumerate audio only devices" in stderr
                or "Could not enumerate audio" in stderr
            ):
                raise RuntimeError(
                    "Windows audio devices not accessible. This usually means:\n"
                    "1. Microphone privacy settings are blocking access\n"
                    "2. No microphone is connected or enabled\n"
                    "3. Windows audio drivers need to be reinstalled\n\n"
                    "To fix this:\n"
                    "• Go to Windows Settings > Privacy & Security > Microphone\n"
                    "• Turn ON 'Allow desktop apps to access your microphone'\n"
                    "• Test your microphone in Windows (try Voice Recorder app)\n"
                    "• Restart WSL after making changes: wsl --shutdown"
                )
            elif "I/O error" in stderr:
                raise RuntimeError(
                    "Audio device I/O error. The microphone may be in use by another application "
                    "or Windows audio permissions are blocking access."
                )
            else:
                raise RuntimeError(f"Windows FFmpeg error: {stderr}")

        chunk_size = sample_rate * 2  # 1 second of 16-bit audio
        audio_queue = queue.Queue()
        error_queue = queue.Queue()

        with self._state_lock:
            self._current_audio_queue = audio_queue

        def read_audio():
            """Read audio data from FFmpeg stdout."""
            try:
                while True:
                    if stop_event.is_set():
                        break

                    data = proc.stdout.read(chunk_size)
                    if not data:
                        # Check if process failed
                        if proc.poll() is not None:
                            break
                    audio_queue.put(data)
            except Exception as e:
                error_queue.put(f"Audio reading error: {e}")
            finally:
                audio_queue.put(None)  # Signal end

        thread = threading.Thread(target=read_audio, daemon=True)
        thread.start()

        with self._state_lock:
            self._current_reader_thread = thread

        try:
            # Check for immediate errors
            if not error_queue.empty():
                error = error_queue.get()
                raise RuntimeError(error)

            while True:
                try:
                    data = audio_queue.get(timeout=1.0)
                    if data is None:
                        break
                    if stop_event.is_set():
                        break
                    yield data
                except queue.Empty:
                    # Check for errors during recording
                    if not error_queue.empty():
                        error = error_queue.get()
                        raise RuntimeError(error) from None
                    continue
        finally:
            self._finalize_recording(proc, stop_event)

    def _get_best_audio_device(self) -> str | None:
        """Get the best available audio input device on Windows."""
        devices = self.list_devices()
        if not devices:
            # If device enumeration fails, try common default device names
            # This is a fallback for when permissions prevent device listing
            # but the actual recording might still work
            common_devices = [
                "default",  # Generic default
                "Microphone",  # Common name
                "Built-in Microphone",  # Built-in devices
                "USB Audio Device",  # USB devices
            ]
            return common_devices[0]  # Try "default" first

        # Prefer USB or built-in microphones
        for device in devices:
            if any(
                keyword in device.name.lower()
                for keyword in ["microphone", "mic", "usb", "built-in"]
            ):
                return device.device_id

        # Return first available device
        return devices[0].device_id

    def _build_windows_ffmpeg_command(self, device: str, sample_rate: int) -> list[str]:
        """Build FFmpeg command to record from Windows audio device."""
        return [
            self._windows_ffmpeg_path,
            "-f",
            "dshow",
            "-i",
            f"audio={device}",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-f",
            "s16le",  # Raw 16-bit little-endian audio
            "pipe:1",  # Output to stdout
        ]

    def list_devices(self) -> list[AudioDeviceInfo]:
        """List available Windows audio devices."""
        if not self._windows_ffmpeg_path:
            return []

        try:
            result = subprocess.run(
                [
                    self._windows_ffmpeg_path,
                    "-list_devices",
                    "true",
                    "-f",
                    "dshow",
                    "-i",
                    "dummy",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            devices = []
            lines = result.stderr.split("\n")

            for line in lines:
                # Look for audio device lines: [dshow @ ...] "Device Name" (audio)
                if (
                    "[dshow @" in line
                    and '"' in line
                    and "(audio)" in line
                    and "Alternative name" not in line
                ):
                    # Parse device line: [dshow @ ...] "Device Name" (audio)
                    if '] "' in line and line.count('"') >= 2:
                        start = line.find('] "') + 3
                        end = line.find('"', start)
                        if start > 2 and end > start:
                            device_name = line[start:end]
                            devices.append(
                                AudioDeviceInfo(
                                    name=device_name,
                                    device_id=device_name,  # Use name as ID for dshow
                                    platform=PlatformType.WINDOWS,
                                )
                            )

            return devices

        except Exception:
            return []

    def stop_current_stream(self) -> None:
        """Stop the current audio stream."""
        with self._state_lock:
            proc = self._current_process
            stop_event = self._current_stop_event
            audio_queue = self._current_audio_queue

        if stop_event:
            stop_event.set()

        if audio_queue:
            audio_queue.put(None)

        if proc:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()

        self._clear_recording_state(proc, stop_event)

    def _finalize_recording(
        self, proc: subprocess.Popen, stop_event: threading.Event
    ) -> None:
        """Clean up recording resources."""
        try:
            if proc:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
        finally:
            self._clear_recording_state(proc, stop_event)

    def _clear_recording_state(
        self,
        expected_proc: subprocess.Popen | None,
        expected_stop_event: threading.Event | None,
    ) -> None:
        """Clear the current recording state."""
        reader_thread: threading.Thread | None = None
        with self._state_lock:
            matches_proc = (
                expected_proc is None or self._current_process is expected_proc
            )
            matches_event = (
                expected_stop_event is None
                or self._current_stop_event is expected_stop_event
            )
            if matches_proc and matches_event:
                reader_thread = self._current_reader_thread
                self._current_stop_event = None
                self._current_process = None
                self._current_audio_queue = None
                self._current_reader_thread = None

        if reader_thread and reader_thread.is_alive():
            reader_thread.join(timeout=0.5)
