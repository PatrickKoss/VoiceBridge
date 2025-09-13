import queue
import subprocess
import threading
from collections.abc import Iterator

from domain.models import AudioDeviceInfo, PlatformType, SystemInfo
from ports.interfaces import AudioRecorder


class FFmpegAudioRecorder(AudioRecorder):
    def __init__(self):
        self.system_info = SystemInfo.current()

    def record_stream(self, sample_rate: int = 16000) -> Iterator[bytes]:
        device = self._get_default_device()
        if not device:
            raise RuntimeError("No audio input device found")

        cmd = self._build_ffmpeg_command(device, sample_rate)

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
        )

        chunk_size = sample_rate * 2  # 1 second of 16-bit audio
        audio_queue = queue.Queue()

        def read_audio():
            try:
                while True:
                    data = proc.stdout.read(chunk_size)
                    if not data:
                        break
                    audio_queue.put(data)
            except Exception:
                pass
            finally:
                audio_queue.put(None)  # Signal end

        thread = threading.Thread(target=read_audio, daemon=True)
        thread.start()

        try:
            while True:
                data = audio_queue.get()
                if data is None:
                    break
                yield data
        finally:
            self._stop_ffmpeg_gracefully(proc)

    def list_devices(self) -> list[AudioDeviceInfo]:
        if self.system_info.platform == PlatformType.WINDOWS:
            return self._list_dshow_devices()
        elif self.system_info.platform == PlatformType.MACOS:
            return self._list_macos_devices()
        else:
            return self._list_linux_devices()

    def _get_default_device(self) -> str:
        devices = self.list_devices()
        return devices[0].device_id if devices else None

    def _build_ffmpeg_command(self, device: str, sample_rate: int) -> list[str]:
        if self.system_info.platform == PlatformType.WINDOWS:
            return [
                "ffmpeg",
                "-f",
                "dshow",
                "-i",
                f"audio={device}",
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                "-f",
                "wav",
                "pipe:1",
            ]
        elif self.system_info.platform == PlatformType.MACOS:
            return [
                "ffmpeg",
                "-f",
                "avfoundation",
                "-i",
                f":{device}",
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                "-f",
                "wav",
                "pipe:1",
            ]
        else:  # Linux
            return [
                "ffmpeg",
                "-f",
                "pulse",
                "-i",
                device,
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                "-f",
                "wav",
                "pipe:1",
            ]

    def _list_dshow_devices(self) -> list[AudioDeviceInfo]:
        try:
            result = subprocess.run(
                ["ffmpeg", "-f", "dshow", "-list_devices", "true", "-i", "dummy"],
                capture_output=True,
                text=True,
            )
            devices = []
            for line in result.stderr.split("\n"):
                if '"' in line and "audio" in line.lower():
                    parts = line.split('"')
                    if len(parts) >= 2:
                        name = parts[1]
                        devices.append(
                            AudioDeviceInfo(
                                name=name, device_id=name, platform=PlatformType.WINDOWS
                            )
                        )
            return devices
        except Exception:
            return []

    def _list_macos_devices(self) -> list[AudioDeviceInfo]:
        try:
            result = subprocess.run(
                ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                capture_output=True,
                text=True,
            )
            devices = []
            in_audio_section = False
            for line in result.stderr.split("\n"):
                # Check if we're entering audio devices section
                if "AVFoundation audio devices:" in line:
                    in_audio_section = True
                    continue
                # Check if we're leaving audio section (entering video)
                if "AVFoundation video devices:" in line:
                    in_audio_section = False
                    continue

                # Parse audio devices in the audio section
                if (
                    in_audio_section
                    and "[AVFoundation indev" in line
                    and "[" in line
                    and "]" in line
                ):
                    # Extract device index and name
                    # Format: [AVFoundation indev @ 0x123] [0] Built-in Microphone
                    parts = line.split("] ")
                    if len(parts) >= 3:
                        # parts[0] = "[AVFoundation indev @ 0x123"
                        # parts[1] = "[0"
                        # parts[2] = "Built-in Microphone"
                        idx_part = parts[1]
                        if "[" in idx_part:
                            idx = idx_part[1:]  # Remove the opening [
                            name = parts[2].strip()
                            if name:  # Only add if name is not empty
                                devices.append(
                                    AudioDeviceInfo(
                                        name=name,
                                        device_id=idx,
                                        platform=PlatformType.MACOS,
                                    )
                                )
            return devices
        except Exception:
            return []

    def _list_linux_devices(self) -> list[AudioDeviceInfo]:
        try:
            result = subprocess.run(
                ["pactl", "list", "short", "sources"], capture_output=True, text=True
            )
            devices = []
            for line in result.stdout.split("\n"):
                if line.strip():
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        device_id = parts[1]
                        name = parts[1] if len(parts) < 3 else parts[2]
                        devices.append(
                            AudioDeviceInfo(
                                name=name,
                                device_id=device_id,
                                platform=PlatformType.LINUX,
                            )
                        )
            return devices
        except Exception:
            return []

    def _stop_ffmpeg_gracefully(self, proc: subprocess.Popen) -> None:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
