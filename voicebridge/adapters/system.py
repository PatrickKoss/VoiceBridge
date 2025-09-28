import os
import platform
import shutil
import subprocess

import psutil

from voicebridge.domain.models import GPUInfo, GPUType, PlatformType, SystemInfo
from voicebridge.ports.interfaces import ClipboardService, SystemService

try:
    import pyperclip
except ImportError:
    pyperclip = None

try:
    from pynput.keyboard import Controller as KeyboardController
except ImportError:
    KeyboardController = None


class PlatformClipboardService(ClipboardService):
    def __init__(self):
        self.system_info = SystemInfo.current()

    def copy_text(self, text: str) -> bool:
        try:
            if self.system_info.platform == PlatformType.WINDOWS:
                return self._copy_windows(text)
            elif self.system_info.platform == PlatformType.MACOS:
                return self._copy_macos(text)
            else:
                return self._copy_linux(text)
        except Exception:
            return False

    def type_text(self, text: str) -> bool:
        try:
            if self.system_info.platform == PlatformType.WINDOWS:
                return self._type_windows(text)
            elif self.system_info.platform == PlatformType.MACOS:
                return self._type_macos(text)
            else:
                return self._type_linux(text)
        except Exception:
            return False

    def _copy_windows(self, text: str) -> bool:
        # Try pyperclip first (most reliable)
        if pyperclip:
            try:
                pyperclip.copy(text)
                return True
            except Exception:
                pass

        # Fallback to PowerShell with proper escaping
        try:
            # Escape special characters for PowerShell
            escaped_text = text.replace("'", "''").replace("`", "``").replace("$", "`$")
            result = subprocess.run(
                ["powershell", "-command", f"Set-Clipboard -Value '{escaped_text}'"],
                capture_output=True,
                timeout=5,
                text=True,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Final fallback to clip.exe
        try:
            result = subprocess.run(
                ["clip"], input=text, capture_output=True, timeout=5, text=True
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _copy_macos(self, text: str) -> bool:
        try:
            result = subprocess.run(
                ["pbcopy"], input=text.encode("utf-8"), capture_output=True, timeout=5
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False

    def _copy_linux(self, text: str) -> bool:
        # Check if we're in WSL and try Windows clipboard tools first
        if os.path.exists("/proc/version"):
            try:
                with open("/proc/version") as f:
                    if "microsoft" in f.read().lower():
                        # We're in WSL, try PowerShell through WSL interop
                        try:
                            # Use PowerShell Set-Clipboard through WSL interop
                            escaped_text = text.replace("'", "''")
                            result = subprocess.run(
                                [
                                    "powershell.exe",
                                    "-command",
                                    f"Set-Clipboard -Value '{escaped_text}'",
                                ],
                                capture_output=True,
                                timeout=5,
                            )
                            if result.returncode == 0:
                                return True
                        except (
                            subprocess.CalledProcessError,
                            subprocess.TimeoutExpired,
                            FileNotFoundError,
                        ):
                            pass

                        # Try clip.exe as fallback
                        try:
                            result = subprocess.run(
                                ["clip.exe"],
                                input=text.encode("utf-8"),
                                capture_output=True,
                                timeout=5,
                            )
                            if result.returncode == 0:
                                return True
                        except (
                            subprocess.CalledProcessError,
                            subprocess.TimeoutExpired,
                            FileNotFoundError,
                        ):
                            pass

                        # Try clip.exe via direct Windows path
                        try:
                            result = subprocess.run(
                                ["/mnt/c/Windows/System32/clip.exe"],
                                input=text.encode("utf-8"),
                                capture_output=True,
                                timeout=5,
                            )
                            if result.returncode == 0:
                                return True
                        except (
                            subprocess.CalledProcessError,
                            subprocess.TimeoutExpired,
                            FileNotFoundError,
                        ):
                            pass
            except OSError:
                pass

        # Try Linux clipboard tools
        for cmd in [["xclip", "-selection", "clipboard"], ["xsel", "--clipboard"]]:
            if shutil.which(cmd[0]):
                process = None
                try:
                    # Use Popen to properly handle stdin closure for xclip
                    process = subprocess.Popen(
                        cmd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    stdout, stderr = process.communicate(
                        input=text.encode("utf-8"), timeout=5
                    )
                    return process.returncode == 0
                except subprocess.TimeoutExpired:
                    # If this clipboard tool times out, kill it and try the next one
                    if process:
                        process.kill()
                        process.wait()
                    continue
                except Exception:
                    # If any other error occurs, try the next tool
                    if process:
                        try:
                            process.kill()
                            process.wait()
                        except Exception:
                            pass
                    continue
        return False

    def _type_windows(self, text: str) -> bool:
        # Try pynput first (most reliable)
        if KeyboardController:
            try:
                keyboard = KeyboardController()
                keyboard.type(text)
                return True
            except Exception:
                pass

        # Fallback to PowerShell SendKeys
        escaped_text = text.replace("'", "''").replace("`", "``").replace("$", "`$")
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-command",
                    f"Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait('{escaped_text}')",
                ],
                capture_output=True,
                timeout=10,  # Typing can take longer than clipboard operations
                text=True,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _type_macos(self, text: str) -> bool:
        # Try pynput first (most reliable)
        if KeyboardController:
            try:
                keyboard = KeyboardController()
                keyboard.type(text)
                return True
            except Exception:
                pass

        # Fallback to AppleScript
        escaped_text = text.replace('"', '\\"').replace("\\", "\\\\")
        try:
            result = subprocess.run(
                [
                    "osascript",
                    "-e",
                    f'tell application "System Events" to keystroke "{escaped_text}"',
                ],
                capture_output=True,
                timeout=10,  # Typing can take longer than clipboard operations
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False

    def _type_linux(self, text: str) -> bool:
        # Try pynput first (most reliable)
        if KeyboardController:
            try:
                keyboard = KeyboardController()
                keyboard.type(text)
                return True
            except Exception:
                pass

        # Fallback to xdotool/ydotool
        for cmd in [["xdotool", "type", text], ["ydotool", "type", text]]:
            if shutil.which(cmd[0]):
                try:
                    result = subprocess.run(cmd, capture_output=True, timeout=10)
                    return result.returncode == 0
                except subprocess.TimeoutExpired:
                    # If this tool times out, try the next one
                    continue
        return False


class StandardSystemService(SystemService):
    def get_system_info(self) -> SystemInfo:
        return SystemInfo.current()

    def ensure_dependencies(self) -> bool:
        required_tools = self._get_required_tools()
        missing = []

        for tool in required_tools:
            if not shutil.which(tool):
                missing.append(tool)

        if missing:
            raise RuntimeError(f"Missing required dependencies: {', '.join(missing)}")

        return True

    def _get_required_tools(self) -> list[str]:
        system = SystemInfo.current()
        tools = ["ffmpeg"]

        if system.platform == PlatformType.LINUX:
            tools.extend(["pactl"])  # PulseAudio for Linux

        return tools

    def detect_gpu_devices(self) -> list[GPUInfo]:
        """Detect available GPU devices for acceleration."""
        gpu_devices = []

        # Try CUDA detection
        cuda_devices = self._detect_cuda_devices()
        gpu_devices.extend(cuda_devices)

        # Try Metal detection (Apple Silicon)
        if platform.system() == "Darwin":
            metal_device = self._detect_metal_device()
            if metal_device:
                gpu_devices.append(metal_device)

        # If no GPU found, return CPU info
        if not gpu_devices:
            gpu_devices.append(
                GPUInfo(
                    gpu_type=GPUType.NONE,
                    device_name="CPU",
                    memory_total=int(psutil.virtual_memory().total / (1024 * 1024)),
                    memory_available=int(
                        psutil.virtual_memory().available / (1024 * 1024)
                    ),
                )
            )

        return gpu_devices

    def _detect_cuda_devices(self) -> list[GPUInfo]:
        """Detect CUDA-capable devices."""
        devices = []
        try:
            # Try to use nvidia-ml-py if available
            import pynvml

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                devices.append(
                    GPUInfo(
                        gpu_type=GPUType.CUDA,
                        device_name=name,
                        memory_total=memory_info.total // (1024 * 1024),
                        memory_available=memory_info.free // (1024 * 1024),
                        compute_capability=self._get_cuda_compute_capability(handle),
                    )
                )

        except (ImportError, Exception):
            # Fallback to nvidia-smi
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,memory.total,memory.free",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        parts = line.split(", ")
                        if len(parts) == 3:
                            name, total_mem, free_mem = parts
                            devices.append(
                                GPUInfo(
                                    gpu_type=GPUType.CUDA,
                                    device_name=name.strip(),
                                    memory_total=int(total_mem),
                                    memory_available=int(free_mem),
                                )
                            )
            except (
                FileNotFoundError,
                subprocess.SubprocessError,
                ValueError,
                Exception,
            ):
                pass

        return devices

    def _detect_metal_device(self) -> GPUInfo | None:
        """Detect Metal-capable devices on macOS."""
        try:
            # Check for Apple Silicon
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and "Apple" in result.stdout:
                # Get memory info
                memory_result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
                )
                total_memory = 0
                if memory_result.returncode == 0:
                    total_memory = int(memory_result.stdout.strip()) // (1024 * 1024)

                return GPUInfo(
                    gpu_type=GPUType.METAL,
                    device_name=result.stdout.strip(),
                    memory_total=total_memory,
                    memory_available=int(total_memory * 0.8),  # Estimate 80% available
                )
        except (FileNotFoundError, subprocess.SubprocessError):
            pass

        return None

    def _get_cuda_compute_capability(self, handle) -> str | None:
        """Get CUDA compute capability if available."""
        try:
            import pynvml

            major = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[0]
            minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[1]
            return f"{major}.{minor}"
        except Exception:
            return None

    def get_memory_usage(self) -> dict[str, float]:
        """Get current memory usage statistics."""
        memory = psutil.virtual_memory()
        return {
            "total_mb": memory.total / (1024 * 1024),
            "available_mb": memory.available / (1024 * 1024),
            "used_mb": memory.used / (1024 * 1024),
            "percent": memory.percent,
        }
