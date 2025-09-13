import platform
import shutil
import subprocess

import psutil
from domain.models import GPUInfo, GPUType, PlatformType, SystemInfo
from ports.interfaces import ClipboardService, SystemService


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
        result = subprocess.run(
            ["powershell", "-command", f"Set-Clipboard -Value '{text}'"],
            capture_output=True,
        )
        return result.returncode == 0

    def _copy_macos(self, text: str) -> bool:
        result = subprocess.run(
            ["pbcopy"], input=text.encode("utf-8"), capture_output=True
        )
        return result.returncode == 0

    def _copy_linux(self, text: str) -> bool:
        # Try xclip first, then xsel as fallback
        for cmd in [["xclip", "-selection", "clipboard"], ["xsel", "--clipboard"]]:
            if shutil.which(cmd[0]):
                result = subprocess.run(
                    cmd, input=text.encode("utf-8"), capture_output=True
                )
                return result.returncode == 0
        return False

    def _type_windows(self, text: str) -> bool:
        # Use PowerShell SendKeys for Windows
        escaped_text = text.replace("'", "''")
        result = subprocess.run(
            [
                "powershell",
                "-command",
                f"Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait('{escaped_text}')",
            ],
            capture_output=True,
        )
        return result.returncode == 0

    def _type_macos(self, text: str) -> bool:
        # Use AppleScript for macOS
        escaped_text = text.replace('"', '\\"').replace("\\", "\\\\")
        result = subprocess.run(
            [
                "osascript",
                "-e",
                f'tell application "System Events" to keystroke "{escaped_text}"',
            ],
            capture_output=True,
        )
        return result.returncode == 0

    def _type_linux(self, text: str) -> bool:
        # Try xdotool first, then ydotool as fallback
        for cmd in [["xdotool", "type", text], ["ydotool", "type", text]]:
            if shutil.which(cmd[0]):
                result = subprocess.run(cmd, capture_output=True)
                return result.returncode == 0
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
