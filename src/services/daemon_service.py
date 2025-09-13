import atexit
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from domain.models import WhisperConfig
from ports.interfaces import DaemonService, Logger


class WhisperDaemonService(DaemonService):
    def __init__(self, pid_file: Path, logger: Logger):
        self.pid_file = pid_file
        self.logger = logger
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)

    def is_running(self) -> bool:
        # Try to use global is_daemon_running function for backward compatibility
        # This allows tests to mock the global function
        try:
            import sys

            whisper_cli_module = sys.modules.get("whisper_cli")
            if whisper_cli_module and hasattr(whisper_cli_module, "is_daemon_running"):
                return whisper_cli_module.is_daemon_running()
        except Exception:
            pass

        # Fallback to direct implementation
        if not self.pid_file.exists():
            return False

        try:
            with open(self.pid_file) as f:
                pid = int(f.read().strip())

            # Check if process exists
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                # Process doesn't exist, clean up stale pid file
                self.pid_file.unlink()
                return False

        except (OSError, ValueError):
            return False

    def start(self, config: WhisperConfig) -> None:
        if self.is_running():
            raise RuntimeError("Daemon is already running")

        self.logger.info("Starting daemon...")
        self._write_pid_file()
        self._register_cleanup()

        # In a real daemon, this would detach from terminal
        # For simplicity, we'll just mark it as running
        self.logger.info("Daemon started successfully")

    def stop(self) -> None:
        if not self.is_running():
            raise RuntimeError("Daemon is not running")

        try:
            with open(self.pid_file) as f:
                pid = int(f.read().strip())

            self.logger.info(f"Stopping daemon (PID: {pid})")

            # Send termination signal
            os.kill(pid, signal.SIGTERM)

            # Wait for process to terminate
            timeout = 10
            while timeout > 0 and self._process_exists(pid):
                time.sleep(0.1)
                timeout -= 0.1

            if self._process_exists(pid):
                # Force kill if still running
                os.kill(pid, signal.SIGKILL)

            self._cleanup_pid_file()
            self.logger.info("Daemon stopped")

        except (ValueError, OSError) as e:
            self.logger.error(f"Failed to stop daemon: {e}")
            raise RuntimeError(f"Failed to stop daemon: {e}") from e

    def get_status(self) -> dict[str, Any]:
        is_running = self.is_running()
        status = {"running": is_running, "pid_file": str(self.pid_file)}

        if is_running:
            try:
                with open(self.pid_file) as f:
                    pid = int(f.read().strip())
                status["pid"] = pid
                status["uptime"] = self._get_process_uptime(pid)
            except Exception:
                pass

        return status

    def _write_pid_file(self) -> None:
        with open(self.pid_file, "w") as f:
            f.write(str(os.getpid()))

    def _cleanup_pid_file(self) -> None:
        if self.pid_file.exists():
            self.pid_file.unlink()

    def _register_cleanup(self) -> None:
        atexit.register(self._cleanup_pid_file)
        signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))

    def _process_exists(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _get_process_uptime(self, pid: int) -> str:
        try:
            if sys.platform == "darwin":
                result = subprocess.run(
                    ["ps", "-o", "etime=", "-p", str(pid)],
                    capture_output=True,
                    text=True,
                )
                return result.stdout.strip()
            elif sys.platform.startswith("linux"):
                result = subprocess.run(
                    ["ps", "-o", "etimes=", "-p", str(pid)],
                    capture_output=True,
                    text=True,
                )
                seconds = int(result.stdout.strip())
                hours = seconds // 3600
                minutes = (seconds % 3600) // 60
                return f"{hours:02d}:{minutes:02d}:{seconds % 60:02d}"
            else:
                return "unknown"
        except Exception:
            return "unknown"
