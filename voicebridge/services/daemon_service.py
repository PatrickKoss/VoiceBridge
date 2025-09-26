import atexit
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from voicebridge.domain.models import WhisperConfig
from voicebridge.ports.interfaces import DaemonService, Logger


class WhisperDaemonService(DaemonService):
    def __init__(self, pid_file: Path, logger: Logger):
        self.pid_file = pid_file
        self.logger = logger
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)

    def is_running(self) -> bool:
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

        # Fork and detach to create a proper daemon process
        try:
            pid = os.fork()
            if pid > 0:
                # Parent process - just exit, child will write its own PID
                self.logger.info("Daemon starting in background")
                return
        except OSError as e:
            raise RuntimeError(f"Failed to fork daemon process: {e}") from e

        # Child process continues here
        try:
            # Detach from parent environment
            os.setsid()

            # Second fork to ensure daemon can't acquire terminal
            try:
                pid = os.fork()
                if pid > 0:
                    # First child exits
                    sys.exit(0)
            except OSError:
                sys.exit(1)

            # We're now in the daemon process
            self._setup_daemon_environment()
            self._register_cleanup()

            # Run the actual daemon loop
            self._run_daemon_loop(config)

        except Exception as e:
            self.logger.error(f"Daemon failed: {e}")
            sys.exit(1)

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

    def _write_pid_file(self, pid: int = None) -> None:
        with open(self.pid_file, "w") as f:
            f.write(str(pid if pid is not None else os.getpid()))

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

    def _setup_daemon_environment(self) -> None:
        """Setup the daemon environment by redirecting standard file descriptors."""
        # Redirect standard file descriptors
        sys.stdout.flush()
        sys.stderr.flush()

        # Redirect to /dev/null
        with open(os.devnull) as null_in:
            os.dup2(null_in.fileno(), sys.stdin.fileno())
        with open(os.devnull, "w") as null_out:
            os.dup2(null_out.fileno(), sys.stdout.fileno())
            os.dup2(null_out.fileno(), sys.stderr.fileno())

        # Clean up any inherited audio resources to avoid PulseAudio issues
        # This is important because pygame initialization in the parent can cause
        # PulseAudio assertion errors when forked
        try:
            import pygame

            if pygame.get_init():
                pygame.quit()
        except (ImportError, AttributeError):
            pass

    def _run_daemon_loop(self, config: WhisperConfig) -> None:
        """Main daemon loop - this is where the actual daemon work happens."""
        self.logger.info("Daemon loop started")

        # Write our own PID to the file since we're now the daemon process
        self._write_pid_file()

        try:
            # Set up signal handlers for graceful shutdown
            def signal_handler(signum, frame):
                self.logger.info(f"Daemon received signal {signum}")
                raise KeyboardInterrupt()

            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)

            # For now, this is a simple daemon that just stays alive
            # In a real implementation, this would listen for transcription requests,
            # manage audio processing, etc.
            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            self.logger.info("Daemon received interrupt signal")
        except Exception as e:
            self.logger.error(f"Daemon loop error: {e}")
        finally:
            self._cleanup_pid_file()
            self.logger.info("Daemon loop exited")
