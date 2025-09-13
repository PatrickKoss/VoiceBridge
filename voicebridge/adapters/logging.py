import logging
from pathlib import Path

from voicebridge.domain.models import PerformanceMetrics
from voicebridge.ports.interfaces import Logger


class FileLogger(Logger):
    def __init__(self, log_file: Path, performance_log: Path, debug: bool = False):
        self.log_file = log_file
        self.performance_log = performance_log

        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Setup main logger
        self.logger = logging.getLogger("whisper-cli")
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)

        # Clear existing handlers
        self.logger.handlers.clear()

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
            )
        )
        self.logger.addHandler(file_handler)

        # Setup performance logger
        self.perf_logger = logging.getLogger("whisper-cli-performance")
        self.perf_logger.setLevel(logging.INFO)
        self.perf_logger.handlers.clear()

        perf_handler = logging.FileHandler(performance_log)
        perf_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        self.perf_logger.addHandler(perf_handler)

    def info(self, message: str) -> None:
        self.logger.info(message)

    def debug(self, message: str) -> None:
        self.logger.debug(message)

    def error(self, message: str) -> None:
        self.logger.error(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def log_performance(self, metrics: PerformanceMetrics) -> None:
        details_str = ", ".join([f"{k}={v}" for k, v in metrics.details.items()])
        message = f"{metrics.operation}: {metrics.duration:.3f}s"
        if details_str:
            message += f" ({details_str})"
        self.perf_logger.info(message)
