import asyncio
import random
import time
from collections.abc import Callable
from typing import Any

from voicebridge.domain.models import RetryConfig, RetryStrategy
from voicebridge.ports.interfaces import RetryService


class WhisperRetryService(RetryService):
    def __init__(self):
        self._retry_stats: dict[str, dict[str, int]] = {}

    def execute_with_retry(self, operation: Callable, config: RetryConfig) -> Any:
        last_exception = None

        for attempt in range(config.max_attempts):
            try:
                return operation()
            except Exception as e:
                last_exception = e

                # Check if error is retryable
                if not self.is_retryable_error(e, config):
                    raise e

                # Don't delay on the last attempt
                if attempt < config.max_attempts - 1:
                    delay = self._calculate_delay(attempt, config)
                    time.sleep(delay)

                # Update stats
                self._update_retry_stats(operation.__name__, "retry")

        # All attempts failed
        self._update_retry_stats(operation.__name__, "failure")
        raise last_exception

    async def execute_with_retry_async(
        self, operation: Callable, config: RetryConfig
    ) -> Any:
        last_exception = None

        for attempt in range(config.max_attempts):
            try:
                if asyncio.iscoroutinefunction(operation):
                    return await operation()
                else:
                    return operation()
            except Exception as e:
                last_exception = e

                # Check if error is retryable
                if not self.is_retryable_error(e, config):
                    raise e

                # Don't delay on the last attempt
                if attempt < config.max_attempts - 1:
                    delay = self._calculate_delay(attempt, config)
                    await asyncio.sleep(delay)

                # Update stats
                self._update_retry_stats(operation.__name__, "retry")

        # All attempts failed
        self._update_retry_stats(operation.__name__, "failure")
        raise last_exception

    def is_retryable_error(self, error: Exception, config: RetryConfig) -> bool:
        error_type_name = type(error).__name__
        error_message = str(error).lower()

        # Check against configured retryable errors
        for retryable_error in config.retryable_errors:
            if retryable_error in error_type_name:
                return True

        # Check common network/temporary errors
        network_errors = [
            "connectionerror",
            "timeout",
            "httperror",
            "networkerror",
            "temporaryerror",
            "serviceunavailable",
            "badgateway",
            "gatewayTimeout",
            "toomanyrequests",
        ]

        for network_error in network_errors:
            if (
                network_error.lower() in error_message
                or network_error.lower() in error_type_name.lower()
            ):
                return True

        # Check HTTP status codes in error messages
        temp_http_codes = ["502", "503", "504", "429", "408"]
        for code in temp_http_codes:
            if code in error_message:
                return True

        return False

    def _calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        if config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = config.initial_delay * (config.exponential_base**attempt)
        elif config.strategy == RetryStrategy.LINEAR:
            delay = config.initial_delay * (attempt + 1)
        else:  # FIXED_INTERVAL
            delay = config.initial_delay

        # Cap at max delay
        delay = min(delay, config.max_delay)

        # Add jitter if enabled
        if config.jitter:
            jitter = random.uniform(0, delay * 0.1)  # Up to 10% jitter
            delay += jitter

        return delay

    def _update_retry_stats(self, operation_name: str, event_type: str) -> None:
        if operation_name not in self._retry_stats:
            self._retry_stats[operation_name] = {
                "attempts": 0,
                "retries": 0,
                "failures": 0,
                "successes": 0,
            }

        self._retry_stats[operation_name]["attempts"] += 1

        if event_type == "retry":
            self._retry_stats[operation_name]["retries"] += 1
        elif event_type == "failure":
            self._retry_stats[operation_name]["failures"] += 1
        elif event_type == "success":
            self._retry_stats[operation_name]["successes"] += 1

    def get_retry_stats(self) -> dict[str, dict[str, int]]:
        return self._retry_stats.copy()

    def reset_stats(self) -> None:
        self._retry_stats.clear()


class RetryDecorator:
    def __init__(self, config: RetryConfig, retry_service: RetryService = None):
        self.config = config
        self.retry_service = retry_service or WhisperRetryService()

    def __call__(self, func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            def operation():
                return func(*args, **kwargs)

            return self.retry_service.execute_with_retry(operation, self.config)

        return wrapper

    @staticmethod
    def with_retry(
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
    ):
        config = RetryConfig(
            max_attempts=max_attempts,
            initial_delay=initial_delay,
            max_delay=max_delay,
            strategy=strategy,
        )
        return RetryDecorator(config)


def retry_on_failure(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    retryable_errors: list[str] = None,
):
    """Convenience decorator for common retry scenarios"""
    config = RetryConfig(
        max_attempts=max_attempts,
        initial_delay=initial_delay,
        retryable_errors=retryable_errors
        or ["ConnectionError", "TimeoutError", "HTTPError"],
    )
    return RetryDecorator(config)
