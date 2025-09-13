from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from voicebridge.domain.models import CircuitBreakerState
from voicebridge.ports.interfaces import CircuitBreakerService


class CircuitBreakerError(Exception):
    pass


class WhisperCircuitBreakerService(CircuitBreakerService):
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_duration: float = 300.0,  # 5 minutes
        success_threshold: int = 3,
    ):
        self._failure_threshold = failure_threshold
        self._timeout_duration = timeout_duration
        self._success_threshold = success_threshold
        self._states: dict[str, CircuitBreakerState] = {}

    def call(self, operation: Callable, service_name: str) -> Any:
        state = self.get_state(service_name)

        # Check current circuit state
        if state.state == "open":
            if self._should_attempt_reset(state):
                self._transition_to_half_open(service_name)
            else:
                raise CircuitBreakerError(
                    f"Circuit breaker is OPEN for service: {service_name}"
                )

        try:
            # Execute the operation
            result = operation()

            # Record success
            self._record_success(service_name)

            return result

        except Exception as e:
            # Record failure
            self._record_failure(service_name)
            raise e

    def get_state(self, service_name: str) -> CircuitBreakerState:
        if service_name not in self._states:
            self._states[service_name] = CircuitBreakerState()
        return self._states[service_name]

    def reset(self, service_name: str) -> None:
        self._states[service_name] = CircuitBreakerState()

    def force_open(self, service_name: str) -> None:
        state = self.get_state(service_name)
        state.state = "open"
        state.last_failure_time = datetime.now()
        state.next_attempt_time = datetime.now() + timedelta(
            seconds=self._timeout_duration
        )

    def force_close(self, service_name: str) -> None:
        state = self.get_state(service_name)
        state.state = "closed"
        state.failure_count = 0
        state.last_failure_time = None
        state.next_attempt_time = None

    def get_stats(self) -> dict[str, dict[str, Any]]:
        stats = {}
        for service_name, state in self._states.items():
            stats[service_name] = {
                "state": state.state,
                "failure_count": state.failure_count,
                "last_failure_time": (
                    state.last_failure_time.isoformat()
                    if state.last_failure_time
                    else None
                ),
                "next_attempt_time": (
                    state.next_attempt_time.isoformat()
                    if state.next_attempt_time
                    else None
                ),
                "is_available": state.state != "open"
                or self._should_attempt_reset(state),
            }
        return stats

    def _record_success(self, service_name: str) -> None:
        state = self.get_state(service_name)

        if state.state == "half_open":
            # Reset to closed after successful call in half-open state
            self._transition_to_closed(service_name)
        elif state.state == "closed":
            # Reset failure count on success
            state.failure_count = 0

    def _record_failure(self, service_name: str) -> None:
        state = self.get_state(service_name)
        state.failure_count += 1
        state.last_failure_time = datetime.now()

        if state.state == "half_open":
            # Go back to open on failure in half-open state
            self._transition_to_open(service_name)
        elif state.state == "closed" and state.failure_count >= self._failure_threshold:
            # Transition to open when threshold is reached
            self._transition_to_open(service_name)

    def _should_attempt_reset(self, state: CircuitBreakerState) -> bool:
        if state.state != "open" or not state.next_attempt_time:
            return False
        return datetime.now() >= state.next_attempt_time

    def _transition_to_open(self, service_name: str) -> None:
        state = self.get_state(service_name)
        state.state = "open"
        state.next_attempt_time = datetime.now() + timedelta(
            seconds=self._timeout_duration
        )

    def _transition_to_half_open(self, service_name: str) -> None:
        state = self.get_state(service_name)
        state.state = "half_open"
        state.next_attempt_time = None

    def _transition_to_closed(self, service_name: str) -> None:
        state = self.get_state(service_name)
        state.state = "closed"
        state.failure_count = 0
        state.last_failure_time = None
        state.next_attempt_time = None


class CircuitBreaker:
    """Decorator class for circuit breaker functionality"""

    def __init__(
        self,
        service_name: str,
        circuit_breaker_service: CircuitBreakerService = None,
        failure_threshold: int = 5,
        timeout_duration: float = 300.0,
    ):
        self.service_name = service_name
        self.circuit_breaker_service = (
            circuit_breaker_service
            or WhisperCircuitBreakerService(
                failure_threshold=failure_threshold, timeout_duration=timeout_duration
            )
        )

    def __call__(self, func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            def operation():
                return func(*args, **kwargs)

            return self.circuit_breaker_service.call(operation, self.service_name)

        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper


class MultiServiceCircuitBreaker:
    """Circuit breaker that manages multiple services"""

    def __init__(self):
        self._circuit_breakers: dict[str, WhisperCircuitBreakerService] = {}
        self._service_configs: dict[str, dict[str, Any]] = {}

    def register_service(
        self,
        service_name: str,
        failure_threshold: int = 5,
        timeout_duration: float = 300.0,
        success_threshold: int = 3,
    ) -> None:
        self._service_configs[service_name] = {
            "failure_threshold": failure_threshold,
            "timeout_duration": timeout_duration,
            "success_threshold": success_threshold,
        }

        self._circuit_breakers[service_name] = WhisperCircuitBreakerService(
            failure_threshold=failure_threshold,
            timeout_duration=timeout_duration,
            success_threshold=success_threshold,
        )

    def call_service(self, service_name: str, operation: Callable) -> Any:
        if service_name not in self._circuit_breakers:
            self.register_service(service_name)

        return self._circuit_breakers[service_name].call(operation, service_name)

    def get_service_state(self, service_name: str) -> CircuitBreakerState:
        if service_name not in self._circuit_breakers:
            self.register_service(service_name)

        return self._circuit_breakers[service_name].get_state(service_name)

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        all_stats = {}
        for _service_name, cb in self._circuit_breakers.items():
            all_stats.update(cb.get_stats())
        return all_stats

    def reset_service(self, service_name: str) -> None:
        if service_name in self._circuit_breakers:
            self._circuit_breakers[service_name].reset(service_name)

    def reset_all(self) -> None:
        for cb in self._circuit_breakers.values():
            for service_name in self._service_configs.keys():
                cb.reset(service_name)


# Convenience decorator for single service
def circuit_breaker(
    service_name: str, failure_threshold: int = 5, timeout_duration: float = 300.0
):
    """Decorator for adding circuit breaker to a function"""
    return CircuitBreaker(
        service_name,
        failure_threshold=failure_threshold,
        timeout_duration=timeout_duration,
    )
