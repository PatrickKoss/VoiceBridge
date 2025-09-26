from datetime import datetime

from voicebridge.domain.models import ProgressTracker
from voicebridge.ports.interfaces import ProgressService


class WhisperProgressService(ProgressService):
    def __init__(self, max_trackers: int = 1000):
        self._trackers: dict[str, ProgressTracker] = {}
        self._max_trackers = max_trackers
        self._callbacks: dict[
            str, list
        ] = {}  # operation_id -> list of callback functions

    def create_tracker(
        self, operation_id: str, operation_type: str, total_steps: int = 0
    ) -> ProgressTracker:
        # Clean up old trackers if we have too many
        self._cleanup_old_trackers()

        tracker = ProgressTracker(
            operation_id=operation_id,
            operation_type=operation_type,
            start_time=datetime.now(),
            total_steps=total_steps,
        )

        self._trackers[operation_id] = tracker
        return tracker

    def update_progress(
        self, operation_id: str, progress: float, current_step: str = ""
    ) -> None:
        tracker = self._trackers.get(operation_id)
        if not tracker:
            return

        # Update progress (clamp between 0 and 1)
        tracker.current_progress = max(0.0, min(1.0, progress))
        tracker.current_step = current_step

        # Update step count if provided
        if tracker.total_steps > 0:
            tracker.steps_completed = int(
                tracker.current_progress * tracker.total_steps
            )

        # Calculate ETA
        tracker.eta_seconds = tracker.calculate_eta()

        # Trigger callbacks
        self._trigger_callbacks(operation_id, tracker)

    def complete_operation(self, operation_id: str) -> None:
        tracker = self._trackers.get(operation_id)
        if not tracker:
            return

        tracker.current_progress = 1.0
        tracker.status = "completed"
        tracker.current_step = "Completed"
        tracker.steps_completed = tracker.total_steps
        tracker.eta_seconds = 0.0

        # Trigger final callbacks
        self._trigger_callbacks(operation_id, tracker)

    def fail_operation(self, operation_id: str, error_message: str = "") -> None:
        tracker = self._trackers.get(operation_id)
        if not tracker:
            return

        tracker.status = "failed"
        tracker.current_step = f"Failed: {error_message}" if error_message else "Failed"
        tracker.eta_seconds = None

        # Trigger callbacks
        self._trigger_callbacks(operation_id, tracker)

    def cancel_operation(self, operation_id: str) -> bool:
        tracker = self._trackers.get(operation_id)
        if not tracker:
            return False

        tracker.status = "cancelled"
        tracker.current_step = "Cancelled"
        tracker.eta_seconds = None

        # Trigger callbacks
        self._trigger_callbacks(operation_id, tracker)
        return True

    def get_tracker(self, operation_id: str) -> ProgressTracker | None:
        return self._trackers.get(operation_id)

    def list_active_operations(self) -> list[dict]:
        active_trackers = [
            tracker
            for tracker in self._trackers.values()
            if tracker.status in ["running", "paused"]
        ]
        return [self._tracker_to_dict(tracker) for tracker in active_trackers]

    def get_operation_status(self, operation_id: str) -> dict | None:
        tracker = self._trackers.get(operation_id)
        if not tracker:
            return None
        return self._tracker_to_dict(tracker)

    def get_operation_stats(self) -> dict[str, any]:
        total_operations = len(self._trackers)
        running_operations = sum(
            1 for t in self._trackers.values() if t.status == "running"
        )
        completed_operations = sum(
            1 for t in self._trackers.values() if t.status == "completed"
        )
        failed_operations = sum(
            1 for t in self._trackers.values() if t.status == "failed"
        )

        return {
            "total_operations": total_operations,
            "running_operations": running_operations,
            "completed_operations": completed_operations,
            "failed_operations": failed_operations,
            "success_rate": (
                completed_operations / total_operations if total_operations > 0 else 0.0
            ),
        }

    def register_progress_callback(self, operation_id: str, callback) -> None:
        if operation_id not in self._callbacks:
            self._callbacks[operation_id] = []
        self._callbacks[operation_id].append(callback)

    def unregister_progress_callback(self, operation_id: str, callback) -> None:
        if operation_id in self._callbacks:
            try:
                self._callbacks[operation_id].remove(callback)
            except ValueError:
                pass

    def _trigger_callbacks(self, operation_id: str, tracker: ProgressTracker) -> None:
        callbacks = self._callbacks.get(operation_id, [])
        for callback in callbacks:
            try:
                callback(tracker)
            except Exception:
                # Ignore callback errors
                pass

    def _cleanup_old_trackers(self) -> None:
        if len(self._trackers) < self._max_trackers:
            return

        # Remove oldest completed/failed operations
        completed_trackers = [
            (op_id, tracker)
            for op_id, tracker in self._trackers.items()
            if tracker.status in ["completed", "failed", "cancelled"]
        ]

        # Sort by start time and remove oldest
        completed_trackers.sort(key=lambda x: x[1].start_time)

        # Remove 25% of completed trackers
        remove_count = min(
            len(completed_trackers) // 4, len(self._trackers) - self._max_trackers + 100
        )

        for i in range(remove_count):
            operation_id, _ = completed_trackers[i]
            del self._trackers[operation_id]
            self._callbacks.pop(operation_id, None)

    def _tracker_to_dict(self, tracker: ProgressTracker) -> dict:
        """Convert a ProgressTracker to a dictionary for CLI display."""
        return {
            "id": tracker.operation_id,
            "type": tracker.operation_type,
            "progress": tracker.current_progress * 100,  # Convert to percentage
            "status": tracker.status,
            "eta_seconds": tracker.eta_seconds,
            "started_at": tracker.start_time.isoformat()
            if tracker.start_time
            else None,
            "current_step": tracker.current_step,
            "steps_completed": tracker.steps_completed,
            "total_steps": tracker.total_steps,
        }


class ProgressBar:
    def __init__(
        self, width: int = 50, show_percentage: bool = True, show_eta: bool = True
    ):
        self.width = width
        self.show_percentage = show_percentage
        self.show_eta = show_eta

    def render(self, tracker: ProgressTracker) -> str:
        progress = tracker.current_progress
        filled = int(progress * self.width)
        bar = "█" * filled + "░" * (self.width - filled)

        parts = [f"[{bar}]"]

        if self.show_percentage:
            parts.append(f"{progress * 100:.1f}%")

        if tracker.current_step:
            parts.append(tracker.current_step)

        if (
            self.show_eta
            and tracker.eta_seconds is not None
            and tracker.eta_seconds > 0
        ):
            eta_str = self._format_duration(tracker.eta_seconds)
            parts.append(f"ETA: {eta_str}")

        return " ".join(parts)

    def _format_duration(self, seconds: float) -> str:
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            seconds = int(seconds % 60)
            return f"{minutes}m {seconds}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"


class LiveProgressDisplay:
    def __init__(self, progress_service: ProgressService, update_interval: float = 0.5):
        self.progress_service = progress_service
        self.update_interval = update_interval
        self.progress_bar = ProgressBar()
        self._active_display = None

    def start_display(self, operation_id: str) -> None:
        import sys
        import threading
        import time

        def update_display():
            while True:
                tracker = self.progress_service.get_tracker(operation_id)
                if not tracker or tracker.status in [
                    "completed",
                    "failed",
                    "cancelled",
                ]:
                    break

                # Clear line and print progress
                sys.stdout.write("\r" + " " * 100 + "\r")
                sys.stdout.write(self.progress_bar.render(tracker))
                sys.stdout.flush()

                time.sleep(self.update_interval)

            # Final update
            if tracker:
                sys.stdout.write("\r" + " " * 100 + "\r")
                sys.stdout.write(self.progress_bar.render(tracker) + "\n")
                sys.stdout.flush()

        if self._active_display:
            self._active_display.join()

        self._active_display = threading.Thread(target=update_display, daemon=True)
        self._active_display.start()

    def stop_display(self) -> None:
        if self._active_display:
            self._active_display.join(timeout=1.0)
