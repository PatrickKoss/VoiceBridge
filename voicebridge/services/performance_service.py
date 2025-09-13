import time
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any

from voicebridge.domain.models import PerformanceMetrics
from voicebridge.ports.interfaces import PerformanceService, SystemService


class WhisperPerformanceService(PerformanceService):
    def __init__(self, system_service: SystemService | None = None):
        self._active_timings: dict[str, dict[str, Any]] = {}
        self._performance_history: list[PerformanceMetrics] = []
        self._system_service = system_service

    def start_timing(self, operation: str) -> str:
        """Start timing an operation and return a unique timing ID."""
        timing_id = str(uuid.uuid4())

        self._active_timings[timing_id] = {
            "operation": operation,
            "start_time": time.time(),
            "start_memory": self._get_memory_usage() if self._system_service else None,
        }

        return timing_id

    def end_timing(self, timing_id: str, **details) -> PerformanceMetrics:
        """End timing and create performance metrics."""
        if timing_id not in self._active_timings:
            raise ValueError(f"Timing ID {timing_id} not found")

        timing_info = self._active_timings.pop(timing_id)
        end_time = time.time()
        duration = end_time - timing_info["start_time"]

        # Calculate memory usage if available
        memory_used = None
        if self._system_service and timing_info["start_memory"]:
            current_memory = self._get_memory_usage()
            memory_used = current_memory - timing_info["start_memory"]

        # Override with provided memory_used_mb if available
        if "memory_used_mb" in details:
            memory_used = details.pop("memory_used_mb")

        metrics = PerformanceMetrics(
            operation=timing_info["operation"],
            duration=duration,
            details=details,
            memory_used_mb=memory_used,
            gpu_used=details.get("gpu_used", False),
            gpu_memory_mb=details.get("gpu_memory_mb"),
            model_load_time=details.get("model_load_time"),
            processing_speed_ratio=details.get("processing_speed_ratio"),
        )

        # Store in history
        self._performance_history.append(metrics)

        # Keep only last 1000 metrics to prevent memory bloat
        if len(self._performance_history) > 1000:
            self._performance_history = self._performance_history[-1000:]

        return metrics

    def get_performance_stats(self) -> dict[str, Any]:
        """Get aggregated performance statistics."""
        if not self._performance_history:
            return {"total_operations": 0, "message": "No performance data available"}

        # Group by operation type
        operations = defaultdict(list)
        for metric in self._performance_history:
            operations[metric.operation].append(metric)

        stats = {"total_operations": len(self._performance_history), "operations": {}}

        # Calculate stats per operation type
        for operation, metrics in operations.items():
            durations = [m.duration for m in metrics]
            memory_usage = [
                m.memory_used_mb for m in metrics if m.memory_used_mb is not None
            ]
            gpu_operations = [m for m in metrics if m.gpu_used]

            operation_stats = {
                "count": len(metrics),
                "avg_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "gpu_operations": len(gpu_operations),
                "gpu_percentage": (len(gpu_operations) / len(metrics)) * 100,
            }

            if memory_usage:
                operation_stats.update(
                    {
                        "avg_memory_mb": sum(memory_usage) / len(memory_usage),
                        "max_memory_mb": max(memory_usage),
                    }
                )

            # Add speed ratios for transcription operations
            speed_ratios = [
                m.processing_speed_ratio
                for m in metrics
                if m.processing_speed_ratio is not None
            ]
            if speed_ratios:
                operation_stats["avg_speed_ratio"] = sum(speed_ratios) / len(
                    speed_ratios
                )

            stats["operations"][operation] = operation_stats

        # Overall system stats
        if self._system_service:
            memory_info = self._system_service.get_memory_usage()
            gpu_devices = self._system_service.detect_gpu_devices()

            stats["system"] = {
                "current_memory_mb": memory_info["used_mb"],
                "memory_percentage": memory_info["percent"],
                "available_gpus": len(
                    [gpu for gpu in gpu_devices if gpu.gpu_type.value != "none"]
                ),
            }

        return stats

    def benchmark_model(self, model_name: str, use_gpu: bool = True) -> dict[str, Any]:
        """Benchmark model loading and inference performance."""
        benchmark_results = {
            "model_name": model_name,
            "use_gpu": use_gpu,
            "timestamp": datetime.now().isoformat(),
            "results": {},
        }

        try:
            # This would need to be implemented with actual transcription service
            # For now, return placeholder results
            benchmark_results["results"] = {
                "model_load_time": 0.0,
                "inference_time": 0.0,
                "memory_usage_mb": 0.0,
                "gpu_memory_mb": 0.0 if use_gpu else None,
                "device_used": "cpu" if not use_gpu else "auto",
            }
        except Exception as e:
            benchmark_results["error"] = str(e)

        return benchmark_results

    def get_recent_metrics(
        self, operation: str | None = None, hours: int = 24
    ) -> list[PerformanceMetrics]:
        """Get recent performance metrics, optionally filtered by operation."""

        recent_metrics = []
        for metric in self._performance_history:
            # Note: We don't have timestamp in PerformanceMetrics, so we'll use the last N metrics
            if operation is None or metric.operation == operation:
                recent_metrics.append(metric)

        # Return last 100 metrics that match
        return recent_metrics[-100:]

    def clear_performance_history(self) -> int:
        """Clear performance history and return count of cleared items."""
        count = len(self._performance_history)
        self._performance_history.clear()
        return count

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if self._system_service:
            memory_info = self._system_service.get_memory_usage()
            return memory_info["used_mb"]
        return 0.0
