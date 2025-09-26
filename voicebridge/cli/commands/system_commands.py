import time

import typer

from voicebridge.cli.commands.base import BaseCommands
from voicebridge.cli.utils.command_helpers import (
    display_error,
    display_info,
    display_progress,
    format_duration,
    format_file_size,
)
from voicebridge.domain.models import GPUType


class SystemCommands(BaseCommands):
    """Commands for system monitoring, GPU management, and performance analysis."""

    def gpu_status(self):
        """Show GPU status and availability."""
        if not self.system_service:
            display_error("System service not available")
            return

        try:
            gpu_devices = self.system_service.detect_gpu_devices()
            gpu_info = gpu_devices[0] if gpu_devices else None

            typer.echo("GPU Status:")
            if gpu_info and gpu_info.gpu_type != GPUType.NONE:
                typer.echo("  Status: Available")
                typer.echo(f"  Device: {gpu_info.device_name}")
                typer.echo(f"  Type: {gpu_info.gpu_type.value}")
                typer.echo(
                    f"  Memory Total: {format_file_size(int(gpu_info.memory_total * 1024**2))}"
                )
                typer.echo(
                    f"  Memory Available: {format_file_size(int(gpu_info.memory_available * 1024**2))}"
                )

                # Show compute capability if available
                if gpu_info.compute_capability:
                    typer.echo(f"  Compute Capability: {gpu_info.compute_capability}")

            else:
                typer.echo("  Status: Not available")
                typer.echo("  Running on CPU")

            # Show memory info
            memory_info = self.system_service.get_memory_usage()
            if memory_info:
                typer.echo("\nMemory Info:")
                typer.echo(
                    f"  Total: {format_file_size(int(memory_info['total_mb'] * 1024**2))}"
                )
                typer.echo(
                    f"  Available: {format_file_size(int(memory_info['available_mb'] * 1024**2))}"
                )
                typer.echo(
                    f"  Used: {format_file_size(int(memory_info['used_mb'] * 1024**2))}"
                )
                typer.echo(f"  Usage: {memory_info['percent']:.1f}%")

        except Exception as e:
            display_error(f"Error getting GPU status: {e}")

    def gpu_benchmark(self, model: str = "base"):
        """Benchmark GPU performance with Whisper model."""
        if not self.performance_service:
            display_error("Performance service not available")
            return

        display_info(f"Benchmarking model '{model}' with GPU acceleration...")

        try:
            # Check for GPU availability first
            gpu_devices = (
                self.system_service.detect_gpu_devices() if self.system_service else []
            )
            any(gpu.gpu_type.value != "none" for gpu in gpu_devices)

            # Run GPU benchmark
            typer.echo("\n🚀 Running GPU benchmark:")
            gpu_results = self.performance_service.benchmark_model(model, use_gpu=True)

            # Run CPU benchmark for comparison
            typer.echo("\n🖥️  Running CPU benchmark:")
            cpu_results = self.performance_service.benchmark_model(model, use_gpu=False)

            typer.echo("\n📊 Results:")

            # Display results
            typer.echo("GPU Results:")
            if gpu_results and not gpu_results.get("error"):
                typer.echo(f"  Audio File: {gpu_results.get('audio_file', 'Unknown')}")
                typer.echo(
                    f"  Audio Duration: {gpu_results.get('audio_duration', 0):.1f}s"
                )
                typer.echo(
                    f"  Model Load Time: {gpu_results.get('model_load_time', 0):.2f}s"
                )
                typer.echo(
                    f"  Inference Time: {gpu_results.get('inference_time', 0):.2f}s"
                )
                typer.echo(
                    f"  Memory Usage: {gpu_results.get('memory_usage_mb', 0):.1f}MB"
                )
                typer.echo(
                    f"  Device: {gpu_results.get('device', 'auto')} (actual: {gpu_results.get('actual_device', 'unknown')})"
                )
                if gpu_results.get("real_time_factor"):
                    typer.echo(
                        f"  Real-time Factor: {gpu_results.get('real_time_factor', 0):.1f}x faster than audio"
                    )
                if gpu_results.get("words_transcribed"):
                    typer.echo(
                        f"  Words Transcribed: {gpu_results.get('words_transcribed', 0)}"
                    )
                    typer.echo(
                        f"  Processing Speed: {gpu_results.get('words_per_minute_processing', 0):.0f} words/min"
                    )
                performance = gpu_results.get("performance_rating", "unknown")
                rating_emoji = {
                    "excellent": "🟢",
                    "good": "🟡",
                    "fair": "🟠",
                    "poor": "🔴",
                }.get(performance, "⚪")
                typer.echo(f"  Performance: {rating_emoji} {performance.title()}")
                if gpu_results.get("transcribed_text"):
                    typer.echo(
                        f"  Sample Text: {gpu_results.get('transcribed_text', 'N/A')}"
                    )
            elif gpu_results and gpu_results.get("error"):
                typer.echo(f"  Error: {gpu_results.get('error')}")
            else:
                typer.echo("  No GPU results available")

            typer.echo("\nCPU Results:")
            if cpu_results and not cpu_results.get("error"):
                typer.echo(f"  Audio File: {cpu_results.get('audio_file', 'Unknown')}")
                typer.echo(
                    f"  Audio Duration: {cpu_results.get('audio_duration', 0):.1f}s"
                )
                typer.echo(
                    f"  Model Load Time: {cpu_results.get('model_load_time', 0):.2f}s"
                )
                typer.echo(
                    f"  Inference Time: {cpu_results.get('inference_time', 0):.2f}s"
                )
                typer.echo(
                    f"  Memory Usage: {cpu_results.get('memory_usage_mb', 0):.1f}MB"
                )
                typer.echo(
                    f"  Device: {cpu_results.get('device', 'cpu')} (actual: {cpu_results.get('actual_device', 'unknown')})"
                )
                if cpu_results.get("real_time_factor"):
                    typer.echo(
                        f"  Real-time Factor: {cpu_results.get('real_time_factor', 0):.1f}x faster than audio"
                    )
                if cpu_results.get("words_transcribed"):
                    typer.echo(
                        f"  Words Transcribed: {cpu_results.get('words_transcribed', 0)}"
                    )
                    typer.echo(
                        f"  Processing Speed: {cpu_results.get('words_per_minute_processing', 0):.0f} words/min"
                    )
                performance = cpu_results.get("performance_rating", "unknown")
                rating_emoji = {
                    "excellent": "🟢",
                    "good": "🟡",
                    "fair": "🟠",
                    "poor": "🔴",
                }.get(performance, "⚪")
                typer.echo(f"  Performance: {rating_emoji} {performance.title()}")
                if cpu_results.get("transcribed_text"):
                    typer.echo(
                        f"  Sample Text: {cpu_results.get('transcribed_text', 'N/A')}"
                    )
            elif cpu_results and cpu_results.get("error"):
                typer.echo(f"  Error: {cpu_results.get('error')}")
            else:
                typer.echo("  No CPU results available")

            # Performance comparison
            if (
                gpu_results
                and cpu_results
                and not gpu_results.get("error")
                and not cpu_results.get("error")
            ):
                gpu_time = gpu_results.get("inference_time", 0)
                cpu_time = cpu_results.get("inference_time", 0)
                if gpu_time > 0 and cpu_time > 0:
                    speedup = cpu_time / gpu_time
                    typer.echo(f"\nGPU Speedup: {speedup:.1f}x faster than CPU")

                # Compare real-time factors
                gpu_rtf = gpu_results.get("real_time_factor", 0)
                cpu_rtf = cpu_results.get("real_time_factor", 0)
                if gpu_rtf > 0 and cpu_rtf > 0:
                    typer.echo(f"GPU processes audio {gpu_rtf:.1f}x real-time")
                    typer.echo(f"CPU processes audio {cpu_rtf:.1f}x real-time")

        except Exception as e:
            display_error(f"Benchmark failed: {e}")

    def performance_stats(self):
        """Show performance statistics and metrics."""
        if not self.performance_service:
            display_error("Performance service not available")
            return

        try:
            stats = self.performance_service.get_performance_stats()

            typer.echo("Performance Statistics:")
            typer.echo(f"  Operations tracked: {stats.get('total_operations', 0)}")

            # Processing times
            if "processing_times" in stats:
                times = stats["processing_times"]
                typer.echo(
                    f"  Average processing time: {format_duration(times.get('average', 0))}"
                )
                typer.echo(
                    f"  Min processing time: {format_duration(times.get('min', 0))}"
                )
                typer.echo(
                    f"  Max processing time: {format_duration(times.get('max', 0))}"
                )

            # Memory usage
            if "memory_usage" in stats:
                memory = stats["memory_usage"]
                typer.echo(
                    f"  Average memory usage: {format_file_size(memory.get('average', 0))}"
                )
                typer.echo(
                    f"  Peak memory usage: {format_file_size(memory.get('peak', 0))}"
                )

            # GPU utilization
            if "gpu_utilization" in stats:
                gpu = stats["gpu_utilization"]
                typer.echo(f"  Average GPU utilization: {gpu.get('average', 0):.1f}%")
                typer.echo(f"  Peak GPU utilization: {gpu.get('peak', 0):.1f}%")

            # Error rates
            if "error_rates" in stats:
                errors = stats["error_rates"]
                typer.echo(f"  Success rate: {errors.get('success_rate', 0):.1%}")
                typer.echo(f"  Total errors: {errors.get('total_errors', 0)}")

            # Recent performance trends
            if "trends" in stats:
                trends = stats["trends"]
                typer.echo("\nRecent trends:")
                for period, data in trends.items():
                    typer.echo(
                        f"  {period}: {data.get('operations', 0)} operations, "
                        f"avg time {format_duration(data.get('avg_time', 0))}"
                    )

        except Exception as e:
            display_error(f"Error getting performance stats: {e}")

    def sessions_list(self):
        """List all transcription sessions."""
        if not self.session_service:
            display_error("Session service not available")
            return

        try:
            sessions = self.session_service.list_sessions()

            if not sessions:
                display_info("No sessions found")
                return

            typer.echo("Transcription Sessions:")
            typer.echo(
                f"{'ID':<8} {'Name':<20} {'Status':<12} {'Duration':<10} {'Created':<20}"
            )
            typer.echo("-" * 80)

            for session in sessions:
                session_id = session.get("id", "")[:8]
                name = session.get("name", "Unknown")[:20]
                status = session.get("status", "Unknown")
                duration = format_duration(session.get("duration", 0))
                created = session.get("created_at", "Unknown")[:20]

                typer.echo(
                    f"{session_id:<8} {name:<20} {status:<12} {duration:<10} {created:<20}"
                )

        except Exception as e:
            display_error(f"Error listing sessions: {e}")

    def sessions_resume(
        self,
        session_id: str | None = None,
        session_name: str | None = None,
    ):
        """Resume a transcription session."""
        if not self.session_service or not self.resume_service:
            display_error("Resume service not available")
            return

        if not session_id and not session_name:
            display_error("Either session_id or session_name must be provided")
            return

        try:
            # Find session
            if session_name:
                session = self.session_service.get_session_by_name(session_name)
                if not session:
                    display_error(f"Session not found: {session_name}")
                    return
                session_id = session.get("id")

            if not session_id:
                display_error("Could not determine session ID")
                return

            display_info(f"Resuming session: {session_id}")

            # Resume transcription
            success = self.resume_service.resume_session(session_id)

            if success:
                display_progress("Session resumed successfully", finished=True)

                # Monitor progress
                while True:
                    status = self.session_service.get_session_status(session_id)
                    if not status or status.get("status") == "completed":
                        break

                    progress = status.get("progress", {})
                    completed = progress.get("completed_chunks", 0)
                    total = progress.get("total_chunks", 0)

                    if total > 0:
                        percentage = (completed / total) * 100
                        typer.echo(
                            f"Progress: {completed}/{total} chunks ({percentage:.1f}%)"
                        )

                    time.sleep(2)

                display_progress("Session completed", finished=True)
            else:
                display_error("Failed to resume session")

        except KeyboardInterrupt:
            display_info("Session paused. Use resume command to continue.")
        except Exception as e:
            display_error(f"Error resuming session: {e}")

    def sessions_cleanup(self):
        """Clean up old and completed sessions."""
        if not self.session_service:
            display_error("Session service not available")
            return

        try:
            result = self.session_service.cleanup_sessions()

            cleaned_count = result.get("cleaned_sessions", 0)
            freed_space = result.get("freed_space_mb", 0)

            display_progress("Cleanup completed", finished=True)
            typer.echo(f"  Sessions cleaned: {cleaned_count}")
            typer.echo(f"  Space freed: {format_file_size(freed_space * 1024 * 1024)}")

        except Exception as e:
            display_error(f"Error during cleanup: {e}")

    def sessions_delete(self, session_id: str):
        """Delete a specific session."""
        if not self.session_service:
            display_error("Session service not available")
            return

        try:
            # Check if session exists
            session = self.session_service.get_session(session_id)
            if not session:
                display_error(f"Session not found: {session_id}")
                return

            # Confirm deletion
            session_name = session.get("name", "Unknown")
            typer.confirm(
                f"Delete session '{session_name}' ({session_id})? This cannot be undone.",
                abort=True,
            )

            success = self.session_service.delete_session(session_id)

            if success:
                display_progress("Session deleted successfully", finished=True)
            else:
                display_error("Failed to delete session")

        except typer.Abort:
            display_info("Session deletion cancelled")
        except Exception as e:
            display_error(f"Error deleting session: {e}")

    def _display_benchmark_results(self, results: dict):
        """Display benchmark results in a formatted way."""
        typer.echo("\nBenchmark Results:")
        typer.echo("=" * 50)

        # System info
        if "system_info" in results:
            sys_info = results["system_info"]
            typer.echo("System Information:")
            typer.echo(f"  GPU: {sys_info.get('gpu_name', 'Unknown')}")
            typer.echo(
                f"  GPU Memory: {format_file_size(sys_info.get('gpu_memory', 0))}"
            )
            typer.echo(f"  CPU: {sys_info.get('cpu_name', 'Unknown')}")
            typer.echo(f"  RAM: {format_file_size(sys_info.get('ram_total', 0))}")
            typer.echo()

        # Benchmark metrics
        if "metrics" in results:
            metrics = results["metrics"]
            typer.echo("Performance Metrics:")
            typer.echo(
                f"  Model load time: {format_duration(metrics.get('model_load_time', 0))}"
            )
            typer.echo(
                f"  Processing time: {format_duration(metrics.get('processing_time', 0))}"
            )
            typer.echo(
                f"  Memory usage: {format_file_size(metrics.get('memory_usage', 0))}"
            )
            typer.echo(f"  GPU utilization: {metrics.get('gpu_utilization', 0):.1f}%")
            typer.echo(
                f"  Audio processed: {format_duration(metrics.get('audio_duration', 0))}"
            )
            typer.echo(f"  Real-time factor: {metrics.get('real_time_factor', 0):.2f}x")
            typer.echo()

        # Recommendations
        if "recommendations" in results:
            recommendations = results["recommendations"]
            typer.echo("Recommendations:")
            for rec in recommendations:
                typer.echo(f"  • {rec}")
            typer.echo()

        # Performance rating
        if "performance_rating" in results:
            rating = results["performance_rating"]
            rating_text = {
                "excellent": "🟢 Excellent",
                "good": "🟡 Good",
                "fair": "🟠 Fair",
                "poor": "🔴 Poor",
            }.get(rating, rating)
            typer.echo(f"Overall Performance: {rating_text}")

    def operations_list(self):
        """List active operations."""
        if not self.progress_service:
            display_error("Progress service not available")
            return

        try:
            operations = self.progress_service.list_active_operations()

            if not operations:
                display_info("No active operations")
                return

            typer.echo("Active Operations:")
            typer.echo(
                f"{'ID':<8} {'Type':<15} {'Progress':<10} {'Status':<12} {'ETA':<10}"
            )
            typer.echo("-" * 65)

            for op in operations:
                op_id = op.get("id", "")[:8]
                op_type = op.get("type", "Unknown")[:15]
                progress = f"{op.get('progress', 0):.1f}%"
                status = op.get("status", "Unknown")
                eta = format_duration(op.get("eta_seconds", 0))

                typer.echo(
                    f"{op_id:<8} {op_type:<15} {progress:<10} {status:<12} {eta:<10}"
                )

        except Exception as e:
            display_error(f"Error listing operations: {e}")

    def operations_cancel(self, operation_id: str):
        """Cancel a specific operation."""
        if not self.progress_service:
            display_error("Progress service not available")
            return

        try:
            success = self.progress_service.cancel_operation(operation_id)

            if success:
                display_progress(f"Operation {operation_id} cancelled", finished=True)
            else:
                display_error(f"Failed to cancel operation {operation_id}")

        except Exception as e:
            display_error(f"Error cancelling operation: {e}")

    def operations_status(self, operation_id: str):
        """Show status of a specific operation."""
        if not self.progress_service:
            display_error("Progress service not available")
            return

        try:
            status = self.progress_service.get_operation_status(operation_id)

            if not status:
                display_error(f"Operation not found: {operation_id}")
                return

            typer.echo(f"Operation Status: {operation_id}")
            typer.echo(f"  Type: {status.get('type', 'Unknown')}")
            typer.echo(f"  Status: {status.get('status', 'Unknown')}")
            typer.echo(f"  Progress: {status.get('progress', 0):.1f}%")
            typer.echo(f"  Started: {status.get('started_at', 'Unknown')}")

            if status.get("eta_seconds"):
                typer.echo(f"  ETA: {format_duration(status['eta_seconds'])}")

            if status.get("error"):
                typer.echo(f"  Error: {status['error']}")

        except Exception as e:
            display_error(f"Error getting operation status: {e}")

    def circuit_breaker_status(self):
        """Show circuit breaker status for all services."""
        if not self.circuit_breaker_service:
            display_error("Circuit breaker service not available")
            return

        try:
            status = self.circuit_breaker_service.get_all_circuit_status()

            typer.echo("Circuit Breaker Status:")
            typer.echo(
                f"{'Service':<20} {'State':<10} {'Failures':<8} {'Success Rate':<12} {'Next Retry':<15}"
            )
            typer.echo("-" * 75)

            for service_name, service_status in status.items():
                state = service_status.get("state", "Unknown")
                failures = service_status.get("failure_count", 0)
                success_rate = f"{service_status.get('success_rate', 0):.1%}"
                next_retry = service_status.get("next_retry_at", "N/A")

                typer.echo(
                    f"{service_name:<20} {state:<10} {failures:<8} {success_rate:<12} {next_retry:<15}"
                )

        except Exception as e:
            display_error(f"Error getting circuit breaker status: {e}")

    def circuit_breaker_reset(self, service: str | None = None):
        """Reset circuit breaker for a service or all services."""
        if not self.circuit_breaker_service:
            display_error("Circuit breaker service not available")
            return

        try:
            if service:
                success = self.circuit_breaker_service.reset_circuit(service)
                if success:
                    display_progress(
                        f"Circuit breaker reset for {service}", finished=True
                    )
                else:
                    display_error(f"Failed to reset circuit breaker for {service}")
            else:
                self.circuit_breaker_service.reset_all_circuits()
                display_progress("All circuit breakers reset", finished=True)

        except Exception as e:
            display_error(f"Error resetting circuit breaker: {e}")

    def circuit_breaker_stats(self):
        """Show circuit breaker statistics."""
        if not self.circuit_breaker_service:
            display_error("Circuit breaker service not available")
            return

        try:
            stats = self.circuit_breaker_service.get_circuit_stats()

            typer.echo("Circuit Breaker Statistics:")
            for service_name, service_stats in stats.items():
                typer.echo(f"\n{service_name}:")
                typer.echo(
                    f"  Total requests: {service_stats.get('total_requests', 0)}"
                )
                typer.echo(
                    f"  Successful requests: {service_stats.get('successful_requests', 0)}"
                )
                typer.echo(
                    f"  Failed requests: {service_stats.get('failed_requests', 0)}"
                )
                typer.echo(f"  Circuit trips: {service_stats.get('circuit_trips', 0)}")
                typer.echo(
                    f"  Average response time: {format_duration(service_stats.get('avg_response_time', 0))}"
                )

        except Exception as e:
            display_error(f"Error getting circuit breaker stats: {e}")
