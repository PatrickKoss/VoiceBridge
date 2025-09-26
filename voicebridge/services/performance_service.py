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
        """Benchmark model loading and inference performance using real speech samples."""
        import os

        benchmark_results = {
            "model_name": model_name,
            "use_gpu": use_gpu,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            import whisper

            # Check if model exists locally
            model_path = os.path.expanduser(f"~/.cache/whisper/{model_name}.pt")
            model_exists = os.path.exists(model_path)

            # Record initial memory
            initial_memory = self._get_memory_usage()

            # Determine device
            device = "cpu"
            if use_gpu:
                try:
                    import torch

                    if torch.cuda.is_available():
                        device = "cuda"
                    elif (
                        hasattr(torch.backends, "mps")
                        and torch.backends.mps.is_available()
                    ):
                        device = "mps"
                except ImportError:
                    pass

            # Load model and time it
            load_start = time.time()
            if not model_exists:
                print(f"  ► Downloading {model_name} model (first time only)...")

            model = whisper.load_model(model_name, device=device)
            load_time = time.time() - load_start

            if not model_exists:
                print(f"  ✓ Model {model_name} downloaded and loaded")

            # Verify model is on correct device
            model_device = str(next(model.parameters()).device)
            print(f"  ✓ Model loaded on {model_device}")

            # Record memory after model load
            post_load_memory = self._get_memory_usage()
            memory_used = post_load_memory - initial_memory

            # Find and load real speech samples
            voices_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "voices"
            )
            print(f"  ► Loading real speech samples from {voices_dir}...")

            if not os.path.exists(voices_dir):
                raise FileNotFoundError(f"Voices directory not found: {voices_dir}")

            # Get available wav files, prioritizing English samples for benchmarking
            wav_files = [f for f in os.listdir(voices_dir) if f.endswith(".wav")]
            english_files = [f for f in wav_files if f.startswith("en-")]

            if not wav_files:
                raise FileNotFoundError("No audio files found in voices directory")

            # Select test file (prefer shorter English samples for faster benchmarking)
            test_files = english_files if english_files else wav_files
            # Sort by file size to get shorter samples first
            test_files.sort(key=lambda f: os.path.getsize(os.path.join(voices_dir, f)))
            test_file = test_files[0] if test_files else wav_files[0]
            test_audio_path = os.path.join(voices_dir, test_file)

            print(f"  ► Using speech sample: {test_file}")

            # Get audio duration using whisper's built-in loading
            test_audio = whisper.load_audio(test_audio_path)
            duration = len(test_audio) / whisper.audio.SAMPLE_RATE

            print(
                f"  ► Running transcription on {device.upper()} (audio: {duration:.1f}s)..."
            )
            inference_start = time.time()

            # For GPU transcription, ensure we use proper parameters
            transcribe_options = {
                "verbose": False,  # Disable verbose to prevent hanging, we'll add our own progress
                "fp16": use_gpu and device == "cuda",  # Use FP16 only on CUDA
                "language": "english"
                if test_file.startswith("en-")
                else None,  # Auto-detect for non-English
            }

            # Add progress tracking with threading
            import sys
            import threading

            progress_chars = ["|", "/", "-", "\\"]
            progress_counter = 0
            progress_active = True

            def progress_updater():
                nonlocal progress_counter
                while progress_active:
                    char = progress_chars[progress_counter % len(progress_chars)]
                    elapsed = time.time() - inference_start
                    # Show progress with audio context
                    sys.stdout.write(
                        f"\r  {char} Transcribing {duration:.1f}s audio... ({elapsed:.1f}s)"
                    )
                    sys.stdout.flush()
                    progress_counter += 1
                    time.sleep(0.3)

            # Start progress thread
            progress_thread = threading.Thread(target=progress_updater, daemon=True)
            progress_thread.start()

            # Run transcription with progress updates and timeout
            try:
                import signal

                # Set up timeout handler (longer timeout for real audio)
                def timeout_handler(signum, frame):
                    raise TimeoutError("Transcription timed out after 60 seconds")

                # Set timeout based on audio duration (minimum 60 seconds)
                timeout_duration = max(
                    60, int(duration * 10)
                )  # 10x audio duration or 60s minimum
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_duration)

                try:
                    # Run transcription - Whisper handles device placement internally
                    result = model.transcribe(test_audio, **transcribe_options)

                finally:
                    # Cancel the timeout
                    signal.alarm(0)

                # Stop progress updates
                progress_active = False
                time.sleep(0.1)  # Give progress thread time to finish

                inference_time = time.time() - inference_start
                sys.stdout.write("\r" + " " * 60 + "\r")  # Clear progress line
                print(f"  ✓ Transcription completed in {inference_time:.2f}s")

            except TimeoutError as e:
                # Stop progress updates
                progress_active = False
                time.sleep(0.1)  # Give progress thread time to finish
                sys.stdout.write("\r" + " " * 60 + "\r")  # Clear progress line
                print(
                    "  ⚠ Transcription timed out - this may indicate performance issues"
                )
                benchmark_results["error"] = f"Timeout: {e}"
                return benchmark_results

            except Exception as e:
                # Stop progress updates
                progress_active = False
                time.sleep(0.1)  # Give progress thread time to finish
                sys.stdout.write("\r" + " " * 60 + "\r")  # Clear progress line
                print(f"  ✗ Transcription failed: {e}")
                raise

            # Calculate meaningful speech processing metrics
            real_time_factor = duration / inference_time if inference_time > 0 else 0
            throughput_factor = (
                1 / real_time_factor if real_time_factor > 0 else 0
            )  # How many audio seconds per wall-clock second

            # GPU memory usage (if applicable)
            gpu_memory_mb = 0.0
            if device in ["cuda", "mps"]:
                try:
                    if device == "cuda":
                        import torch

                        gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                    # For MPS, we can't easily get memory usage, so estimate
                    elif device == "mps":
                        gpu_memory_mb = memory_used * 0.8  # Rough estimate
                except Exception:
                    gpu_memory_mb = 0.0

            # Extract meaningful transcription info
            transcribed_text = result.get("text", "").strip()
            word_count = len(transcribed_text.split()) if transcribed_text else 0
            words_per_minute = (
                (word_count / (inference_time / 60)) if inference_time > 0 else 0
            )

            # Performance assessment
            performance_rating = (
                "excellent"
                if real_time_factor > 10
                else "good"
                if real_time_factor > 5
                else "fair"
                if real_time_factor > 1
                else "poor"
            )

            benchmark_results.update(
                {
                    "model_load_time": load_time,
                    "inference_time": inference_time,
                    "memory_usage_mb": memory_used,
                    "gpu_memory_mb": gpu_memory_mb if use_gpu else None,
                    "device": device,
                    "actual_device": model_device,
                    "audio_file": test_file,
                    "audio_duration": duration,
                    "real_time_factor": real_time_factor,
                    "throughput_factor": throughput_factor,
                    "words_transcribed": word_count,
                    "words_per_minute_processing": words_per_minute,
                    "performance_rating": performance_rating,
                    "transcribed_text": transcribed_text[:100] + "..."
                    if len(transcribed_text) > 100
                    else transcribed_text,
                }
            )

        except ImportError as e:
            benchmark_results["error"] = f"Missing dependency: {e}"
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
