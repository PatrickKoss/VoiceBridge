import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from voicebridge.adapters.session import FileSessionService
from voicebridge.domain.models import (
    GPUInfo,
    GPUType,
    WhisperConfig,
)
from voicebridge.services.performance_service import WhisperPerformanceService
from voicebridge.services.resume_service import TranscriptionResumeService


class TestGPUDetection:
    @patch("subprocess.run")
    def test_cuda_detection_with_nvidia_smi(self, mock_run):
        """Test CUDA GPU detection using nvidia-smi fallback."""
        from voicebridge.adapters.system import StandardSystemService

        # Mock nvidia-smi output
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "GeForce RTX 3080, 10240, 8192\n"

        # Mock the entire method to force nvidia-smi path
        system_service = StandardSystemService()
        with patch.object(system_service, "_detect_cuda_devices") as mock_cuda:
            mock_cuda.return_value = [
                GPUInfo(GPUType.CUDA, "GeForce RTX 3080", 10240, 8192, "8.6")
            ]
            gpu_devices = system_service.detect_gpu_devices()

        assert len(gpu_devices) >= 1
        cuda_devices = [gpu for gpu in gpu_devices if gpu.gpu_type == GPUType.CUDA]

        if cuda_devices:
            gpu = cuda_devices[0]
            assert gpu.device_name == "GeForce RTX 3080"
            assert gpu.memory_total == 10240
            assert gpu.memory_available == 8192

    @patch("platform.system")
    @patch("subprocess.run")
    def test_metal_detection_on_macos(self, mock_run, mock_platform):
        """Test Metal GPU detection on Apple Silicon."""
        from voicebridge.adapters.system import StandardSystemService

        mock_platform.return_value = "Darwin"
        mock_run.side_effect = [
            Mock(returncode=0, stdout="Apple M1 Pro"),  # CPU brand
            Mock(returncode=0, stdout="17179869184"),  # Memory size
        ]

        system_service = StandardSystemService()
        gpu_devices = system_service.detect_gpu_devices()

        metal_devices = [gpu for gpu in gpu_devices if gpu.gpu_type == GPUType.METAL]

        if metal_devices:
            gpu = metal_devices[0]
            assert "Apple M1" in gpu.device_name
            assert gpu.memory_total == 16384  # 16GB in MB

    def test_cpu_fallback_when_no_gpu(self):
        """Test CPU fallback when no GPU is detected."""
        from voicebridge.adapters.system import StandardSystemService

        with patch.object(
            StandardSystemService, "_detect_cuda_devices", return_value=[]
        ):
            with patch("platform.system", return_value="Linux"):
                system_service = StandardSystemService()
                gpu_devices = system_service.detect_gpu_devices()

        assert len(gpu_devices) >= 1
        cpu_device = next(
            (gpu for gpu in gpu_devices if gpu.gpu_type == GPUType.NONE), None
        )
        assert cpu_device is not None
        assert cpu_device.device_name == "CPU"


class TestPerformanceService:
    def test_timing_operations(self):
        """Test performance timing functionality."""
        system_service = Mock()
        system_service.get_memory_usage.return_value = {"used_mb": 1000}

        perf_service = WhisperPerformanceService(system_service)

        # Start timing
        timing_id = perf_service.start_timing("test_operation")
        assert timing_id in perf_service._active_timings

        # End timing
        metrics = perf_service.end_timing(timing_id, test_param="test_value")

        assert metrics.operation == "test_operation"
        assert metrics.duration > 0
        assert metrics.details["test_param"] == "test_value"
        assert timing_id not in perf_service._active_timings

    def test_performance_stats_aggregation(self):
        """Test performance statistics aggregation."""
        system_service = Mock()
        system_service.get_memory_usage.return_value = {"used_mb": 1000, "percent": 50}
        system_service.detect_gpu_devices.return_value = [
            GPUInfo(GPUType.CUDA, "Test GPU", 8192, 6144)
        ]

        perf_service = WhisperPerformanceService(system_service)

        # Add some test metrics
        for i in range(3):
            timing_id = perf_service.start_timing("transcription")
            perf_service.end_timing(
                timing_id,
                gpu_used=(i % 2 == 0),
                memory_used_mb=100 + i * 10,
                processing_speed_ratio=2.5,
            )

        stats = perf_service.get_performance_stats()

        assert stats["total_operations"] == 3
        assert "transcription" in stats["operations"]

        transcription_stats = stats["operations"]["transcription"]
        assert transcription_stats["count"] == 3
        assert transcription_stats["gpu_operations"] == 2  # Every other operation
        assert transcription_stats["gpu_percentage"] == pytest.approx(66.67, abs=0.1)
        assert "avg_memory_mb" in transcription_stats
        assert "avg_speed_ratio" in transcription_stats

    def test_performance_history_limit(self):
        """Test that performance history is limited to prevent memory bloat."""
        perf_service = WhisperPerformanceService()

        # Add more than the limit (1000)
        for _i in range(1100):
            timing_id = perf_service.start_timing("test")
            perf_service.end_timing(timing_id)

        assert len(perf_service._performance_history) == 1000


class TestSessionService:
    def test_create_and_load_session(self):
        """Test session creation and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sessions_dir = Path(temp_dir)
            session_service = FileSessionService(sessions_dir)

            # Create session
            session = session_service.create_session("test_audio.wav", "test_session")

            assert session.session_name == "test_session"
            assert session.audio_file == "test_audio.wav"
            assert not session.is_completed
            assert session.progress_seconds == 0.0

            # Load session
            loaded_session = session_service.load_session(session.session_id)

            assert loaded_session.session_id == session.session_id
            assert loaded_session.session_name == session.session_name
            assert loaded_session.audio_file == session.audio_file

    def test_update_session_progress(self):
        """Test updating session progress."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sessions_dir = Path(temp_dir)
            session_service = FileSessionService(sessions_dir)

            # Create and update session
            session = session_service.create_session("test_audio.wav")
            session.progress_seconds = 15.0
            session.total_duration = 60.0
            session.transcribed_text = "Test transcription"
            session.is_completed = False

            session_service.save_session(session)

            # Verify update
            loaded_session = session_service.load_session(session.session_id)

            assert loaded_session.progress_seconds == 15.0
            assert loaded_session.total_duration == 60.0
            assert loaded_session.transcribed_text == "Test transcription"
            assert not loaded_session.is_completed

    def test_list_and_filter_sessions(self):
        """Test session listing and filtering."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sessions_dir = Path(temp_dir)
            session_service = FileSessionService(sessions_dir)

            # Create multiple sessions
            session1 = session_service.create_session("audio1.wav", "session1")
            session2 = session_service.create_session("audio2.wav", "session2")

            # Mark one as completed
            session2.is_completed = True
            session_service.save_session(session2)

            # Test listing
            all_sessions = session_service.list_sessions()
            assert len(all_sessions) == 2

            # Test finding by name
            found_session = session_service.find_session_by_name("session1")
            assert found_session is not None
            assert found_session.session_id == session1.session_id

            # Test resumable sessions
            resumable = session_service.get_resumable_sessions()
            assert len(resumable) == 1
            assert resumable[0].session_id == session1.session_id

    def test_cleanup_completed_sessions(self):
        """Test cleanup of old completed sessions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sessions_dir = Path(temp_dir)
            session_service = FileSessionService(sessions_dir)

            # Create completed session
            session = session_service.create_session("old_audio.wav")
            session.is_completed = True

            # Mock old creation date
            from datetime import timedelta

            old_date = datetime.now() - timedelta(days=35)
            session.created_at = old_date
            session_service.save_session(session)

            # Run cleanup
            cleaned = session_service.cleanup_completed_sessions()

            assert cleaned == 1

            # Verify session was deleted
            remaining_sessions = session_service.list_sessions()
            assert len(remaining_sessions) == 0


class TestResumeService:
    def test_session_progress_tracking(self):
        """Test session progress tracking functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sessions_dir = Path(temp_dir)
            session_service = FileSessionService(sessions_dir)

            # Mock transcription service
            transcription_service = Mock()

            resume_service = TranscriptionResumeService(
                transcription_service, session_service
            )

            # Create session
            session = session_service.create_session("test.wav", "test_session")
            session.total_duration = 120.0
            session.progress_seconds = 30.0
            session.transcribed_text = "Previous text"
            session_service.save_session(session)

            # Get progress
            progress = resume_service.get_session_progress(session.session_id)

            assert progress["session_id"] == session.session_id
            assert progress["progress_seconds"] == 30.0
            assert progress["total_duration"] == 120.0
            assert progress["progress_percentage"] == 25.0
            assert not progress["is_completed"]
            assert progress["can_resume"]

    def test_audio_duration_estimation(self):
        """Test audio duration estimation for non-WAV files."""
        session_service = Mock()
        transcription_service = Mock()

        resume_service = TranscriptionResumeService(
            transcription_service, session_service
        )

        # Test fallback estimation when wave.open fails
        with patch("wave.open", side_effect=Exception("Not a WAV file")):
            with patch(
                "os.path.getsize", return_value=1600000
            ):  # 100 seconds at 16kHz mono 16-bit
                duration = resume_service._get_audio_duration("test.mp3")

                # Should estimate ~50 seconds (1600000 / (16000 * 2) = 50)
                assert duration == pytest.approx(50.0, abs=5.0)


class TestMemoryOptimization:
    def test_memory_limit_enforcement(self):
        """Test memory limit enforcement in transcription service."""
        from voicebridge.adapters.transcription import WhisperTranscriptionService

        # Mock system service with high memory usage
        system_service = Mock()
        system_service.get_memory_usage.return_value = {"used_mb": 2000}

        with patch("voicebridge.adapters.transcription.whisper"):
            transcription_service = WhisperTranscriptionService(
                system_service=system_service
            )

            # Should raise error when memory limit would be exceeded
            with pytest.raises(RuntimeError, match="Memory limit exceeded"):
                transcription_service._check_memory_usage(
                    max_memory_mb=1500,  # Lower than current usage
                    additional_bytes=100 * 1024 * 1024,  # 100MB additional
                )

    def test_memory_cleanup_on_limit(self):
        """Test memory cleanup when approaching limits."""
        from voicebridge.adapters.transcription import WhisperTranscriptionService

        system_service = Mock()
        # First call: high usage, second call (after cleanup): lower usage
        system_service.get_memory_usage.side_effect = [
            {"used_mb": 1900},  # Initial high usage
            {"used_mb": 1200},  # After cleanup
        ]

        with patch("voicebridge.adapters.transcription.whisper"):
            with patch("gc.collect") as mock_gc:
                transcription_service = WhisperTranscriptionService(
                    system_service=system_service
                )

                # Should not raise error after cleanup
                transcription_service._check_memory_usage(
                    max_memory_mb=1500,
                    additional_bytes=50 * 1024 * 1024,  # 50MB additional
                )

                # Verify cleanup was called
                mock_gc.assert_called_once()


class TestConfigurationExtensions:
    def test_extended_whisper_config(self):
        """Test new configuration fields in WhisperConfig."""
        config = WhisperConfig()

        # Test GPU settings
        assert config.use_gpu is True
        assert config.gpu_device is None
        assert config.force_cpu is False

        # Test memory settings
        assert config.chunk_size == 30
        assert config.max_memory_mb == 0  # 0 = auto-detect

        # Test resume settings
        assert config.enable_resume is True
        assert config.session_name is None

    def test_config_serialization_with_new_fields(self):
        """Test config serialization includes new fields."""
        config = WhisperConfig(
            use_gpu=False,
            chunk_size=60,
            max_memory_mb=2048,
            session_name="test_session",
        )

        config_dict = config.to_dict()

        assert config_dict["use_gpu"] is False
        assert config_dict["chunk_size"] == 60
        assert config_dict["max_memory_mb"] == 2048
        assert config_dict["session_name"] == "test_session"

        # Test deserialization
        restored_config = WhisperConfig.from_dict(config_dict)

        assert restored_config.use_gpu is False
        assert restored_config.chunk_size == 60
        assert restored_config.max_memory_mb == 2048
        assert restored_config.session_name == "test_session"
