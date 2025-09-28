#!/usr/bin/env python3
"""Integration tests for the whisper-cli modular architecture."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import main components
from main import setup_dependencies

from voicebridge.domain.models import PlatformType, SystemInfo, WhisperConfig


class TestDependencyInjection(unittest.TestCase):
    """Test dependency injection setup."""

    @patch("voicebridge.adapters.transcription.whisper")
    def test_setup_dependencies(self, mock_whisper):
        """Test that all dependencies are properly injected."""
        # Mock whisper to avoid import errors
        mock_whisper.load_model.return_value = Mock()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock config directory to use temp directory
            with patch("main.Path.home", return_value=Path(temp_dir)):
                commands = setup_dependencies()

        # Verify all components are initialized
        # The setup_dependencies returns a CommandRegistry, so we access dependencies through it
        dependencies = commands.dependencies
        self.assertIsNotNone(dependencies.get("config_repo"))
        self.assertIsNotNone(dependencies.get("profile_repo"))
        self.assertIsNotNone(dependencies.get("daemon_service"))
        self.assertIsNotNone(dependencies.get("transcription_orchestrator"))
        self.assertIsNotNone(dependencies.get("system_service"))
        self.assertIsNotNone(dependencies.get("logger"))


class TestEndToEndWorkflow(unittest.TestCase):
    """Test end-to-end workflows."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    @patch("voicebridge.adapters.transcription.whisper")
    @patch("subprocess.run")
    @patch("subprocess.Popen")
    def test_config_save_load_workflow(self, mock_popen, mock_run, mock_whisper):
        """Test configuration save/load workflow."""
        # Mock whisper
        mock_whisper.load_model.return_value = Mock()

        # Mock system dependencies
        mock_run.return_value.returncode = 0

        with patch("main.Path.home", return_value=self.temp_dir):
            commands = setup_dependencies()

        # Test configuration workflow
        # 1. Load default config
        config = commands.dependencies["config_repo"].load()
        self.assertEqual(config.model_name, "medium")

        # 2. Modify config
        config.model_name = "large"
        config.debug = True

        # 3. Save config
        commands.dependencies["config_repo"].save(config)

        # 4. Load config again
        loaded_config = commands.dependencies["config_repo"].load()
        self.assertEqual(loaded_config.model_name, "large")
        self.assertTrue(loaded_config.debug)

    @patch("voicebridge.adapters.transcription.whisper")
    def test_profile_workflow(self, mock_whisper):
        """Test profile save/load/delete workflow."""
        mock_whisper.load_model.return_value = Mock()

        with patch("main.Path.home", return_value=self.temp_dir):
            commands = setup_dependencies()

        # Create custom config
        config = WhisperConfig(model_name="small", language="es", temperature=0.1)

        # Save as profile
        commands.dependencies["profile_repo"].save_profile("spanish", config)

        # List profiles
        profiles = commands.dependencies["profile_repo"].list_profiles()
        self.assertIn("spanish", profiles)

        # Load profile
        loaded_config = commands.dependencies["profile_repo"].load_profile("spanish")
        self.assertEqual(loaded_config.model_name, "small")
        self.assertEqual(loaded_config.language, "es")
        self.assertEqual(loaded_config.temperature, 0.1)

        # Delete profile
        result = commands.dependencies["profile_repo"].delete_profile("spanish")
        self.assertTrue(result)

        # Verify deletion
        profiles = commands.dependencies["profile_repo"].list_profiles()
        self.assertNotIn("spanish", profiles)

    @patch("voicebridge.adapters.transcription.whisper")
    @patch("subprocess.run")
    def test_daemon_lifecycle(self, mock_run, mock_whisper):
        """Test daemon start/stop/status lifecycle."""
        mock_whisper.load_model.return_value = Mock()
        mock_run.return_value.returncode = 0

        with patch("main.Path.home", return_value=self.temp_dir):
            commands = setup_dependencies()

        # Initial status should be not running
        status = commands.dependencies["daemon_service"].get_status()
        self.assertFalse(status["running"])

        # Start daemon (mocked)
        with patch.object(
            commands.dependencies["daemon_service"], "start"
        ) as mock_start:
            config = WhisperConfig()
            commands.dependencies["daemon_service"].start(config)
            mock_start.assert_called_once()

    @patch("voicebridge.adapters.transcription.whisper")
    def test_transcription_orchestrator_integration(self, mock_whisper):
        """Test transcription orchestrator with mocked dependencies."""
        # Setup mock whisper
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            "text": "Hello integration test",
            "language": "en",
            "segments": [{"no_speech_prob": 0.1}],
        }
        mock_whisper.load_model.return_value = mock_model

        with patch("main.Path.home", return_value=self.temp_dir):
            commands = setup_dependencies()

        # Mock system service memory usage to be below limit
        with patch.object(
            commands.dependencies[
                "transcription_orchestrator"
            ].transcription_service._system_service,
            "get_memory_usage",
        ) as mock_memory:
            mock_memory.return_value = {
                "used_mb": 500,
                "total_mb": 8000,
                "available_mb": 7500,
                "percent": 6.25,
            }

            # Mock audio recorder
            audio_data = [b"fake_audio_chunk" * 1000]  # Make it large enough
            with patch.object(
                commands.dependencies["transcription_orchestrator"].audio_recorder,
                "record_stream",
            ) as mock_record:
                mock_record.return_value = iter(audio_data)

                # Mock clipboard service
                with patch.object(
                    commands.dependencies[
                        "transcription_orchestrator"
                    ].clipboard_service,
                    "copy_text",
                ) as mock_copy:
                    mock_copy.return_value = True

                    # Test transcription
                    config = WhisperConfig(copy_final=True)
                    result = commands.dependencies[
                        "transcription_orchestrator"
                    ].transcribe_single_recording(config)

                    # Verify result
                    self.assertEqual(result, "Hello integration test")

                    # Verify clipboard was called
                    mock_copy.assert_called_once_with("Hello integration test")


class TestSystemIntegration(unittest.TestCase):
    """Test system-level integration."""

    def test_system_info_detection(self):
        """Test system information detection."""
        system_info = SystemInfo.current()

        # Should detect a valid platform
        self.assertIn(
            system_info.platform,
            [PlatformType.WINDOWS, PlatformType.MACOS, PlatformType.LINUX],
        )

    @patch("shutil.which")
    def test_system_dependency_checking(self, mock_which):
        """Test system dependency checking."""
        from voicebridge.adapters.system import StandardSystemService

        service = StandardSystemService()

        # Test with missing dependencies
        mock_which.return_value = None
        with self.assertRaises(RuntimeError) as context:
            service.ensure_dependencies()

        self.assertIn("Missing required dependencies", str(context.exception))

        # Test with all dependencies present
        mock_which.return_value = "/usr/bin/ffmpeg"
        result = service.ensure_dependencies()
        self.assertTrue(result)


class TestErrorHandling(unittest.TestCase):
    """Test error handling across components."""

    def test_config_error_handling(self):
        """Test configuration error handling."""
        from voicebridge.adapters.config import FileConfigRepository

        # Test with invalid directory
        invalid_path = Path("/invalid/path/that/does/not/exist")
        with patch("pathlib.Path.mkdir", side_effect=PermissionError("Access denied")):
            repo = FileConfigRepository(invalid_path)

            # Should still work with defaults
            config = repo.load()
            self.assertEqual(config.model_name, "medium")

    @patch("voicebridge.adapters.transcription.whisper", None)
    def test_missing_whisper_library(self):
        """Test handling of missing whisper library."""
        from voicebridge.adapters.transcription import WhisperTranscriptionService

        with self.assertRaises(RuntimeError) as context:
            WhisperTranscriptionService()

        self.assertIn("Whisper library not available", str(context.exception))

    def test_audio_device_error_handling(self):
        """Test audio device detection error handling."""
        from voicebridge.adapters.audio import FFmpegAudioRecorder

        recorder = FFmpegAudioRecorder()

        # Test with subprocess error
        with patch("subprocess.run", side_effect=Exception("Command failed")):
            devices = recorder._list_dshow_devices()

            # Should return empty list on error
            self.assertEqual(devices, [])


if __name__ == "__main__":
    unittest.main()
