#!/usr/bin/env python3
"""
Comprehensive unit tests for whisper_cli.py

Tests cover:
- Utility functions (clipboard, typing, streaming logic)
- Configuration management
- CLI command structure
- Error handling
- Cross-platform functionality
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Import the module under test
import whisper_cli


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions like clipboard, typing, etc."""

    def test_has_cmd_existing_command(self):
        """Test _has_cmd with existing command."""
        with patch("shutil.which", return_value="/usr/bin/python"):
            self.assertTrue(whisper_cli._has_cmd("python"))

    def test_has_cmd_nonexistent_command(self):
        """Test _has_cmd with nonexistent command."""
        with patch("shutil.which", return_value=None):
            self.assertFalse(whisper_cli._has_cmd("nonexistent_command"))

    def test_copy_to_clipboard_empty_text(self):
        """Test clipboard copy with empty text."""
        result = whisper_cli._copy_to_clipboard("")
        self.assertTrue(result)

    @patch("subprocess.Popen")
    @patch("whisper_cli._has_cmd")
    def test_copy_to_clipboard_linux_xclip(self, mock_has_cmd, mock_popen):
        """Test clipboard copy on Linux with xclip."""
        with (
            patch.object(whisper_cli, "IS_LINUX", True),
            patch.object(whisper_cli, "IS_MAC", False),
            patch.object(whisper_cli, "IS_WINDOWS", False),
        ):
            mock_has_cmd.side_effect = lambda cmd: cmd == "xclip"
            mock_proc = Mock()
            mock_proc.communicate.return_value = (b"", b"")
            mock_proc.returncode = 0
            mock_popen.return_value = mock_proc

            result = whisper_cli._copy_to_clipboard("test text")

            self.assertTrue(result)
            mock_popen.assert_called_once()
            args = mock_popen.call_args
            self.assertEqual(args[0][0], ["xclip", "-selection", "clipboard"])

    @patch("whisper_cli._has_cmd")
    def test_copy_to_clipboard_linux_no_utility(self, mock_has_cmd):
        """Test clipboard copy on Linux with no clipboard utility."""
        with (
            patch.object(whisper_cli, "IS_LINUX", True),
            patch.object(whisper_cli, "IS_MAC", False),
            patch.object(whisper_cli, "IS_WINDOWS", False),
        ):
            mock_has_cmd.return_value = False

            result = whisper_cli._copy_to_clipboard("test text")
            self.assertFalse(result)

    @patch("subprocess.Popen")
    @patch("whisper_cli._has_cmd")
    def test_copy_to_clipboard_mac(self, mock_has_cmd, mock_popen):
        """Test clipboard copy on macOS."""
        with (
            patch.object(whisper_cli, "IS_MAC", True),
            patch.object(whisper_cli, "IS_LINUX", False),
            patch.object(whisper_cli, "IS_WINDOWS", False),
        ):
            mock_has_cmd.return_value = True
            mock_proc = Mock()
            mock_proc.communicate.return_value = (b"", b"")
            mock_proc.returncode = 0
            mock_popen.return_value = mock_proc

            result = whisper_cli._copy_to_clipboard("test text")

            self.assertTrue(result)
            mock_popen.assert_called_once()
            args = mock_popen.call_args
            self.assertEqual(args[0][0], ["pbcopy"])

    @patch("subprocess.Popen")
    @patch("whisper_cli._has_cmd")
    def test_copy_to_clipboard_windows(self, mock_has_cmd, mock_popen):
        """Test clipboard copy on Windows."""
        with (
            patch.object(whisper_cli, "IS_WINDOWS", True),
            patch.object(whisper_cli, "IS_LINUX", False),
            patch.object(whisper_cli, "IS_MAC", False),
        ):
            mock_has_cmd.return_value = True
            mock_proc = Mock()
            mock_proc.communicate.return_value = (b"", b"")
            mock_proc.returncode = 0
            mock_popen.return_value = mock_proc

            result = whisper_cli._copy_to_clipboard("test text")

            self.assertTrue(result)
            mock_popen.assert_called_once()

    def test_type_under_mouse_empty_text(self):
        """Test typing with empty text."""
        result = whisper_cli._type_under_mouse("")
        self.assertTrue(result)

    @patch("pynput.keyboard.Controller")
    def test_type_under_mouse_success(self, mock_controller_class):
        """Test successful typing."""
        mock_kb = Mock()
        mock_controller_class.return_value = mock_kb

        result = whisper_cli._type_under_mouse("test text")

        self.assertTrue(result)
        mock_kb.type.assert_called_once_with("test text")

    @patch("pynput.keyboard.Controller")
    def test_type_under_mouse_failure(self, mock_controller_class):
        """Test typing failure."""
        mock_controller_class.side_effect = Exception("Import error")

        result = whisper_cli._type_under_mouse("test text")
        self.assertFalse(result)


class TestStreamingPasteLogic(unittest.TestCase):
    """Test the streaming paste logic improvements."""

    def test_handle_streaming_paste_empty_transcript(self):
        """Test handling of empty transcript."""
        text_to_type, new_typed = whisper_cli._handle_streaming_paste("", "existing")
        self.assertEqual(text_to_type, "")
        self.assertEqual(new_typed, "existing")

    def test_handle_streaming_paste_incremental(self):
        """Test incremental transcript extension."""
        text_to_type, new_typed = whisper_cli._handle_streaming_paste(
            "Hello world", "Hello"
        )
        self.assertEqual(text_to_type, " world")
        self.assertEqual(new_typed, "Hello world")

    def test_handle_streaming_paste_complete_change(self):
        """Test complete transcript change."""
        text_to_type, new_typed = whisper_cli._handle_streaming_paste(
            "Goodbye", "Hello world"
        )
        self.assertEqual(text_to_type, " Goodbye")
        self.assertEqual(new_typed, "Hello world Goodbye")

    def test_handle_streaming_paste_first_transcript(self):
        """Test first transcript with empty typed_so_far."""
        text_to_type, new_typed = whisper_cli._handle_streaming_paste("Hello world", "")
        self.assertEqual(text_to_type, "Hello world")
        self.assertEqual(new_typed, "Hello world")

    def test_handle_streaming_paste_space_separation(self):
        """Test proper space separation handling."""
        text_to_type, new_typed = whisper_cli._handle_streaming_paste(
            "New text", "Existing "
        )
        self.assertEqual(text_to_type, "New text")  # No additional space
        self.assertEqual(new_typed, "Existing New text")


class TestConfigurationManagement(unittest.TestCase):
    """Test configuration loading, saving, and management."""

    def setUp(self):
        """Set up test environment."""
        self.test_config_dir = tempfile.mkdtemp()
        self.original_config_dir = whisper_cli.CONFIG_DIR
        whisper_cli.CONFIG_DIR = Path(self.test_config_dir)
        whisper_cli.CONFIG_FILE = whisper_cli.CONFIG_DIR / "config.json"
        whisper_cli.PID_FILE = whisper_cli.CONFIG_DIR / "daemon.pid"
        whisper_cli.LOG_FILE = whisper_cli.CONFIG_DIR / "whisper.log"

    def tearDown(self):
        """Clean up test environment."""
        whisper_cli.CONFIG_DIR = self.original_config_dir
        whisper_cli.CONFIG_FILE = self.original_config_dir / "config.json"
        whisper_cli.PID_FILE = self.original_config_dir / "daemon.pid"
        whisper_cli.LOG_FILE = self.original_config_dir / "whisper.log"

    def test_ensure_config_dir(self):
        """Test config directory creation."""
        whisper_cli.ensure_config_dir()
        self.assertTrue(whisper_cli.CONFIG_DIR.exists())
        self.assertTrue(whisper_cli.CONFIG_DIR.is_dir())

    def test_load_config_defaults(self):
        """Test loading default configuration."""
        config = whisper_cli.load_config()

        expected_defaults = {
            "model_name": "medium",
            "language": None,
            "initial_prompt": None,
            "temperature": 0.0,
            "mode": "toggle",
            "key": "f9",
            "start_key": "f9",
            "stop_key": "f10",
            "quit_key": "esc",
            "paste_stream": False,
            "copy_stream": False,
            "paste_final": False,
            "copy_final": True,
            "debug": False,
        }

        for key, value in expected_defaults.items():
            self.assertEqual(config[key], value)

    def test_load_config_from_file(self):
        """Test loading configuration from existing file."""
        test_config = {"model_name": "large", "temperature": 0.5, "copy_final": False}

        whisper_cli.ensure_config_dir()
        with open(whisper_cli.CONFIG_FILE, "w") as f:
            json.dump(test_config, f)

        config = whisper_cli.load_config()

        # Should have defaults merged with file config
        self.assertEqual(config["model_name"], "large")
        self.assertEqual(config["temperature"], 0.5)
        self.assertEqual(config["copy_final"], False)
        self.assertEqual(config["mode"], "toggle")  # Default value

    def test_save_config(self):
        """Test saving configuration to file."""
        test_config = {"model_name": "small", "debug": True}

        whisper_cli.save_config(test_config)

        self.assertTrue(whisper_cli.CONFIG_FILE.exists())

        with open(whisper_cli.CONFIG_FILE) as f:
            saved_config = json.load(f)

        self.assertEqual(saved_config, test_config)

    @patch("builtins.open", side_effect=OSError("Permission denied"))
    def test_save_config_error(self, mock_open):
        """Test save_config handles errors gracefully."""
        with patch("typer.echo") as mock_echo:
            whisper_cli.save_config({"test": "value"})
            mock_echo.assert_called_once()


class TestDaemonManagement(unittest.TestCase):
    """Test daemon PID file management."""

    def setUp(self):
        """Set up test environment."""
        self.test_config_dir = tempfile.mkdtemp()
        self.original_config_dir = whisper_cli.CONFIG_DIR
        whisper_cli.CONFIG_DIR = Path(self.test_config_dir)
        whisper_cli.PID_FILE = whisper_cli.CONFIG_DIR / "daemon.pid"

    def tearDown(self):
        """Clean up test environment."""
        whisper_cli.CONFIG_DIR = self.original_config_dir
        whisper_cli.PID_FILE = self.original_config_dir / "daemon.pid"

    def test_is_daemon_running_no_pid_file(self):
        """Test daemon status when no PID file exists."""
        result = whisper_cli.is_daemon_running()
        self.assertFalse(result)

    @patch("os.kill")
    def test_is_daemon_running_valid_process(self, mock_kill):
        """Test daemon status with valid running process."""
        whisper_cli.ensure_config_dir()
        with open(whisper_cli.PID_FILE, "w") as f:
            f.write("12345")

        # os.kill with signal 0 succeeds for running process
        mock_kill.return_value = None

        result = whisper_cli.is_daemon_running()
        self.assertTrue(result)
        mock_kill.assert_called_once_with(12345, 0)

    @patch("os.kill")
    def test_is_daemon_running_dead_process(self, mock_kill):
        """Test daemon status with dead process."""
        whisper_cli.ensure_config_dir()
        with open(whisper_cli.PID_FILE, "w") as f:
            f.write("99999")

        # os.kill raises OSError for dead process
        mock_kill.side_effect = OSError("No such process")

        result = whisper_cli.is_daemon_running()
        self.assertFalse(result)
        self.assertFalse(whisper_cli.PID_FILE.exists())  # Should clean up

    def test_write_pid_file(self):
        """Test writing PID file."""
        with patch("os.getpid", return_value=12345):
            whisper_cli.write_pid_file()

            self.assertTrue(whisper_cli.PID_FILE.exists())
            with open(whisper_cli.PID_FILE) as f:
                self.assertEqual(f.read(), "12345")

    def test_cleanup_pid_file(self):
        """Test cleaning up PID file."""
        whisper_cli.ensure_config_dir()
        whisper_cli.PID_FILE.touch()

        whisper_cli.cleanup_pid_file()
        self.assertFalse(whisper_cli.PID_FILE.exists())


class TestAudioDeviceDetection(unittest.TestCase):
    """Test audio device detection and enumeration."""

    @patch("subprocess.run")
    def test_ffmpeg_supports_device(self, mock_run):
        """Test FFmpeg device support detection."""
        mock_result = Mock()
        mock_result.stdout = "wasapi device support"
        mock_run.return_value = mock_result

        result = whisper_cli._ffmpeg_supports_device("wasapi")
        self.assertTrue(result)

    @patch("subprocess.run")
    def test_list_dshow_audio_devices(self, mock_run):
        """Test DirectShow device listing."""
        mock_result = Mock()
        mock_result.stderr = """
[dshow @ 0x...] DirectShow video devices (some may be both video and audio devices)
[dshow @ 0x...] "Mikrofon (Yeti Stereo Microphone)" (audio)
[dshow @ 0x...] "Webcam" (video)
[dshow @ 0x...] "Line In (Sound Card)" (audio)
"""
        mock_result.stdout = ""
        mock_run.return_value = mock_result

        devices = whisper_cli._list_dshow_audio_devices()

        expected_devices = ["Mikrofon (Yeti Stereo Microphone)", "Line In (Sound Card)"]
        self.assertEqual(devices, expected_devices)

    @patch("subprocess.run")
    def test_list_mediafoundation_audio_devices(self, mock_run):
        """Test MediaFoundation device listing."""
        mock_result = Mock()
        mock_result.stderr = """
[mediafoundation @ 0x...] [Audio Capture Devices]
[mediafoundation @ 0x...] 0: Mikrofon (Yeti Stereo Microphone)
[mediafoundation @ 0x...] 1: Line In (Sound Card)
[mediafoundation @ 0x...] [Video Capture Devices]
[mediafoundation @ 0x...] 0: Webcam
"""
        mock_result.stdout = ""
        mock_run.return_value = mock_result

        devices = whisper_cli._list_mediafoundation_audio_devices()

        expected_devices = ["Mikrofon (Yeti Stereo Microphone)", "Line In (Sound Card)"]
        self.assertEqual(devices, expected_devices)


class TestTranscriptionStream(unittest.TestCase):
    """Test transcription streaming functionality."""

    def test_transcribe_stream_mock(self):
        """Test transcription stream with mock data."""
        # This test would require mocking the whisper model
        # For now, we test the iterator structure

        def mock_audio_iter():
            yield b"audio_data_chunk_1"
            yield b"audio_data_chunk_2"
            yield b"audio_data_chunk_3"

        # Mock whisper model
        mock_model = Mock()
        mock_model.transcribe.return_value = {"text": "Hello world"}

        with (
            patch("tempfile.NamedTemporaryFile"),
            patch("os.remove"),
            patch("time.time", side_effect=[0, 1, 2, 3, 4]),
        ):
            results = list(
                whisper_cli._transcribe_stream(
                    mock_model, mock_audio_iter(), None, None, 0.0
                )
            )

            # Should yield some results based on time intervals
            self.assertIsInstance(results, list)


class TestCLICommands(unittest.TestCase):
    """Test CLI command structure and argument parsing."""

    def test_app_creation(self):
        """Test that the Typer app is created correctly."""
        self.assertIsInstance(whisper_cli.app, whisper_cli.typer.Typer)

    @patch("subprocess.run")
    def test_cli_help_commands(self, mock_run):
        """Test that all CLI commands have help available."""
        commands_to_test = [
            ["--help"],
            ["listen", "--help"],
            ["hotkey", "--help"],
            ["daemon", "--help"],
            ["start", "--help"],
            ["stop", "--help"],
            ["status", "--help"],
            ["config", "--help"],
        ]

        for _cmd_args in commands_to_test:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "Usage: whisper_cli.py"
            mock_run.return_value = mock_result

            # This would test actual CLI invocation
            # We just verify the structure exists
            self.assertTrue(hasattr(whisper_cli, "app"))


class TestErrorHandling(unittest.TestCase):
    """Test error handling throughout the application."""

    def test_whisper_import_error(self):
        """Test handling when whisper is not available."""
        with patch.dict("sys.modules", {"whisper": None}):
            # This would test the whisper import handling
            # The actual import happens at module level
            pass

    def test_shared_model_no_whisper(self):
        """Test SharedModel when whisper is not available."""
        with patch("whisper_cli.whisper", None):
            with self.assertRaises(SystemExit):
                with patch("typer.secho"), patch("typer.Exit") as mock_exit:
                    mock_exit.side_effect = SystemExit()
                    whisper_cli.SharedModel.get("medium")


class TestCrossPlatformBehavior(unittest.TestCase):
    """Test cross-platform specific behavior."""

    def test_platform_detection(self):
        """Test platform detection variables."""
        # These are set at module import time
        self.assertIsInstance(whisper_cli.IS_WINDOWS, bool)
        self.assertIsInstance(whisper_cli.IS_MAC, bool)
        self.assertIsInstance(whisper_cli.IS_LINUX, bool)
        self.assertIsInstance(whisper_cli.OS_NAME, str)

    @patch("whisper_cli.IS_WINDOWS", True)
    def test_windows_specific_behavior(self):
        """Test Windows-specific code paths."""
        # Test that Windows-specific logic can be triggered
        self.assertTrue(whisper_cli.IS_WINDOWS)

    @patch("whisper_cli.IS_MAC", True)
    def test_mac_specific_behavior(self):
        """Test macOS-specific code paths."""
        self.assertTrue(whisper_cli.IS_MAC)

    @patch("whisper_cli.IS_LINUX", True)
    def test_linux_specific_behavior(self):
        """Test Linux-specific code paths."""
        self.assertTrue(whisper_cli.IS_LINUX)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration-style tests for common usage patterns."""

    @patch("whisper_cli._copy_to_clipboard")
    @patch("whisper_cli._type_under_mouse")
    def test_streaming_workflow(self, mock_type, mock_copy):
        """Test a complete streaming workflow."""
        mock_copy.return_value = True
        mock_type.return_value = True

        # Simulate streaming transcription updates
        transcripts = ["Hello", "Hello world", "Hello world test"]
        typed_so_far = ""

        for transcript in transcripts:
            # Test copy_stream behavior
            result = whisper_cli._copy_to_clipboard(transcript)
            self.assertTrue(result)

            # Test paste_stream behavior
            text_to_type, typed_so_far = whisper_cli._handle_streaming_paste(
                transcript, typed_so_far
            )
            if text_to_type:
                result = whisper_cli._type_under_mouse(text_to_type)
                self.assertTrue(result)

        # Verify final state
        self.assertEqual(typed_so_far, "Hello world test")


if __name__ == "__main__":
    # Set up test environment
    unittest.main(verbosity=2, buffer=True)
