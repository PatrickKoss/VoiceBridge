#!/usr/bin/env python3
"""Unit tests for audio recording adapter."""

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from voicebridge.adapters.audio import FFmpegAudioRecorder
from voicebridge.domain.models import AudioDeviceInfo, PlatformType, SystemInfo


class TestFFmpegAudioRecorder(unittest.TestCase):
    """Test FFmpegAudioRecorder adapter."""

    def setUp(self):
        """Set up test fixtures."""
        self.recorder = FFmpegAudioRecorder()

    @patch.object(SystemInfo, "current")
    def test_initialization(self, mock_current):
        """Test recorder initialization."""
        mock_current.return_value = SystemInfo(platform=PlatformType.LINUX)

        recorder = FFmpegAudioRecorder()
        self.assertIsInstance(recorder.system_info, SystemInfo)

    @patch("subprocess.run")
    @patch.object(
        FFmpegAudioRecorder, "_get_default_device", return_value="default_mic"
    )
    @patch("subprocess.Popen")
    def test_record_stream_basic(self, mock_popen, mock_default_device, mock_run):
        """Test basic audio recording stream."""
        # Mock ffmpeg process
        mock_proc = Mock()
        mock_proc.stdout.read.side_effect = [b"audio_data_1", b"audio_data_2", b""]
        mock_popen.return_value = mock_proc

        # Test recording
        stream = self.recorder.record_stream()

        # Should be able to start the stream (generator)
        self.assertIsNotNone(stream)

    def test_record_stream_no_device(self):
        """Test recording when no device is available."""
        with patch.object(self.recorder, "_get_default_device", return_value=None):
            with self.assertRaises(RuntimeError) as context:
                list(self.recorder.record_stream())

            self.assertIn("No audio input device found", str(context.exception))

    @patch("subprocess.run")
    def test_list_dshow_devices(self, mock_run):
        """Test listing DirectShow devices on Windows."""
        # Mock ffmpeg output for dshow devices
        mock_run.return_value.stderr = """
[dshow @ 0x123] DirectShow video devices (some may be both video and audio devices)
[dshow @ 0x123]  "Integrated Camera"
[dshow @ 0x123]     Alternative name "@device_pv_{123}"
[dshow @ 0x123] DirectShow audio devices
[dshow @ 0x123]  "Microphone (Realtek High Definition Audio)"
[dshow @ 0x123]     Alternative name "@device_cm_{456}"
[dshow @ 0x123]  "Stereo Mix (Realtek High Definition Audio)"
"""

        self.recorder.system_info.platform = PlatformType.WINDOWS
        devices = self.recorder._list_dshow_devices()

        self.assertEqual(len(devices), 2)
        self.assertEqual(devices[0].name, "Microphone (Realtek High Definition Audio)")
        self.assertEqual(devices[0].platform, PlatformType.WINDOWS)
        self.assertEqual(devices[1].name, "Stereo Mix (Realtek High Definition Audio)")

    @patch("subprocess.run")
    def test_list_macos_devices(self, mock_run):
        """Test listing macOS AVFoundation devices."""
        mock_run.return_value.stderr = """
[AVFoundation indev @ 0x123] AVFoundation video devices:
[AVFoundation indev @ 0x123] [0] FaceTime HD Camera
[AVFoundation indev @ 0x123] AVFoundation audio devices:
[AVFoundation indev @ 0x123] [0] Built-in Microphone
[AVFoundation indev @ 0x123] [1] External Microphone
"""

        self.recorder.system_info.platform = PlatformType.MACOS
        devices = self.recorder._list_macos_devices()

        self.assertEqual(len(devices), 2)
        self.assertEqual(devices[0].name, "Built-in Microphone")
        self.assertEqual(devices[0].device_id, "0")
        self.assertEqual(devices[0].platform, PlatformType.MACOS)

    @patch("subprocess.run")
    def test_list_linux_devices(self, mock_run):
        """Test listing Linux PulseAudio devices."""
        mock_run.return_value.stdout = """
0	alsa_input.pci-0000_00_1f.3.analog-stereo	module-alsa-card.c	s16le 2ch 44100Hz	SUSPENDED
1	alsa_input.usb-Blue_Microphones_Yeti_Stereo.analog-stereo	module-alsa-card.c	s16le 2ch 44100Hz	RUNNING
"""

        self.recorder.system_info.platform = PlatformType.LINUX
        devices = self.recorder._list_linux_devices()

        self.assertEqual(len(devices), 2)
        self.assertEqual(
            devices[0].device_id, "alsa_input.pci-0000_00_1f.3.analog-stereo"
        )
        self.assertEqual(devices[0].platform, PlatformType.LINUX)
        self.assertEqual(
            devices[1].device_id,
            "alsa_input.usb-Blue_Microphones_Yeti_Stereo.analog-stereo",
        )

    def test_build_ffmpeg_command_windows(self):
        """Test FFmpeg command building for Windows."""
        self.recorder.system_info.platform = PlatformType.WINDOWS
        cmd = self.recorder._build_ffmpeg_command("Microphone", 16000)

        expected = [
            "ffmpeg",
            "-f",
            "dshow",
            "-i",
            "audio=Microphone",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-f",
            "wav",
            "pipe:1",
        ]
        self.assertEqual(cmd, expected)

    def test_build_ffmpeg_command_macos(self):
        """Test FFmpeg command building for macOS."""
        self.recorder.system_info.platform = PlatformType.MACOS
        cmd = self.recorder._build_ffmpeg_command("0", 22050)

        expected = [
            "ffmpeg",
            "-f",
            "avfoundation",
            "-i",
            ":0",
            "-ar",
            "22050",
            "-ac",
            "1",
            "-f",
            "wav",
            "pipe:1",
        ]
        self.assertEqual(cmd, expected)

    def test_build_ffmpeg_command_linux(self):
        """Test FFmpeg command building for Linux."""
        self.recorder.system_info.platform = PlatformType.LINUX
        cmd = self.recorder._build_ffmpeg_command("default", 16000)

        expected = [
            "ffmpeg",
            "-f",
            "pulse",
            "-i",
            "default",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-f",
            "wav",
            "pipe:1",
        ]
        self.assertEqual(cmd, expected)

    def test_get_default_device(self):
        """Test getting default device."""
        mock_devices = [
            AudioDeviceInfo("Device 1", "dev1", PlatformType.LINUX),
            AudioDeviceInfo("Device 2", "dev2", PlatformType.LINUX),
        ]

        with patch.object(self.recorder, "list_devices", return_value=mock_devices):
            default = self.recorder._get_default_device()

        self.assertEqual(default, "dev1")

    def test_get_default_device_no_devices(self):
        """Test getting default device when no devices available."""
        with patch.object(self.recorder, "list_devices", return_value=[]):
            with patch("subprocess.run") as mock_subprocess:
                mock_subprocess.side_effect = subprocess.TimeoutExpired("ffmpeg", 3)
                with self.assertRaises(RuntimeError) as context:
                    self.recorder._get_default_device()

                # Should raise RuntimeError about PulseAudio timeout
                self.assertIn("Audio system timeout", str(context.exception))

    def test_stop_ffmpeg_gracefully(self):
        """Test graceful FFmpeg process termination."""
        mock_proc = Mock()
        mock_proc.poll.return_value = None  # Process is running
        mock_proc.wait.return_value = None

        self.recorder._stop_ffmpeg_gracefully(mock_proc)

        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once_with(timeout=2)

    def test_stop_ffmpeg_force_kill(self):
        """Test force killing FFmpeg process when termination fails."""
        mock_proc = Mock()
        mock_proc.poll.return_value = None  # Process is running
        mock_proc.wait.side_effect = [subprocess.TimeoutExpired("ffmpeg", 2), None]

        self.recorder._stop_ffmpeg_gracefully(mock_proc)

        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()
        self.assertEqual(mock_proc.wait.call_count, 2)

    def test_list_devices_error_handling(self):
        """Test error handling in device listing."""
        with patch("subprocess.run", side_effect=Exception("Command failed")):
            self.recorder.system_info.platform = PlatformType.WINDOWS
            devices = self.recorder._list_dshow_devices()

        self.assertEqual(devices, [])

    def test_list_devices_dispatching(self):
        """Test device listing dispatches to correct platform method."""
        with patch.object(
            self.recorder, "_list_dshow_devices", return_value=[]
        ) as mock_dshow:
            self.recorder.system_info.platform = PlatformType.WINDOWS
            self.recorder.list_devices()
            mock_dshow.assert_called_once()

        with patch.object(
            self.recorder, "_list_macos_devices", return_value=[]
        ) as mock_macos:
            self.recorder.system_info.platform = PlatformType.MACOS
            self.recorder.list_devices()
            mock_macos.assert_called_once()

        with patch.object(
            self.recorder, "_list_linux_devices", return_value=[]
        ) as mock_linux:
            self.recorder.system_info.platform = PlatformType.LINUX
            self.recorder.list_devices()
            mock_linux.assert_called_once()


if __name__ == "__main__":
    unittest.main()
