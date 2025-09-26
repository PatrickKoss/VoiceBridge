#!/usr/bin/env python3
"""Unit tests for domain models."""

import sys
import unittest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from voicebridge.domain.models import (
    AudioDeviceInfo,
    OperationMode,
    PerformanceMetrics,
    PlatformType,
    SystemInfo,
    TranscriptionResult,
    TTSConfig,
    # TTS models
    TTSMode,
    TTSOutputMode,
    TTSResult,
    TTSStreamingMode,
    VoiceInfo,
    WhisperConfig,
)


class TestWhisperConfig(unittest.TestCase):
    """Test WhisperConfig model."""

    def test_default_config(self):
        """Test default configuration values."""
        config = WhisperConfig()

        self.assertEqual(config.model_name, "medium")
        self.assertIsNone(config.language)
        self.assertIsNone(config.initial_prompt)
        self.assertEqual(config.temperature, 0.0)
        self.assertEqual(config.mode, OperationMode.TOGGLE)
        self.assertEqual(config.key, "ctrl+f2")
        self.assertFalse(config.paste_stream)
        self.assertTrue(config.copy_final)
        self.assertFalse(config.debug)

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = WhisperConfig(
            model_name="small", language="en", temperature=0.1, debug=True
        )

        result = config.to_dict()

        self.assertEqual(result["model_name"], "small")
        self.assertEqual(result["language"], "en")
        self.assertEqual(result["temperature"], 0.1)
        self.assertEqual(result["mode"], "toggle")  # Enum converted to value
        self.assertTrue(result["debug"])

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "model_name": "large",
            "language": "fr",
            "temperature": 0.5,
            "mode": "push_to_talk",
            "debug": True,
        }

        config = WhisperConfig.from_dict(data)

        self.assertEqual(config.model_name, "large")
        self.assertEqual(config.language, "fr")
        self.assertEqual(config.temperature, 0.5)
        self.assertEqual(config.mode, OperationMode.PUSH_TO_TALK)
        self.assertTrue(config.debug)

    def test_config_roundtrip(self):
        """Test config -> dict -> config roundtrip."""
        original = WhisperConfig(
            model_name="tiny",
            language="es",
            initial_prompt="Hello",
            temperature=0.3,
            paste_stream=True,
            copy_final=False,
        )

        data = original.to_dict()
        restored = WhisperConfig.from_dict(data)

        self.assertEqual(original.model_name, restored.model_name)
        self.assertEqual(original.language, restored.language)
        self.assertEqual(original.initial_prompt, restored.initial_prompt)
        self.assertEqual(original.temperature, restored.temperature)
        self.assertEqual(original.paste_stream, restored.paste_stream)
        self.assertEqual(original.copy_final, restored.copy_final)


class TestTranscriptionResult(unittest.TestCase):
    """Test TranscriptionResult model."""

    def test_basic_result(self):
        """Test basic transcription result."""
        result = TranscriptionResult(text="Hello world")

        self.assertEqual(result.text, "Hello world")
        self.assertIsNone(result.confidence)
        self.assertIsNone(result.language)
        self.assertIsNone(result.duration)

    def test_full_result(self):
        """Test transcription result with all fields."""
        result = TranscriptionResult(
            text="Bonjour le monde", confidence=0.95, language="fr", duration=2.5
        )

        self.assertEqual(result.text, "Bonjour le monde")
        self.assertEqual(result.confidence, 0.95)
        self.assertEqual(result.language, "fr")
        self.assertEqual(result.duration, 2.5)


class TestAudioDeviceInfo(unittest.TestCase):
    """Test AudioDeviceInfo model."""

    def test_device_info(self):
        """Test audio device info."""
        device = AudioDeviceInfo(
            name="Built-in Microphone", device_id="mic-001", platform=PlatformType.MACOS
        )

        self.assertEqual(device.name, "Built-in Microphone")
        self.assertEqual(device.device_id, "mic-001")
        self.assertEqual(device.platform, PlatformType.MACOS)


class TestSystemInfo(unittest.TestCase):
    """Test SystemInfo model."""

    def test_system_info_creation(self):
        """Test system info creation."""
        info = SystemInfo(platform=PlatformType.LINUX)
        self.assertEqual(info.platform, PlatformType.LINUX)

    def test_current_system_info(self):
        """Test getting current system info."""
        info = SystemInfo.current()

        # Should detect one of the supported platforms
        self.assertIn(
            info.platform,
            [PlatformType.WINDOWS, PlatformType.MACOS, PlatformType.LINUX],
        )


class TestPerformanceMetrics(unittest.TestCase):
    """Test PerformanceMetrics model."""

    def test_basic_metrics(self):
        """Test basic performance metrics."""
        metrics = PerformanceMetrics(operation="transcription", duration=1.5)

        self.assertEqual(metrics.operation, "transcription")
        self.assertEqual(metrics.duration, 1.5)
        self.assertEqual(metrics.details, {})

    def test_metrics_with_details(self):
        """Test performance metrics with details."""
        details = {"text_length": 100, "model": "medium"}
        metrics = PerformanceMetrics(
            operation="transcription", duration=2.1, details=details
        )

        self.assertEqual(metrics.operation, "transcription")
        self.assertEqual(metrics.duration, 2.1)
        self.assertEqual(metrics.details, details)


class TestTTSConfig(unittest.TestCase):
    """Test TTSConfig model."""

    def test_default_tts_config(self):
        """Test default TTS configuration values."""
        config = TTSConfig()

        self.assertEqual(config.model_path, "aoi-ot/VibeVoice-7B")
        self.assertEqual(config.voice_samples_dir, "voices")
        self.assertEqual(config.default_voice, "en-Alice_woman")
        self.assertEqual(config.cfg_scale, 1.3)
        self.assertEqual(config.inference_steps, 10)
        self.assertEqual(config.tts_mode, TTSMode.CLIPBOARD)
        self.assertEqual(config.streaming_mode, TTSStreamingMode.NON_STREAMING)
        self.assertEqual(config.output_mode, TTSOutputMode.PLAY_AUDIO)
        self.assertEqual(config.sample_rate, 24000)
        self.assertTrue(config.auto_play)

    def test_tts_config_to_dict(self):
        """Test converting TTS config to dictionary."""
        config = TTSConfig(
            model_path="custom/model",
            default_voice="en-Bob_man",
            cfg_scale=1.5,
            streaming_mode=TTSStreamingMode.STREAMING,
            auto_play=False,
        )

        result = config.to_dict()

        self.assertEqual(result["model_path"], "custom/model")
        self.assertEqual(result["default_voice"], "en-Bob_man")
        self.assertEqual(result["cfg_scale"], 1.5)
        self.assertEqual(result["streaming_mode"], "streaming")
        self.assertFalse(result["auto_play"])

    def test_tts_config_from_dict(self):
        """Test creating TTS config from dictionary."""
        data = {
            "model_path": "test/model",
            "cfg_scale": 2.0,
            "tts_mode": "manual",
            "output_mode": "save",
            "use_gpu": False,
        }

        config = TTSConfig.from_dict(data)

        self.assertEqual(config.model_path, "test/model")
        self.assertEqual(config.cfg_scale, 2.0)
        self.assertEqual(config.tts_mode, TTSMode.MANUAL)
        self.assertEqual(config.output_mode, TTSOutputMode.SAVE_FILE)
        self.assertFalse(config.use_gpu)


class TestTTSResult(unittest.TestCase):
    """Test TTSResult model."""

    def test_basic_tts_result(self):
        """Test basic TTS result."""
        audio_data = b"fake_audio_data"
        result = TTSResult(
            audio_data=audio_data,
            sample_rate=24000,
            text="Hello world",
            generation_time=1.5,
        )

        self.assertEqual(result.audio_data, audio_data)
        self.assertEqual(result.sample_rate, 24000)
        self.assertEqual(result.text, "Hello world")
        self.assertEqual(result.generation_time, 1.5)
        self.assertIsNone(result.voice_used)
        self.assertFalse(result.streaming_mode)

    def test_full_tts_result(self):
        """Test TTS result with all fields."""
        audio_data = b"complete_audio_data"
        result = TTSResult(
            audio_data=audio_data,
            sample_rate=22050,
            text="Bonjour le monde",
            generation_time=2.3,
            voice_used="fr-Sophie_woman.wav",
            streaming_mode=True,
        )

        self.assertEqual(result.audio_data, audio_data)
        self.assertEqual(result.sample_rate, 22050)
        self.assertEqual(result.text, "Bonjour le monde")
        self.assertEqual(result.generation_time, 2.3)
        self.assertEqual(result.voice_used, "fr-Sophie_woman.wav")
        self.assertTrue(result.streaming_mode)


class TestVoiceInfo(unittest.TestCase):
    """Test VoiceInfo model."""

    def test_voice_info_creation(self):
        """Test voice info creation."""
        voice = VoiceInfo(
            name="en-Alice_woman",
            file_path="/path/to/voices/en-Alice_woman.wav",
            display_name="en Alice woman",
            language="en",
            gender="woman",
        )

        self.assertEqual(voice.name, "en-Alice_woman")
        self.assertEqual(voice.file_path, "/path/to/voices/en-Alice_woman.wav")
        self.assertEqual(voice.display_name, "en Alice woman")
        self.assertEqual(voice.language, "en")
        self.assertEqual(voice.gender, "woman")

    def test_voice_info_minimal(self):
        """Test voice info with minimal fields."""
        voice = VoiceInfo(name="test_voice", file_path="/path/to/test.wav")

        self.assertEqual(voice.name, "test_voice")
        self.assertEqual(voice.file_path, "/path/to/test.wav")
        self.assertIsNone(voice.display_name)
        self.assertIsNone(voice.language)
        self.assertIsNone(voice.gender)


class TestEnums(unittest.TestCase):
    """Test enum types."""

    def test_operation_mode_values(self):
        """Test OperationMode enum values."""
        self.assertEqual(OperationMode.TOGGLE.value, "toggle")
        self.assertEqual(OperationMode.PUSH_TO_TALK.value, "push_to_talk")

    def test_platform_type_values(self):
        """Test PlatformType enum values."""
        self.assertEqual(PlatformType.WINDOWS.value, "windows")
        self.assertEqual(PlatformType.MACOS.value, "darwin")
        self.assertEqual(PlatformType.LINUX.value, "linux")

    def test_tts_mode_values(self):
        """Test TTSMode enum values."""
        self.assertEqual(TTSMode.CLIPBOARD.value, "clipboard")
        self.assertEqual(TTSMode.MOUSE.value, "mouse")
        self.assertEqual(TTSMode.MANUAL.value, "manual")

    def test_tts_streaming_mode_values(self):
        """Test TTSStreamingMode enum values."""
        self.assertEqual(TTSStreamingMode.STREAMING.value, "streaming")
        self.assertEqual(TTSStreamingMode.NON_STREAMING.value, "non_streaming")

    def test_tts_output_mode_values(self):
        """Test TTSOutputMode enum values."""
        self.assertEqual(TTSOutputMode.PLAY_AUDIO.value, "play")
        self.assertEqual(TTSOutputMode.SAVE_FILE.value, "save")
        self.assertEqual(TTSOutputMode.BOTH.value, "both")


if __name__ == "__main__":
    unittest.main()
