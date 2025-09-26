#!/usr/bin/env python3
"""Unit tests for TTS adapters."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# numpy will be mocked, not imported directly

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from voicebridge.adapters.vibevoice_tts import VibeVoiceTTSAdapter
from voicebridge.domain.models import TTSConfig, TTSResult


class TestVibeVoiceTTSAdapter(unittest.TestCase):
    """Test VibeVoice TTS adapter."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the VibeVoice imports to avoid dependency issues
        self.vibevoice_mock = Mock()
        self.model_mock = Mock()
        self.processor_mock = Mock()
        self.streamer_mock = Mock()

        # Create patcher for vibevoice imports
        self.import_patcher = patch.multiple(
            "voicebridge.adapters.vibevoice_tts",
            VibeVoiceForConditionalGenerationInference=self.model_mock,
            VibeVoiceProcessor=self.processor_mock,
            AudioStreamer=self.streamer_mock,
            VIBEVOICE_AVAILABLE=True,
        )
        self.import_patcher.start()

        # Mock torch
        self.torch_patcher = patch("voicebridge.adapters.vibevoice_tts.torch")
        self.torch_mock = self.torch_patcher.start()
        self.torch_mock.cuda.is_available.return_value = False
        self.torch_mock.bfloat16 = "bfloat16"

        # Mock soundfile
        self.sf_patcher = patch("voicebridge.adapters.vibevoice_tts.sf")
        self.sf_mock = self.sf_patcher.start()

        # Mock librosa
        self.librosa_patcher = patch("voicebridge.adapters.vibevoice_tts.librosa")
        self.librosa_mock = self.librosa_patcher.start()

        # Mock numpy
        self.np_patcher = patch("voicebridge.adapters.vibevoice_tts.np")
        self.np_mock = self.np_patcher.start()
        # Mock numpy array creation and methods
        mock_array = Mock()
        mock_array.__len__ = Mock(return_value=1000)  # Add len method for the Mock
        mock_array.astype = Mock(return_value=mock_array)
        mock_array.flatten = Mock(return_value=mock_array)
        mock_array.shape = (1000,)  # Default 1D array shape
        mock_array.mean = Mock(return_value=0.0)
        mock_array.dtype = Mock()
        self.np_mock.array.return_value = mock_array
        self.np_mock.float32 = "float32"
        self.np_mock.int16 = "int16"

        # Mock numpy functions that return comparable values
        self.np_mock.max.return_value = 0.5  # Less than 1.0, so no normalization needed
        self.np_mock.abs.return_value = mock_array  # Return mock array for chaining

    def tearDown(self):
        """Clean up test fixtures."""
        self.import_patcher.stop()
        self.torch_patcher.stop()
        self.sf_patcher.stop()
        self.librosa_patcher.stop()
        self.np_patcher.stop()

    def test_adapter_initialization_success(self):
        """Test successful TTS adapter initialization."""
        adapter = VibeVoiceTTSAdapter()

        self.assertIsNone(adapter.model)
        self.assertIsNone(adapter.processor)
        self.assertIsNone(adapter.current_streamer)
        self.assertFalse(adapter.is_generating_flag)
        self.assertFalse(adapter.stop_requested)

    def test_adapter_initialization_no_vibevoice(self):
        """Test TTS adapter initialization when VibeVoice is not available."""
        with patch("voicebridge.adapters.vibevoice_tts.VIBEVOICE_AVAILABLE", False):
            with self.assertRaises(RuntimeError) as context:
                VibeVoiceTTSAdapter()

            self.assertIn("VibeVoice is not available", str(context.exception))

    @patch("voicebridge.adapters.vibevoice_tts.os.path.exists")
    def test_load_audio_success(self, mock_exists):
        """Test successful audio loading."""
        mock_exists.return_value = True
        mock_audio_array = Mock()
        mock_audio_array.__len__ = Mock(return_value=1000)
        mock_audio_array.astype.return_value = mock_audio_array
        mock_audio_array.flatten.return_value = mock_audio_array
        mock_audio_array.shape = (1000,)  # 1D array shape
        self.sf_mock.read.return_value = (mock_audio_array, 24000)

        adapter = VibeVoiceTTSAdapter()
        result = adapter._load_audio("test.wav")

        self.assertIsNotNone(result)
        self.sf_mock.read.assert_called_once_with("test.wav")

    @patch("voicebridge.adapters.vibevoice_tts.os.path.exists")
    def test_load_audio_stereo_to_mono(self, mock_exists):
        """Test audio loading converts stereo to mono."""
        mock_exists.return_value = True
        stereo_audio = Mock()
        # Create a mock shape that behaves like a tuple but is mockable
        mock_shape = Mock()
        mock_shape.__len__ = Mock(return_value=2)  # 2D array
        mock_shape.__gt__ = Mock(return_value=True)  # shape > 1 is True
        mock_shape.__getitem__ = Mock(side_effect=[3, 2])  # shape[0]=3, shape[1]=2
        stereo_audio.shape = mock_shape

        # Mock the complete chain: stereo_audio.mean(axis=1).astype(dtype).flatten()
        stereo_audio.mean.return_value = Mock()
        stereo_audio.mean.return_value.astype.return_value = Mock()
        stereo_audio.mean.return_value.astype.return_value.flatten.return_value = Mock()
        self.sf_mock.read.return_value = (stereo_audio, 24000)

        adapter = VibeVoiceTTSAdapter()
        result = adapter._load_audio("test_stereo.wav")

        # Just verify the method returns something (not None)
        self.assertIsNotNone(result)

    @patch("voicebridge.adapters.vibevoice_tts.os.path.exists")
    def test_load_audio_resampling(self, mock_exists):
        """Test audio loading with resampling."""
        mock_exists.return_value = True
        mock_audio = Mock()
        mock_audio.shape = (3,)
        mock_audio.flatten.return_value = mock_audio
        self.sf_mock.read.return_value = (mock_audio, 44100)  # Wrong sample rate
        resampled_audio = Mock()
        resampled_audio.shape = (2,)
        self.librosa_mock.resample.return_value = resampled_audio  # Resampled

        adapter = VibeVoiceTTSAdapter()
        _result = adapter._load_audio("test_44khz.wav")

        self.librosa_mock.resample.assert_called_once()
        # Check that librosa.resample was called with correct parameters
        call_args = self.librosa_mock.resample.call_args
        self.assertEqual(call_args[1]["orig_sr"], 44100)
        self.assertEqual(call_args[1]["target_sr"], 24000)

    def test_load_model_first_time(self):
        """Test loading model for the first time."""
        self.processor_mock.from_pretrained.return_value = Mock()
        self.model_mock.from_pretrained.return_value = Mock()

        adapter = VibeVoiceTTSAdapter()
        adapter._load_model("test/model/path")

        self.processor_mock.from_pretrained.assert_called_once_with("test/model/path")
        self.model_mock.from_pretrained.assert_called_once()
        self.assertEqual(adapter._model_path, "test/model/path")

    def test_load_model_same_path_twice(self):
        """Test loading the same model path twice (should use cached)."""
        self.processor_mock.from_pretrained.return_value = Mock()
        mock_model = Mock()
        self.model_mock.from_pretrained.return_value = mock_model

        adapter = VibeVoiceTTSAdapter()
        adapter._load_model("test/model/path")
        adapter._load_model("test/model/path")  # Second call

        # Should only be called once due to caching
        self.processor_mock.from_pretrained.assert_called_once()
        self.model_mock.from_pretrained.assert_called_once()

    def test_load_model_flash_attention_fallback(self):
        """Test model loading falls back from flash_attention_2 to SDPA."""
        self.processor_mock.from_pretrained.return_value = Mock()

        # First call (flash_attention_2) fails, second call (SDPA) succeeds
        self.model_mock.from_pretrained.side_effect = [
            Exception("Flash attention failed"),
            Mock(),
        ]

        adapter = VibeVoiceTTSAdapter()
        adapter._load_model("test/model/path")

        # Should be called twice due to fallback
        self.assertEqual(self.model_mock.from_pretrained.call_count, 2)

        # Check that both attention implementations were tried
        calls = self.model_mock.from_pretrained.call_args_list
        self.assertEqual(calls[0][1]["attn_implementation"], "flash_attention_2")
        self.assertEqual(calls[1][1]["attn_implementation"], "sdpa")

    @patch("voicebridge.adapters.vibevoice_tts.os.path.exists")
    def test_generate_speech_success(self, mock_exists):
        """Test successful speech generation."""
        mock_exists.return_value = True

        # Mock audio loading
        mock_audio = Mock()
        mock_audio.shape = (3,)
        mock_audio.flatten.return_value = mock_audio
        self.sf_mock.read.return_value = (mock_audio, 24000)

        # Mock model components
        mock_processor = Mock()
        mock_model = Mock()
        mock_processor.return_value = {"input_ids": "fake_inputs"}
        # Create a mock tensor with proper shape handling
        mock_speech_tensor = Mock()

        # Create a shape that responds properly to len()
        class MockShape:
            def __len__(self):
                return 1

            def __gt__(self, other):
                return len(self) > other

        mock_shape = MockShape()
        mock_speech_tensor.shape = mock_shape

        # Mock audio processing methods
        mock_speech_tensor.squeeze = Mock(return_value=mock_speech_tensor)
        mock_speech_tensor.__truediv__ = Mock(return_value=mock_speech_tensor)
        mock_speech_tensor.__mul__ = Mock(return_value=mock_speech_tensor)
        mock_speech_tensor.astype = Mock(return_value=mock_speech_tensor)
        mock_speech_tensor.tobytes = Mock(return_value=b"fake_audio_bytes")

        # Mock the output that gets returned from the generate call
        mock_output = Mock()
        mock_tensor = Mock()
        mock_tensor.to.return_value.cpu.return_value.numpy.return_value.astype.return_value = mock_speech_tensor
        mock_output.speech_outputs = [mock_tensor]
        mock_model.generate.return_value = mock_output
        mock_model.set_ddpm_inference_steps = Mock()

        self.processor_mock.from_pretrained.return_value = mock_processor
        self.model_mock.from_pretrained.return_value = mock_model

        adapter = VibeVoiceTTSAdapter()

        config = TTSConfig()
        result = adapter.generate_speech("Hello world", ["test.wav"], config)

        self.assertIsInstance(result, TTSResult)
        self.assertEqual(result.text, "Hello world")
        self.assertEqual(result.sample_rate, 24000)
        self.assertIsNotNone(result.audio_data)
        self.assertFalse(result.streaming_mode)

    def test_generate_speech_empty_text(self):
        """Test speech generation with empty text."""
        adapter = VibeVoiceTTSAdapter()
        config = TTSConfig()

        with self.assertRaises(ValueError) as context:
            adapter.generate_speech("", ["test.wav"], config)

        self.assertIn("Text cannot be empty", str(context.exception))

    @patch("voicebridge.adapters.vibevoice_tts.os.path.exists")
    def test_generate_speech_missing_voice_sample(self, mock_exists):
        """Test speech generation with missing voice sample."""
        mock_exists.return_value = False

        adapter = VibeVoiceTTSAdapter()
        config = TTSConfig()

        with self.assertRaises(RuntimeError) as context:
            adapter.generate_speech("Hello", ["missing.wav"], config)

        self.assertIn("Failed to load voice samples", str(context.exception))

    def test_load_voice_samples_success(self):
        """Test successful voice sample loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create fake voice files
            voice_files = [
                "en-Alice_woman.wav",
                "en-Bob_man.wav",
                "fr-Sophie_woman.wav",
                "invalid.txt",  # Should be ignored
            ]

            for filename in voice_files:
                Path(temp_dir, filename).touch()

            adapter = VibeVoiceTTSAdapter()
            voices = adapter.load_voice_samples(temp_dir)

            # Should load 3 voice files (ignore .txt)
            self.assertEqual(len(voices), 3)

            # Check Alice voice
            alice = voices.get("en-Alice_woman")
            self.assertIsNotNone(alice)
            self.assertEqual(alice.name, "en-Alice_woman")
            self.assertEqual(alice.language, "en")
            self.assertEqual(alice.gender, "woman")
            self.assertEqual(alice.display_name, "en Alice woman")

    def test_load_voice_samples_nonexistent_directory(self):
        """Test loading voice samples from nonexistent directory."""
        adapter = VibeVoiceTTSAdapter()
        voices = adapter.load_voice_samples("/nonexistent/path")

        self.assertEqual(len(voices), 0)

    def test_stop_generation(self):
        """Test stopping generation."""
        adapter = VibeVoiceTTSAdapter()
        adapter.current_streamer = Mock()

        adapter.stop_generation()

        self.assertTrue(adapter.stop_requested)
        adapter.current_streamer.end.assert_called_once()

    def test_is_generating(self):
        """Test checking generation status."""
        adapter = VibeVoiceTTSAdapter()

        self.assertFalse(adapter.is_generating())

        adapter.is_generating_flag = True
        self.assertTrue(adapter.is_generating())

    @patch("voicebridge.adapters.vibevoice_tts.os.path.exists")
    @patch("voicebridge.adapters.vibevoice_tts.threading.Thread")
    def test_generate_speech_streaming(self, mock_thread, mock_exists):
        """Test streaming speech generation."""
        mock_exists.return_value = True

        # Mock audio loading
        mock_audio = Mock()
        mock_audio.shape = (3,)
        mock_audio.flatten.return_value = mock_audio
        self.sf_mock.read.return_value = (mock_audio, 24000)

        # Mock model components
        mock_processor = Mock()
        mock_model = Mock()
        mock_processor.return_value = {"input_ids": "fake_inputs"}
        self.processor_mock.from_pretrained.return_value = mock_processor
        self.model_mock.from_pretrained.return_value = mock_model

        # Mock streamer
        mock_streamer_instance = Mock()
        mock_audio_stream = [
            self.torch_mock.tensor([0.1, 0.2]),
            self.torch_mock.tensor([0.3, 0.4]),
        ]
        mock_streamer_instance.get_stream.return_value = iter(mock_audio_stream)
        self.streamer_mock.return_value = mock_streamer_instance

        # Mock torch tensor behavior
        for tensor in mock_audio_stream:
            # Mock the numpy array that will be returned
            mock_tensor_array = Mock()
            mock_tensor_array.shape = Mock()
            mock_tensor_array.shape.__len__ = Mock(return_value=1)  # 1D tensor
            # Mock numpy operations for audio processing
            mock_tensor_array.max = Mock(return_value=0.5)  # Max amplitude
            mock_tensor_array.__mul__ = Mock(return_value=mock_tensor_array)
            mock_tensor_array.astype = Mock(return_value=mock_tensor_array)
            mock_tensor_array.tobytes = Mock(return_value=b"fake_audio")

            # Set up the tensor chain: cpu().numpy()
            tensor.cpu.return_value.numpy.return_value = mock_tensor_array
            tensor.dtype = self.torch_mock.bfloat16
            tensor.float.return_value = tensor

        # Mock np.max and np.abs for the streaming code
        self.np_mock.max.return_value = 0.5
        self.np_mock.abs.return_value = mock_tensor_array

        adapter = VibeVoiceTTSAdapter()
        config = TTSConfig()

        # Generate streaming audio
        audio_chunks = list(
            adapter.generate_speech_streaming("Hello streaming", ["test.wav"], config)
        )

        self.assertEqual(len(audio_chunks), 2)
        for chunk in audio_chunks:
            self.assertIsInstance(chunk, bytes)


if __name__ == "__main__":
    unittest.main()
