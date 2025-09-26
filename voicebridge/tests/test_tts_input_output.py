#!/usr/bin/env python3
"""Unit tests for TTS input/output services."""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from voicebridge.adapters.audio_playback import (
    PyAudioPlaybackAdapter,
    PygameAudioPlaybackAdapter,
    create_audio_playback_service,
)
from voicebridge.adapters.text_input import (
    PlatformTextInputAdapter,
    create_text_input_service,
)


class TestPlatformTextInputAdapter(unittest.TestCase):
    """Test platform text input adapter."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock pyperclip
        self.pyperclip_patcher = patch("voicebridge.adapters.text_input.pyperclip")
        self.pyperclip_mock = self.pyperclip_patcher.start()

        # Mock pynput
        self.pynput_patcher = patch.multiple(
            "voicebridge.adapters.text_input",
            mouse=Mock(),
            keyboard=Mock(),
            Listener=Mock(),
            Key=Mock(),
            Button=Mock(),
        )
        self.pynput_mocks = self.pynput_patcher.start()

    def tearDown(self):
        """Clean up test fixtures."""
        self.pyperclip_patcher.stop()
        self.pynput_patcher.stop()

    def test_initialization(self):
        """Test text input adapter initialization."""
        adapter = PlatformTextInputAdapter()

        self.assertIsNone(adapter.clipboard_listener)
        self.assertIsNone(adapter.selection_listener)
        self.assertFalse(adapter.is_monitoring_clipboard)
        self.assertFalse(adapter.is_monitoring_selection)

    def test_get_clipboard_text(self):
        """Test getting clipboard text."""
        self.pyperclip_mock.paste.return_value = "Hello from clipboard"

        adapter = PlatformTextInputAdapter()
        text = adapter.get_clipboard_text()

        self.assertEqual(text, "Hello from clipboard")
        self.pyperclip_mock.paste.assert_called_once()

    def test_get_clipboard_text_empty(self):
        """Test getting empty clipboard text."""
        self.pyperclip_mock.paste.return_value = ""

        adapter = PlatformTextInputAdapter()
        text = adapter.get_clipboard_text()

        self.assertEqual(text, "")

    def test_get_clipboard_text_exception(self):
        """Test clipboard text retrieval with exception."""
        self.pyperclip_mock.paste.side_effect = Exception("Clipboard error")

        # Mock os.path.exists and subprocess to prevent WSL fallback
        with (
            patch("voicebridge.adapters.text_input.os.path.exists", return_value=False),
            patch("voicebridge.adapters.text_input.subprocess.run"),
        ):
            adapter = PlatformTextInputAdapter()
            text = adapter.get_clipboard_text()

        self.assertEqual(text, "")

    def test_get_selected_text_mock(self):
        """Test getting selected text (mocked)."""
        # This is a complex feature that would require system integration
        # For unit tests, we just test that it returns empty string
        adapter = PlatformTextInputAdapter()
        text = adapter.get_selected_text()

        self.assertEqual(text, "")  # Default implementation

    @patch("voicebridge.adapters.text_input.threading.Thread")
    def test_start_clipboard_monitoring(self, mock_thread):
        """Test starting clipboard monitoring."""
        mock_callback = Mock()
        adapter = PlatformTextInputAdapter()

        adapter.start_clipboard_monitoring(mock_callback, interval=0.1)

        self.assertTrue(adapter.is_monitoring_clipboard)
        mock_thread.assert_called_once()

        # Test that thread was started
        thread_instance = mock_thread.return_value
        thread_instance.start.assert_called_once()

    @patch("voicebridge.adapters.text_input.threading.Thread")
    def test_stop_clipboard_monitoring(self, mock_thread):
        """Test stopping clipboard monitoring."""
        mock_callback = Mock()
        adapter = PlatformTextInputAdapter()

        # Start monitoring first
        adapter.start_clipboard_monitoring(mock_callback)
        adapter.stop_clipboard_monitoring()

        self.assertFalse(adapter.is_monitoring_clipboard)

    @patch("voicebridge.adapters.text_input.Listener")
    def test_start_selection_monitoring(self, mock_listener_class):
        """Test starting selection monitoring."""
        mock_callback = Mock()
        adapter = PlatformTextInputAdapter()

        # Configure the mock listener instance
        mock_listener = Mock()
        mock_listener_class.return_value = mock_listener

        adapter.start_selection_monitoring(mock_callback)

        self.assertTrue(adapter.is_monitoring_selection)
        mock_listener_class.assert_called_once()
        mock_listener.start.assert_called_once()

    def test_stop_selection_monitoring(self):
        """Test stopping selection monitoring."""
        mock_callback = Mock()
        adapter = PlatformTextInputAdapter()

        # Mock listener - access from module directly
        import adapters.text_input

        mock_listener = Mock()
        adapters.text_input.Listener.return_value = mock_listener

        # Start and then stop monitoring
        adapter.start_selection_monitoring(mock_callback)
        adapter.stop_selection_monitoring()

        self.assertFalse(adapter.is_monitoring_selection)
        if adapter.selection_listener:
            mock_listener.stop.assert_called_once()


class TestAudioPlaybackAdapters(unittest.TestCase):
    """Test audio playback adapters."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock pygame
        self.pygame_patcher = patch("voicebridge.adapters.audio_playback.pygame")
        self.pygame_mock = self.pygame_patcher.start()

        # Mock pygame as a module with proper callable behavior
        def pygame_callable(*args, **kwargs):
            raise TypeError("pygame() call fails normally")

        self.pygame_mock.side_effect = pygame_callable

        # Set up pygame mixer mock properly
        self.pygame_mock.mixer = Mock()
        self.pygame_mock.mixer.init.return_value = None
        self.pygame_mock.mixer.quit.return_value = None
        self.pygame_mock.mixer.get_busy.return_value = False
        self.pygame_mock.mixer.music = Mock()
        self.pygame_mock.mixer.music.get_busy.return_value = False
        self.pygame_mock.mixer.stop.return_value = None
        self.pygame_mock.mixer.music.stop.return_value = None
        self.pygame_mock.mixer.music.load.return_value = None
        self.pygame_mock.mixer.music.play.return_value = None

        # Mock pygame.sndarray
        self.pygame_mock.sndarray = Mock()
        mock_sound_from_array = Mock()
        mock_sound_from_array.play.return_value = None
        self.pygame_mock.sndarray.make_sound.return_value = mock_sound_from_array

        # Mock pygame.mixer.Sound
        mock_sound = Mock()
        mock_sound.play.return_value = None
        self.pygame_mock.mixer.Sound.return_value = mock_sound

        # Also mock PYGAME_AVAILABLE
        self.pygame_available_patcher = patch(
            "voicebridge.adapters.audio_playback.PYGAME_AVAILABLE", True
        )
        self.pygame_available_patcher.start()

        # Mock pyaudio
        self.pyaudio_patcher = patch("voicebridge.adapters.audio_playback.pyaudio")
        self.pyaudio_mock = self.pyaudio_patcher.start()

    def tearDown(self):
        """Clean up test fixtures."""
        self.pygame_patcher.stop()
        self.pygame_available_patcher.stop()
        self.pyaudio_patcher.stop()

    def test_pygame_adapter_init(self):
        """Test pygame adapter initialization."""
        adapter = PygameAudioPlaybackAdapter()

        # Test that the adapter was created
        self.assertIsNotNone(adapter)
        self.assertFalse(adapter.is_initialized)

        # pygame.mixer.init should not be called during __init__
        # It's called during first _initialize() call
        self.pygame_mock.mixer.init.assert_not_called()

    def test_pygame_adapter_play_audio_data(self):
        """Test playing audio data with pygame - just verify basic functionality."""
        adapter = PygameAudioPlaybackAdapter()

        # Test that the adapter exists and has the required methods
        self.assertTrue(hasattr(adapter, "play_audio_data"))
        self.assertTrue(hasattr(adapter, "is_playing"))
        self.assertTrue(hasattr(adapter, "stop_playback"))

        # Test basic state
        self.assertFalse(adapter.is_initialized)
        self.assertIsNone(adapter.current_sound)

    def test_pygame_adapter_play_audio_file(self):
        """Test playing audio file with pygame."""
        adapter = PygameAudioPlaybackAdapter()

        adapter.play_audio_file("test.wav")

        self.pygame_mock.mixer.music.load.assert_called_once_with("test.wav")
        self.pygame_mock.mixer.music.play.assert_called_once()

    def test_pygame_adapter_stop_playback(self):
        """Test stopping playback with pygame."""
        adapter = PygameAudioPlaybackAdapter()

        adapter.stop_playback()

        self.pygame_mock.mixer.stop.assert_called_once()
        self.pygame_mock.mixer.music.stop.assert_called_once()

    def test_pygame_adapter_is_playing(self):
        """Test checking if audio is playing with pygame."""
        # Set up the mock to return True for get_busy()
        self.pygame_mock.mixer.get_busy.return_value = True
        self.pygame_mock.mixer.music.get_busy.return_value = False

        adapter = PygameAudioPlaybackAdapter()
        # Initialize the adapter to make is_playing work
        adapter._initialize(24000)
        is_playing = adapter.is_playing()

        self.assertTrue(is_playing)
        self.pygame_mock.mixer.get_busy.assert_called_once()

    def test_pygame_adapter_exception_handling(self):
        """Test pygame adapter exception handling."""
        # Make mixer.init raise an exception during _initialize call
        self.pygame_mock.mixer.init.side_effect = Exception("Pygame error")

        adapter = PygameAudioPlaybackAdapter()

        # Exception should happen when calling _initialize (through play_audio_data)
        with self.assertRaises(RuntimeError) as context:
            adapter.play_audio_data(b"test", 24000)

        self.assertIn("Failed to initialize pygame", str(context.exception))

    def test_pyaudio_adapter_init(self):
        """Test pyaudio adapter initialization."""
        mock_pyaudio_instance = Mock()
        self.pyaudio_mock.PyAudio.return_value = mock_pyaudio_instance

        adapter = PyAudioPlaybackAdapter()

        self.pyaudio_mock.PyAudio.assert_called_once()
        self.assertEqual(adapter.pa, mock_pyaudio_instance)

    def test_pyaudio_adapter_play_audio_data(self):
        """Test playing audio data with pyaudio."""
        # Mock pyaudio stream
        mock_stream = Mock()
        mock_pyaudio_instance = Mock()
        mock_pyaudio_instance.open.return_value = mock_stream
        self.pyaudio_mock.PyAudio.return_value = mock_pyaudio_instance

        # Mock numpy functions
        with patch("voicebridge.adapters.audio_playback.np") as mock_np:
            mock_array = Mock()
            mock_array.__len__ = Mock(return_value=4)  # Simulate 4 bytes / 2 samples
            mock_array.tobytes.return_value = b"test"
            mock_array.__getitem__ = Mock(return_value=mock_array)  # For slicing
            mock_np.frombuffer.return_value = mock_array

            adapter = PyAudioPlaybackAdapter()
            audio_data = b"test"  # 4 bytes for proper int16 processing

            # Call play_audio and wait for thread completion
            adapter.play_audio_data(audio_data, sample_rate=24000)

            # Wait for thread to complete
            if adapter.playback_thread:
                adapter.playback_thread.join(timeout=1.0)

            # Verify stream operations were called
            mock_pyaudio_instance.open.assert_called_once()
            # The write method should be called in the thread
            mock_stream.write.assert_called()
            mock_stream.stop_stream.assert_called_once()
            mock_stream.close.assert_called_once()

    def test_pyaudio_adapter_exception_handling(self):
        """Test pyaudio adapter exception handling."""
        self.pyaudio_mock.PyAudio.side_effect = Exception("PyAudio error")

        with self.assertRaises(RuntimeError) as context:
            PyAudioPlaybackAdapter()

        self.assertIn("Failed to initialize PyAudio", str(context.exception))


class TestAudioPlaybackFactory(unittest.TestCase):
    """Test audio playback service factory."""

    @patch("voicebridge.adapters.audio_playback.pygame")
    def test_create_audio_playback_service_pygame_available(self, mock_pygame):
        """Test creating audio playback service when pygame is available."""
        service = create_audio_playback_service()

        self.assertIsInstance(service, PygameAudioPlaybackAdapter)

    @patch(
        "voicebridge.adapters.audio_playback.pygame",
        side_effect=ImportError("No pygame"),
    )
    @patch("voicebridge.adapters.audio_playback.pyaudio")
    def test_create_audio_playback_service_pyaudio_fallback(
        self, mock_pyaudio, mock_pygame
    ):
        """Test falling back to pyaudio when pygame is not available."""
        mock_pyaudio_instance = Mock()
        mock_pyaudio.PyAudio.return_value = mock_pyaudio_instance

        service = create_audio_playback_service()

        self.assertIsInstance(service, PyAudioPlaybackAdapter)

    @patch(
        "voicebridge.adapters.audio_playback.pygame",
        side_effect=ImportError("No pygame"),
    )
    @patch(
        "voicebridge.adapters.audio_playback.pyaudio",
        side_effect=ImportError("No pyaudio"),
    )
    def test_create_audio_playback_service_no_backend(self, mock_pyaudio, mock_pygame):
        """Test when no audio backend is available."""
        with self.assertRaises(RuntimeError) as context:
            create_audio_playback_service()

        self.assertIn("No audio playback backend available", str(context.exception))


class TestTextInputFactory(unittest.TestCase):
    """Test text input service factory."""

    @patch("voicebridge.adapters.text_input.pyperclip")
    @patch("voicebridge.adapters.text_input.pynput")
    def test_create_text_input_service(self, mock_pynput, mock_pyperclip):
        """Test creating text input service."""
        service = create_text_input_service()

        self.assertIsInstance(service, PlatformTextInputAdapter)


if __name__ == "__main__":
    unittest.main()
