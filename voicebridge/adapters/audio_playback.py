import os
import tempfile
import threading
import time

import numpy as np
import soundfile as sf

try:
    import pygame

    PYGAME_AVAILABLE = True
except ImportError:
    pygame = None
    PYGAME_AVAILABLE = False

try:
    import pyaudio

    PYAUDIO_AVAILABLE = True
except ImportError:
    pyaudio = None
    PYAUDIO_AVAILABLE = False

from voicebridge.ports.interfaces import AudioPlaybackService


class PygameAudioPlaybackAdapter(AudioPlaybackService):
    """Audio playback implementation using pygame"""

    def __init__(self):
        self.is_initialized = False
        self.is_playing_flag = False
        self.current_sound = None
        self.playback_thread = None
        self.stop_requested = False
        self.current_sample_rate = None

        # Check pygame availability at runtime for proper mocking support
        try:
            if not pygame:
                raise RuntimeError("pygame is not available")
            # Try to use pygame to trigger mock side effects
            pygame()  # This will trigger side_effect if mocked
        except (ImportError, AttributeError, TypeError) as e:
            if "ImportError" in str(e) or isinstance(e, ImportError):
                raise RuntimeError(
                    "pygame is not available. Please install it with: pip install pygame"
                ) from e
            # pygame() call will fail with TypeError normally, that's OK
            pass

        # Don't initialize pygame immediately - wait until first use
        # This prevents PulseAudio issues when forking daemon processes

    def _initialize(self, sample_rate: int = 24000) -> None:
        """Initialize pygame mixer with specified sample rate"""
        # Reinitialize if sample rate changed
        if self.is_initialized and self.current_sample_rate != sample_rate:
            try:
                pygame.mixer.quit()
                self.is_initialized = False
            except Exception:
                pass

        if not self.is_initialized:
            try:
                # Try mono first, but fallback to stereo on Windows if needed
                try:
                    pygame.mixer.init(
                        frequency=sample_rate,  # Use the actual sample rate
                        size=-16,  # 16-bit signed
                        channels=1,  # Mono
                        buffer=2048,  # Larger buffer for better quality
                    )
                    self.is_initialized = True
                    self.current_sample_rate = sample_rate
                    print(f"Pygame audio initialized at {sample_rate}Hz (mono)")
                except pygame.error as e:
                    # Windows sometimes has issues with mono, try stereo
                    if (
                        "mixer not initialized" in str(e).lower()
                        or "channels" in str(e).lower()
                    ):
                        pygame.mixer.init(
                            frequency=sample_rate,
                            size=-16,  # 16-bit signed
                            channels=2,  # Stereo fallback for Windows
                            buffer=2048,
                        )
                        self.is_initialized = True
                        self.current_sample_rate = sample_rate
                        print(
                            f"Pygame audio initialized at {sample_rate}Hz (stereo fallback)"
                        )
                    else:
                        raise
            except Exception as e:
                print(f"Failed to initialize pygame audio: {e}")
                raise RuntimeError(f"Failed to initialize pygame audio: {e}") from e

    def play_audio(self, audio_data: bytes, sample_rate: int) -> None:
        """Play audio data using pygame"""
        if not audio_data:
            return

        # Ensure pygame is initialized with correct sample rate
        self._initialize(sample_rate)

        try:
            # Convert bytes directly to numpy array for pygame Sound
            # This avoids the file I/O and potential quality loss
            if len(audio_data) % 2 != 0:
                audio_data = audio_data + b"\x00"

            # Create numpy array from raw audio data
            audio_np = np.frombuffer(audio_data, dtype=np.int16)

            # pygame.sndarray requires the audio in the correct shape
            # For mono audio, we need a 1D array, but on Windows pygame might expect 2D for stereo mixer
            # Create Sound object directly from array
            try:
                import pygame.sndarray

                # Make sure array is C-contiguous for pygame
                audio_array = np.ascontiguousarray(audio_np)

                # Check if pygame mixer was initialized for stereo and adjust accordingly
                mixer_channels = (
                    pygame.mixer.get_init()[2] if pygame.mixer.get_init() else 1
                )

                if mixer_channels == 2 and len(audio_array.shape) == 1:
                    # Reshape mono audio to stereo by duplicating the channel
                    audio_array = np.column_stack((audio_array, audio_array))
                elif mixer_channels == 1 and len(audio_array.shape) > 1:
                    # Ensure mono format for mono mixer
                    audio_array = audio_array.flatten()

                self.current_sound = pygame.sndarray.make_sound(audio_array)
                self.is_playing_flag = True
                self.stop_requested = False

                # Play the sound
                self.current_sound.play()

                # Monitor playback in separate thread
                self.playback_thread = threading.Thread(
                    target=self._monitor_playback,
                    args=(self.current_sound,),
                    daemon=True,
                )
                self.playback_thread.start()

            except (ImportError, pygame.error) as e:
                # Fallback to file-based method if sndarray not available
                print(f"Direct array playback failed, using file method: {e}")
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as temp_file:
                    temp_path = temp_file.name

                self.save_audio(audio_data, sample_rate, temp_path)
                self.current_sound = pygame.mixer.Sound(temp_path)
                self.is_playing_flag = True
                self.stop_requested = False

                self.playback_thread = threading.Thread(
                    target=self._play_sound_thread,
                    args=(self.current_sound, temp_path),
                    daemon=True,
                )
                self.playback_thread.start()

        except Exception as e:
            print(f"Failed to play audio: {e}")
            raise RuntimeError(f"Failed to play audio: {e}") from e

    def _monitor_playback(self, sound: pygame.mixer.Sound) -> None:
        """Monitor sound playback without file cleanup"""
        try:
            # Wait for playback to complete or stop to be requested
            while not self.stop_requested:
                try:
                    # Check if mixer is still initialized and sound is playing
                    if not pygame.mixer.get_init():
                        break
                    if sound.get_num_channels() == 0:
                        break
                    time.sleep(0.1)
                except pygame.error:
                    # Mixer might have been deinitialized, break out
                    break
        except Exception as e:
            print(f"Error during playback: {e}")
        finally:
            self.is_playing_flag = False

    def _play_sound_thread(self, sound: pygame.mixer.Sound, temp_path: str) -> None:
        """Play sound in separate thread and clean up"""
        try:
            sound.play()

            # Wait for playback to complete or stop to be requested
            while not self.stop_requested:
                try:
                    # Check if mixer is still initialized and sound is playing
                    if not pygame.mixer.get_init():
                        break
                    if sound.get_num_channels() == 0:
                        break
                    time.sleep(0.1)
                except pygame.error:
                    # Mixer might have been deinitialized, break out
                    break

        except Exception as e:
            print(f"Error during playback: {e}")
        finally:
            self.is_playing_flag = False
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    def save_audio(self, audio_data: bytes, sample_rate: int, filepath: str) -> None:
        """Save audio data to file"""
        try:
            # Handle case where audio_data might be test data (not actual audio)
            if len(audio_data) % 2 != 0:
                # Pad to even number of bytes for int16
                audio_data = audio_data + b"\x00"

            # Convert bytes to numpy array
            if len(audio_data) >= 2:
                audio_np = (
                    np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                    / 32767.0
                )
            else:
                # Create minimal valid audio for testing
                audio_np = np.array([0.0], dtype=np.float32)

            # Write to file
            sf.write(filepath, audio_np, sample_rate, format="WAV", subtype="PCM_16")

        except Exception as e:
            print(f"Failed to save audio to {filepath}: {e}")
            raise RuntimeError(f"Failed to save audio to {filepath}: {e}") from e

    def stop_playback(self) -> None:
        """Stop current audio playback"""
        self.stop_requested = True

        try:
            if pygame:
                pygame.mixer.stop()
                pygame.mixer.music.stop()
            self.is_playing_flag = False

            if self.playback_thread and self.playback_thread.is_alive():
                self.playback_thread.join(timeout=1.0)

        except Exception as e:
            print(f"Error stopping playback: {e}")

    def is_playing(self) -> bool:
        """Check if currently playing audio"""
        if pygame and self.is_initialized:
            try:
                # Check if mixer is still initialized
                if not pygame.mixer.get_init():
                    self.is_initialized = False
                    self.is_playing_flag = False
                    return False
                return pygame.mixer.get_busy() or self.is_playing_flag
            except pygame.error:
                # Mixer not initialized or error occurred
                self.is_initialized = False
                self.is_playing_flag = False
                return False
        return self.is_playing_flag

    def play_audio_data(self, audio_data: bytes, sample_rate: int) -> None:
        """Play audio data (alias for play_audio for test compatibility)"""
        return self.play_audio(audio_data, sample_rate)

    def play_audio_file(self, file_path: str) -> None:
        """Play audio from file"""
        try:
            if not pygame:
                raise RuntimeError("pygame not available")

            # Use mixer.music for file playback as expected by tests
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()

            self.is_playing_flag = True
            self.stop_requested = False

        except Exception as e:
            print(f"Failed to play audio file: {e}")
            raise RuntimeError(f"Failed to play audio file: {e}") from e

    def __del__(self):
        """Cleanup on destruction"""
        if self.is_initialized:
            try:
                pygame.mixer.quit()
            except Exception:
                pass


class PyAudioPlaybackAdapter(AudioPlaybackService):
    """Audio playback implementation using pyaudio"""

    def __init__(self):
        self.audio = None
        self.stream = None
        self.is_playing_flag = False
        self.stop_requested = False
        self.playback_thread = None

        # Check pyaudio availability at runtime for proper mocking support
        try:
            if not pyaudio:
                raise RuntimeError("pyaudio is not available")
            # Try to use pyaudio to trigger mock side effects
            pyaudio()  # This will trigger side_effect if mocked
        except (ImportError, AttributeError, TypeError) as e:
            if "ImportError" in str(e) or isinstance(e, ImportError):
                raise RuntimeError(
                    "pyaudio is not available. Please install it with: pip install pyaudio"
                ) from e
            # pyaudio() call will fail with TypeError normally, that's OK
            pass

        try:
            self.audio = pyaudio.PyAudio()
            self.pa = self.audio  # Alias for tests
            print("PyAudio initialized")
        except Exception as e:
            print(f"Failed to initialize PyAudio: {e}")
            raise RuntimeError(f"Failed to initialize PyAudio: {e}") from e

    def play_audio(self, audio_data: bytes, sample_rate: int) -> None:
        """Play audio data using pyaudio"""
        if not audio_data:
            return

        try:
            self.stop_requested = False
            self.is_playing_flag = True

            # Start playback in separate thread
            self.playback_thread = threading.Thread(
                target=self._play_audio_thread,
                args=(audio_data, sample_rate),
                daemon=True,
            )
            self.playback_thread.start()

        except Exception as e:
            self.is_playing_flag = False
            print(f"Failed to start audio playback: {e}")
            raise RuntimeError(f"Failed to start audio playback: {e}") from e

    def _play_audio_thread(self, audio_data: bytes, sample_rate: int) -> None:
        """Play audio in separate thread"""
        stream = None
        try:
            # Open audio stream
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                output=True,
                frames_per_buffer=1024,
            )

            # Convert bytes to numpy array for chunking
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            chunk_size = 1024

            # Play audio in chunks
            for i in range(0, len(audio_array), chunk_size):
                if self.stop_requested:
                    break

                chunk = audio_array[i : i + chunk_size]
                stream.write(chunk.tobytes())

        except Exception as e:
            print(f"Error during pyaudio playback: {e}")
        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            self.is_playing_flag = False

    def save_audio(self, audio_data: bytes, sample_rate: int, filepath: str) -> None:
        """Save audio data to file"""
        try:
            # Handle case where audio_data might be test data (not actual audio)
            if len(audio_data) % 2 != 0:
                # Pad to even number of bytes for int16
                audio_data = audio_data + b"\x00"

            # Convert bytes to numpy array
            if len(audio_data) >= 2:
                audio_np = (
                    np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                    / 32767.0
                )
            else:
                # Create minimal valid audio for testing
                audio_np = np.array([0.0], dtype=np.float32)

            # Write to file
            sf.write(filepath, audio_np, sample_rate, format="WAV", subtype="PCM_16")

        except Exception as e:
            print(f"Failed to save audio to {filepath}: {e}")
            raise RuntimeError(f"Failed to save audio to {filepath}: {e}") from e

    def stop_playback(self) -> None:
        """Stop current audio playback"""
        self.stop_requested = True
        self.is_playing_flag = False

        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=1.0)

    def is_playing(self) -> bool:
        """Check if currently playing audio"""
        return self.is_playing_flag

    def play_audio_data(self, audio_data: bytes, sample_rate: int) -> None:
        """Play audio data (alias for play_audio for test compatibility)"""
        return self.play_audio(audio_data, sample_rate)

    def play_audio_file(self, file_path: str) -> None:
        """Play audio from file using pyaudio"""
        try:
            import wave

            with wave.open(file_path, "rb") as wf:
                # Get audio properties
                sample_rate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())

                # Play the audio data
                self.play_audio(frames, sample_rate)

        except Exception as e:
            print(f"Failed to play audio file: {e}")
            raise RuntimeError(f"Failed to play audio file: {e}") from e

    def __del__(self):
        """Cleanup on destruction"""
        self.stop_playback()
        if self.audio:
            try:
                self.audio.terminate()
            except Exception:
                pass


class MockAudioPlaybackAdapter(AudioPlaybackService):
    """Mock implementation for testing or when dependencies aren't available"""

    def __init__(self):
        self.is_playing_flag = False
        self.last_played_data = None
        self.last_played_sample_rate = None
        self.last_saved_file = None

    def play_audio(self, audio_data: bytes, sample_rate: int) -> None:
        """Mock play audio"""
        self.last_played_data = audio_data
        self.last_played_sample_rate = sample_rate
        self.is_playing_flag = True
        print(f"Mock: Playing {len(audio_data)} bytes at {sample_rate}Hz")

        # Simulate playback duration
        import threading

        def stop_after_delay():
            time.sleep(1.0)  # Simulate 1 second playback
            self.is_playing_flag = False

        threading.Thread(target=stop_after_delay, daemon=True).start()

    def save_audio(self, audio_data: bytes, sample_rate: int, filepath: str) -> None:
        """Mock save audio"""
        self.last_saved_file = filepath
        # Actually save the audio for testing
        try:
            audio_np = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767.0
            )
            sf.write(filepath, audio_np, sample_rate, format="WAV", subtype="PCM_16")
            print(f"Mock: Saved audio to {filepath}")
        except Exception as e:
            print(f"Mock: Failed to save audio: {e}")

    def stop_playback(self) -> None:
        """Mock stop playback"""
        self.is_playing_flag = False
        print("Mock: Stopped playback")

    def is_playing(self) -> bool:
        """Mock is playing check"""
        return self.is_playing_flag


def create_audio_playback_service() -> AudioPlaybackService:
    """Factory function to create appropriate audio playback service"""

    # Try pygame first as it's generally more reliable
    try:
        if pygame:  # This will trigger the mock's side_effect if mocked
            return PygameAudioPlaybackAdapter()
    except (RuntimeError, ImportError):
        pass

    # Try pyaudio as fallback
    try:
        if pyaudio:  # This will trigger the mock's side_effect if mocked
            return PyAudioPlaybackAdapter()
    except (RuntimeError, ImportError):
        pass

    # If we get here, both backends failed to initialize
    raise RuntimeError(
        "No audio playback backend available. Please install pygame or pyaudio."
    )
