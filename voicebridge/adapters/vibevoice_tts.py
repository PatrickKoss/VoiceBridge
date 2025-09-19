import os
import threading
import time
import traceback
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vibevoice.modular.streamer import AudioStreamer

import numpy as np
import soundfile as sf
import torch
from transformers.utils import logging

# Import librosa for audio resampling
try:
    import librosa
except ImportError:
    librosa = None

from voicebridge.domain.models import TTSConfig, TTSResult, VoiceInfo
from voicebridge.ports.interfaces import TTSService

# Import VibeVoice components
try:
    from vibevoice.modular.modeling_vibevoice_inference import (
        VibeVoiceForConditionalGenerationInference,
    )
    from vibevoice.modular.streamer import AudioStreamer
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    VIBEVOICE_AVAILABLE = True
except ImportError:
    # Define placeholder classes for testing
    VibeVoiceForConditionalGenerationInference = None
    VibeVoiceProcessor = None
    AudioStreamer = None
    VIBEVOICE_AVAILABLE = False


logging.set_verbosity_error()
logger = logging.get_logger(__name__)


class VibeVoiceTTSAdapter(TTSService):
    """VibeVoice TTS implementation using the VibeVoice model"""

    def __init__(self):
        self.model = None
        self.processor = None
        self.current_streamer = None
        self.is_generating_flag = False
        self.stop_requested = False
        self._model_path = None

        if not VIBEVOICE_AVAILABLE:
            raise RuntimeError(
                "VibeVoice is not available. Please install the required dependencies."
            )

    def _load_model(self, model_path: str) -> None:
        """Lazy load the VibeVoice model"""
        if self.model is None or self._model_path != model_path:
            logger.info(f"Loading VibeVoice model from {model_path}")

            try:
                self.processor = VibeVoiceProcessor.from_pretrained(model_path)

                # Try flash_attention_2 first
                try:
                    self.model = (
                        VibeVoiceForConditionalGenerationInference.from_pretrained(
                            model_path,
                            torch_dtype=torch.bfloat16,
                            device_map="cuda" if torch.cuda.is_available() else "cpu",
                            attn_implementation="flash_attention_2",
                        )
                    )
                    logger.info("Loaded model with flash_attention_2")
                except Exception as e:
                    logger.warning(f"Flash attention failed, falling back to SDPA: {e}")
                    self.model = (
                        VibeVoiceForConditionalGenerationInference.from_pretrained(
                            model_path,
                            torch_dtype=torch.bfloat16,
                            device_map="cuda" if torch.cuda.is_available() else "cpu",
                            attn_implementation="sdpa",
                        )
                    )
                    logger.info("Loaded model with SDPA")

                self.model.eval()
                self._model_path = model_path

                if hasattr(self.model.model, "language_model"):
                    attention_impl = getattr(
                        self.model.model.language_model.config,
                        "_attn_implementation",
                        "unknown",
                    )
                    logger.info(f"Language model attention: {attention_impl}")

            except Exception as e:
                logger.error(f"Failed to load VibeVoice model: {e}")
                raise RuntimeError(f"Failed to load VibeVoice model: {e}") from e

    def _load_audio(self, audio_path: str) -> np.ndarray:
        """Load audio file and return as numpy array"""
        try:
            wav, sr = sf.read(audio_path)
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)  # Convert to mono
            if sr != 24000:
                # Simple resampling (in production, use librosa.resample)
                if librosa is not None:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=24000)
                else:
                    # Fallback: simple linear interpolation (not as good as librosa)
                    from scipy.signal import resample

                    target_length = int(len(wav) * 24000 / sr)
                    wav = resample(wav, target_length)
            return wav.astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to load audio {audio_path}: {e}")
            raise RuntimeError(f"Failed to load audio {audio_path}: {e}") from e

    def generate_speech(
        self, text: str, voice_samples: list[str], config: TTSConfig
    ) -> TTSResult:
        """Generate speech from text (non-streaming)"""
        if not text.strip():
            raise ValueError("Text cannot be empty")

        start_time = time.time()
        self._load_model(config.model_path)

        # Load voice samples
        try:
            voice_data = []
            for voice_path in voice_samples:
                if not os.path.exists(voice_path):
                    raise FileNotFoundError(f"Voice sample not found: {voice_path}")
                voice_data.append(self._load_audio(voice_path))
        except Exception as e:
            logger.error(f"Failed to load voice samples: {e}")
            raise RuntimeError(f"Failed to load voice samples: {e}") from e

        # Format text for single speaker, handling line breaks properly
        # Replace line breaks with spaces to keep all text on one line
        # This prevents parsing errors in the VibeVoice processor
        cleaned_text = text.replace("\n", " ").replace("\r", " ")
        # Remove multiple spaces
        cleaned_text = " ".join(cleaned_text.split())
        formatted_text = f"Speaker 1: {cleaned_text}"

        try:
            # Process inputs
            inputs = self.processor(
                text=[formatted_text],
                voice_samples=[voice_data],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )

            # Set inference steps
            self.model.set_ddpm_inference_steps(num_steps=config.inference_steps)

            # Generate audio
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=config.cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={"do_sample": False},
                verbose=False,
            )

            generation_time = time.time() - start_time

            if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
                # Convert to numpy and normalize
                # First convert BFloat16 to Float32 in torch, then to numpy
                audio_tensor = outputs.speech_outputs[0].to(torch.float32).cpu()
                audio_np = audio_tensor.numpy().astype(np.float32)
                if len(audio_np.shape) > 1:
                    audio_np = audio_np.squeeze()

                # Normalize audio to prevent clipping
                if np.max(np.abs(audio_np)) > 1.0:
                    audio_np = audio_np / np.max(np.abs(audio_np))

                # Convert to bytes (16-bit PCM)
                audio_16bit = (audio_np * 32767).astype(np.int16)
                audio_bytes = audio_16bit.tobytes()

                voice_used = voice_samples[0] if voice_samples else None
                voice_name = os.path.basename(voice_used) if voice_used else None

                return TTSResult(
                    audio_data=audio_bytes,
                    sample_rate=config.sample_rate,
                    text=text,
                    generation_time=generation_time,
                    voice_used=voice_name,
                    streaming_mode=False,
                )
            else:
                raise RuntimeError("Model failed to generate audio")

        except Exception as e:
            logger.error(f"Speech generation failed: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Speech generation failed: {e}") from e

    def generate_speech_streaming(
        self, text: str, voice_samples: list[str], config: TTSConfig
    ) -> Iterator[bytes]:
        """Generate speech with streaming output"""
        if not text.strip():
            raise ValueError("Text cannot be empty")

        self._load_model(config.model_path)
        self.is_generating_flag = True
        self.stop_requested = False

        try:
            # Load voice samples
            voice_data = []
            for voice_path in voice_samples:
                if not os.path.exists(voice_path):
                    raise FileNotFoundError(f"Voice sample not found: {voice_path}")
                voice_data.append(self._load_audio(voice_path))

            # Format text for single speaker, handling line breaks properly
            cleaned_text = text.replace("\n", " ").replace("\r", " ")
            cleaned_text = " ".join(cleaned_text.split())
            formatted_text = f"Speaker 1: {cleaned_text}"

            # Process inputs
            inputs = self.processor(
                text=[formatted_text],
                voice_samples=[voice_data],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )

            # Create streamer
            audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
            self.current_streamer = audio_streamer

            # Start generation in thread
            generation_thread = threading.Thread(
                target=self._generate_with_streamer,
                args=(inputs, config.cfg_scale, audio_streamer, config.inference_steps),
            )
            generation_thread.start()

            # Allow generation to start
            time.sleep(0.5)

            # Yield audio chunks
            try:
                audio_stream = audio_streamer.get_stream(0)
                for audio_chunk in audio_stream:
                    if self.stop_requested:
                        break

                    if torch.is_tensor(audio_chunk):
                        # Convert to numpy
                        if audio_chunk.dtype == torch.bfloat16:
                            audio_chunk = audio_chunk.float()
                        audio_np = audio_chunk.cpu().numpy().astype(np.float32)

                        if len(audio_np.shape) > 1:
                            audio_np = audio_np.squeeze()

                        # Normalize and convert to bytes
                        if np.max(np.abs(audio_np)) > 1.0:
                            audio_np = audio_np / np.max(np.abs(audio_np))

                        audio_16bit = (audio_np * 32767).astype(np.int16)
                        yield audio_16bit.tobytes()

            except Exception as e:
                logger.error(f"Streaming error: {e}")

            finally:
                # Clean up
                if audio_streamer:
                    try:
                        audio_streamer.end()
                    except Exception:
                        pass

                generation_thread.join(timeout=5.0)

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise RuntimeError(f"Streaming generation failed: {e}") from e

        finally:
            self.is_generating_flag = False
            self.current_streamer = None

    def _generate_with_streamer(
        self,
        inputs,
        cfg_scale: float,
        audio_streamer: "AudioStreamer",
        inference_steps: int,
    ) -> None:
        """Helper method to run generation with streamer in a separate thread"""
        try:
            if self.stop_requested:
                audio_streamer.end()
                return

            self.model.set_ddpm_inference_steps(num_steps=inference_steps)

            def check_stop_generation():
                return self.stop_requested

            _outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={"do_sample": False},
                audio_streamer=audio_streamer,
                stop_check_fn=check_stop_generation,
                verbose=False,
                refresh_negative=True,
            )

        except Exception as e:
            logger.error(f"Error in generation thread: {e}")
            logger.error(traceback.format_exc())
        finally:
            try:
                audio_streamer.end()
            except Exception:
                pass

    def load_voice_samples(self, voices_dir: str) -> dict[str, VoiceInfo]:
        """Load available voice samples from directory"""
        voices = {}
        voices_path = Path(voices_dir)

        if not voices_path.exists():
            logger.warning(f"Voices directory not found: {voices_dir}")
            return voices

        # Supported audio formats
        supported_formats = [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"]

        try:
            for audio_file in voices_path.iterdir():
                if (
                    audio_file.is_file()
                    and audio_file.suffix.lower() in supported_formats
                ):
                    name = audio_file.stem

                    # Parse display name and metadata from filename
                    display_name = name.replace("_", " ").replace("-", " ")
                    language = None
                    gender = None

                    # Try to extract metadata from filename (e.g., "en-Alice_woman")
                    if "-" in name:
                        parts = name.split("-")
                        if len(parts) >= 2:
                            language = parts[0]
                            rest = parts[1]
                            if "_" in rest:
                                speaker_parts = rest.split("_")
                                if len(speaker_parts) >= 2:
                                    gender = speaker_parts[1].lower()

                    voices[name] = VoiceInfo(
                        name=name,
                        file_path=str(audio_file),
                        display_name=display_name,
                        language=language,
                        gender=gender,
                    )

            logger.info(f"Loaded {len(voices)} voice samples from {voices_dir}")

        except Exception as e:
            logger.error(f"Failed to load voice samples: {e}")

        return voices

    def stop_generation(self) -> None:
        """Stop current generation if running"""
        self.stop_requested = True
        if self.current_streamer is not None:
            try:
                self.current_streamer.end()
            except Exception as e:
                logger.error(f"Error stopping streamer: {e}")
        logger.info("TTS generation stop requested")

    def is_generating(self) -> bool:
        """Check if currently generating speech"""
        return self.is_generating_flag
