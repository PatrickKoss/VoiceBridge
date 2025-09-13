#!/usr/bin/env python3
"""
Setup script for TTS functionality in whisper-cli.
This script helps users set up VibeVoice model and voice samples.
"""

import sys
from pathlib import Path


def setup_vibevoice_model():
    """Setup instructions for VibeVoice model."""
    print("=== VibeVoice Model Setup ===")
    print()
    print("To use TTS functionality, you need to set up the VibeVoice model.")
    print()
    print("1. Install VibeVoice dependencies:")
    print("   cd /path/to/VibeVoice")
    print("   pip install -e .")
    print()
    print("2. Download or specify the model path:")
    print("   - Default model: WestZhang/VibeVoice-Large-pt")
    print("   - Or use a local model path")
    print()
    print("3. Set up voice samples directory:")
    print("   - Default: demo/voices")
    print("   - Copy voice samples (WAV files) to this directory")
    print()
    return True


def setup_voice_samples():
    """Setup voice samples directory."""
    print("=== Voice Samples Setup ===")
    print()

    # Create voice samples directory
    voices_dir = Path("demo/voices")
    voices_dir.mkdir(parents=True, exist_ok=True)

    print(f"Created voices directory: {voices_dir.absolute()}")
    print()
    print("Voice samples should be:")
    print("- WAV format (recommended)")
    print("- 24kHz sample rate (recommended)")
    print("- Mono audio")
    print("- 3-10 seconds long")
    print("- Clear, single speaker")
    print()
    print("Naming convention examples:")
    print("- en-Alice_woman.wav")
    print("- en-Carter_man.wav")
    print("- es-Maria_woman.wav")
    print()

    return True


def check_dependencies():
    """Check if required dependencies are installed."""
    print("=== Dependencies Check ===")
    print()

    required = [
        "torch",
        "transformers",
        "soundfile",
        "pyperclip",
        "pygame",
        "librosa",
        "numpy",
        "pynput",
    ]

    missing = []

    for package in required:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing.append(package)

    print()

    if missing:
        print("Missing dependencies. Install with:")
        print(f"pip install {' '.join(missing)}")
        return False
    else:
        print("All dependencies are installed!")
        return True


def create_example_config():
    """Create example TTS configuration."""
    print("=== Example Configuration ===")
    print()

    config_example = """
# Example TTS configuration in ~/.config/whisper-cli/config.json

{
  "tts_enabled": true,
  "tts_config": {
    "model_path": "WestZhang/VibeVoice-Large-pt",
    "voice_samples_dir": "demo/voices",
    "default_voice": "en-Alice_woman",
    "cfg_scale": 1.3,
    "inference_steps": 10,
    "tts_mode": "clipboard",
    "streaming_mode": "non_streaming",
    "output_mode": "play",
    "tts_toggle_key": "f11",
    "tts_generate_key": "f12",
    "tts_stop_key": "ctrl+alt+s",
    "sample_rate": 24000,
    "auto_play": true,
    "use_gpu": true,
    "max_text_length": 2000,
    "chunk_text_threshold": 500
  }
}
"""

    print("Example configuration:")
    print(config_example)
    return True


def main():
    """Main setup function."""
    print("whisper-cli TTS Setup")
    print("=" * 50)
    print()

    # Check dependencies
    deps_ok = check_dependencies()
    print()

    # Setup VibeVoice model
    setup_vibevoice_model()
    print()

    # Setup voice samples
    setup_voice_samples()
    print()

    # Show example configuration
    create_example_config()
    print()

    print("=== Next Steps ===")
    print()
    print("1. Install missing dependencies if any")
    print("2. Set up VibeVoice model (see instructions above)")
    print("3. Copy voice samples to demo/voices/")
    print("4. Configure TTS settings:")
    print("   whisper-cli tts config set --help")
    print("5. Test TTS:")
    print('   whisper-cli tts generate "Hello, this is a test"')
    print()

    if not deps_ok:
        print("⚠️  Please install missing dependencies first!")
        sys.exit(1)
    else:
        print("✅ Setup completed successfully!")


if __name__ == "__main__":
    main()
