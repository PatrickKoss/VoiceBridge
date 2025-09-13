# CLAUDE.md - VoiceBridge Project

This file provides guidance to Claude Code when working with the VoiceBridge project.

## Project Overview

VoiceBridge is a comprehensive bidirectional voice-text CLI tool that bridges speech and text seamlessly. Built on OpenAI's Whisper for speech recognition and VibeVoice for text-to-speech synthesis, with advanced features including:

### Core Features
- **Speech-to-Text (STT)**: Real-time transcription, file processing, batch operations
- **Text-to-Speech (TTS)**: High-quality voice synthesis with custom voices
- **GPU acceleration** (CUDA/Metal) with automatic device detection
- **Memory optimization** and streaming for large audio files  
- **Resume capability** for interrupted transcriptions
- **Performance monitoring** and session management
- **Audio processing**: noise reduction, normalization, splitting, enhancement
- **Export formats**: JSON, SRT, VTT, plain text, CSV
- **Hotkey support** for hands-free operation
- **Hexagonal architecture** with ports and adapters pattern

## Development Setup

### Virtual Environment & Dependencies

**IMPORTANT**: This project uses a local virtual environment at `venv/`. Always use the Makefile commands or the venv Python/pip directly.

```bash
# Initialize environment and install all dependencies
make prepare

# Manual venv setup (if needed)
python3 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt
venv/bin/pip install -r test_requirements.txt
```

### Key Commands

```bash
make help         # Show all available commands
make prepare      # Initialize venv and install dependencies
make lint         # Run ruff linting and auto-fix issues
make test         # Run all tests with coverage report
make test-fast    # Run tests without coverage
make clean        # Clean up cache files and venv
```

### Manual Commands (using venv directly)

```bash
# Always use venv python/pip for any manual commands
venv/bin/python -m ruff check --fix .    # Linting
venv/bin/python -m pytest                # Testing  
venv/bin/python main.py --help           # Run CLI
venv/bin/pip install package-name        # Install packages
```

## Architecture

### Directory Structure
```
voicebridge/
├── domain/          # Core business logic and models
│   └── models.py    # Data models (WhisperConfig, TTSConfig, GPUInfo, etc.)
├── ports/           # Interfaces/abstract base classes  
│   └── interfaces.py
├── adapters/        # External integrations
│   ├── audio/       # Audio recording, playback, processing
│   ├── system.py    # GPU detection, memory monitoring
│   ├── transcription.py     # Whisper service implementation
│   ├── vibevoice_tts.py     # VibeVoice TTS implementation
│   ├── session.py   # Session persistence
│   └── config.py    # Configuration management
├── services/        # Application services
│   ├── transcription_service.py  # STT orchestration
│   ├── tts_service.py           # TTS orchestration and daemon
│   ├── performance_service.py   # Performance monitoring
│   ├── batch_service.py         # Batch processing
│   ├── export_service.py        # Export functionality
│   ├── confidence_service.py    # Quality analysis
│   └── resume_service.py        # Resume functionality
├── cli/             # Command line interface
│   ├── app.py       # Main CLI app with Typer
│   └── commands.py  # Command implementations
└── tests/          # Test suite
```

### Key Features Implemented

#### Speech-to-Text (STT)
1. **Real-time Transcription**: Hotkey-driven live speech recognition
2. **File Processing**: Support for MP3, WAV, M4A, FLAC, OGG formats
3. **Batch Processing**: Directory-wide transcription with parallel workers
4. **GPU Acceleration**: Automatic detection and selection of CUDA/Metal devices
5. **Memory Optimization**: Chunked processing and memory limit enforcement
6. **Resume Capability**: Session persistence for interrupted transcriptions
7. **Export Formats**: JSON, SRT, VTT, plain text, CSV output

#### Text-to-Speech (TTS)
1. **VibeVoice Integration**: High-quality neural voice synthesis
2. **Multiple Input Modes**: Clipboard monitoring, text selection, direct input
3. **Custom Voices**: Voice sample detection and management
4. **Streaming/Non-streaming**: Real-time or complete generation modes
5. **Hotkey Controls**: Global shortcuts for hands-free operation
6. **Audio Output**: Play immediately, save to file, or both

#### Advanced Processing
1. **Audio Enhancement**: Noise reduction, normalization, silence trimming
2. **Audio Splitting**: Duration, silence, or size-based segmentation
3. **Confidence Analysis**: Quality assessment and review flagging
4. **Performance Monitoring**: Comprehensive metrics collection and reporting
5. **Session Management**: Progress tracking and resume functionality
6. **Profile Management**: Multiple configuration profiles
7. **Webhook Integration**: External notification support

## Code Standards

- **Architecture**: Hexagonal/Ports & Adapters pattern
- **Python Version**: 3.10+
- **Linting**: ruff with auto-fix
- **Testing**: pytest with coverage reporting
- **Type Hints**: Required for all public interfaces

## CLI Usage

### Speech-to-Text Commands

```bash
# Real-time transcription with hotkeys
venv/bin/python main.py listen
venv/bin/python main.py hotkey --key f9 --mode toggle

# File transcription
venv/bin/python main.py transcribe audio.mp3 --output transcript.txt
venv/bin/python main.py batch-transcribe /path/to/audio/ --workers 4

# Resumable transcription for long files
venv/bin/python main.py listen-resumable audio.wav --session-name "my-session"

# Real-time streaming transcription
venv/bin/python main.py realtime --chunk-duration 2.0 --output-format live
```

### Text-to-Speech Commands

```bash
# Generate speech from text
venv/bin/python main.py tts generate "Hello, this is VoiceBridge!"
venv/bin/python main.py tts generate "Text here" --voice en-Alice_woman --output audio.wav

# Clipboard and selection monitoring
venv/bin/python main.py tts listen-clipboard --streaming
venv/bin/python main.py tts listen-selection

# TTS daemon mode
venv/bin/python main.py tts daemon start --mode clipboard
venv/bin/python main.py tts daemon status
venv/bin/python main.py tts daemon stop

# Voice management
venv/bin/python main.py tts voices
venv/bin/python main.py tts config show
```

### Audio Processing Commands

```bash
# Audio information and formats
venv/bin/python main.py audio info audio.mp3
venv/bin/python main.py audio formats

# Audio enhancement and splitting
venv/bin/python main.py audio preprocess input.wav output.wav --noise-reduction 0.8
venv/bin/python main.py audio split large_file.mp3 --method duration --chunk-duration 300
```

### System and Performance Commands

```bash
# GPU status and benchmarking
venv/bin/python main.py gpu status
venv/bin/python main.py gpu benchmark --model base

# Performance monitoring
venv/bin/python main.py performance stats

# Session management
venv/bin/python main.py sessions list
venv/bin/python main.py sessions resume --session-id <id>
venv/bin/python main.py sessions cleanup
```

### Export and Analysis Commands

```bash
# Export transcriptions
venv/bin/python main.py export session <session-id> --format srt
venv/bin/python main.py export batch --format json --output-dir results/

# Confidence analysis
venv/bin/python main.py confidence analyze <session-id> --detailed
venv/bin/python main.py confidence analyze-all --threshold 0.7
```

### Configuration Commands

```bash
# General configuration
venv/bin/python main.py config --show
venv/bin/python main.py config --set-key use_gpu --value true

# Profile management
venv/bin/python main.py profile save my-profile
venv/bin/python main.py profile load my-profile
venv/bin/python main.py profile list

# TTS configuration
venv/bin/python main.py tts config set --default-voice en-Alice_woman
venv/bin/python main.py tts config set --cfg-scale 1.5
```

## Development Workflow

1. **Setup**: `make prepare`
2. **Development**: Edit code, use `make lint` frequently
3. **Testing**: `make test` or `make test-fast`
4. **Before Commit**: Ensure `make lint` and `make test` both pass

## Important Notes

### Development
- Always use `venv/bin/python` and `venv/bin/pip` for Python commands
- Use `make lint` to auto-fix most style issues
- Use `make test` to run full test suite with coverage
- Follow hexagonal architecture patterns for new features
- Add comprehensive tests for both STT and TTS functionality

### System Requirements
- **Python 3.10+** for modern type hints and async support
- **FFmpeg** for audio processing and format conversion
- **GPU support**: CUDA (NVIDIA) and Metal (Apple Silicon) detection
- **Audio dependencies**: pygame, pyaudio for playback
- **Input handling**: pyperclip, pynput for clipboard and hotkeys

### Configuration
- Main config stored in `~/.config/whisper-cli/`
- Session files stored in local `sessions/` directory
- TTS voice samples in `demo/voices/` or configured directory
- Performance metrics kept in memory (last 1000 operations)
- Profile-based configuration for different use cases

### TTS Setup
- VibeVoice model: `WestZhang/VibeVoice-Large-pt`
- Voice samples: 3-10 second WAV files, 24kHz recommended
- Naming convention: `language-name_gender.wav` (e.g., `en-Alice_woman.wav`)
- Run `python setup_tts.py` for guided TTS setup

### Hotkeys
- **Default STT hotkey**: F9 (start/stop recording)
- **Default TTS hotkey**: F12 (generate speech from clipboard/selection)
- **TTS stop hotkey**: Ctrl+Alt+S (stop current generation)
- Hotkeys work globally across all applications