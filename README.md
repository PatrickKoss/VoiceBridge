# VoiceBridge 🎙️ ↔️ 📝

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Platform Support](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)]()

> **The ultimate bidirectional voice-text bridge.** Seamlessly convert speech to text and text to speech with professional-grade accuracy, real-time processing, and hotkey-driven workflows.

## 🚀 What is VoiceBridge?

VoiceBridge eliminates the friction between voice and text. Whether you're transcribing interviews, creating accessible content, building voice-driven workflows, or simply need hands-free text input, VoiceBridge provides a powerful, flexible CLI that adapts to your needs.

**Built on OpenAI's Whisper** for world-class speech recognition and **VibeVoice** for natural text-to-speech synthesis.

## 🎯 What Problems Does It Solve?

- **Content Creators**: Transcribe podcasts, interviews, and videos with timestamp precision
- **Accessibility**: Convert text to natural speech for screen readers and audio content
- **Productivity**: Voice-to-text note-taking with hotkey triggers during meetings
- **Developers**: Integrate speech processing into applications and workflows
- **Researchers**: Batch process audio data with confidence analysis and quality metrics
- **Writers**: Dictate drafts and have articles read back with custom voices

## ✨ Key Features

### 🎤 Speech-to-Text (STT)

- **Real-time transcription** with hotkeys (F9 toggle/hold modes)
- **Interactive mode** with press-and-hold 'r' to record
- **File processing** (MP3, WAV, M4A, FLAC, OGG) with chunked processing
- **Batch transcription** of entire directories with parallel workers
- **Resume capability** for interrupted long transcriptions with session management
- **Streaming transcription** with real-time output and live updates
- **GPU acceleration** (CUDA/Metal) with automatic device detection
- **Memory optimization** with configurable limits and streaming
- **Custom vocabulary** management for domain-specific terms
- **Export formats**: JSON, SRT, VTT, plain text, CSV with timestamps and confidence
- **Confidence analysis** and quality assessment with detailed reporting
- **Webhook integration** for external notifications and automation
- **Post-processing** with spell check, grammar correction, and custom rules
- **Profile management** for different use cases and configurations
- **Performance monitoring** with comprehensive metrics and benchmarking

### 🗣️ Text-to-Speech (TTS)

- **High-quality voice synthesis** with VibeVoice neural models
- **Multiple input modes**: clipboard monitoring, text selection, direct input
- **Custom voice samples** with automatic detection and voice cloning
- **Streaming and non-streaming** modes for real-time or complete generation
- **Daemon mode** for background processing and system integration
- **Hotkey controls** for hands-free operation (F12 generate, Ctrl+Alt+S stop)
- **Voice management** with sample validation and quality checks
- **GPU acceleration** for faster synthesis and model loading
- **Configuration profiles** for different voice settings and use cases
- **Audio output options**: play immediately, save to file, or both

### 🔧 Advanced Processing

- **Audio enhancement**: noise reduction, normalization, silence trimming, fade effects
- **Audio splitting**: by duration, silence detection, or file size with smart segmentation
- **Confidence analysis** and quality assessment with detailed statistics
- **Session management** with progress tracking, resume capability, and persistence
- **Performance monitoring** with GPU benchmarking, memory usage, and operation tracking
- **Webhook integration** for external notifications and workflow automation
- **Profile management** for different use cases and quick configuration switching
- **Vocabulary management** for improved recognition of technical terms and proper nouns
- **Post-processing pipeline** with spell check, grammar correction, and custom rules
- **API server** for integration with external applications and services
- **Comprehensive testing** with E2E test suites for all major functionality

## 🚀 Quick Start

### Installation

VoiceBridge uses **uv** for fast dependency management. Install uv first if you don't have it:

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/yourusername/voicebridge.git
cd voicebridge

# Set up environment (CPU version)
make prepare

# Or with CUDA support for GPU acceleration
make prepare-cuda

# Or with system tray support
make prepare-tray
```

### Basic Usage

```bash
# Listen for speech and transcribe with hotkeys
uv run python -m voicebridge stt listen

# Transcribe an audio file
uv run python -m voicebridge stt transcribe audio.mp3 --output transcript.txt

# Generate speech from text
uv run python -m voicebridge tts generate "Hello, this is VoiceBridge!"

# Start clipboard monitoring for TTS
uv run python -m voicebridge tts listen-clipboard
```

## 📖 Examples

### 1. Content Creator Workflow

```bash
# Transcribe a podcast episode with timestamps
uv run python -m voicebridge stt transcribe podcast_episode.mp3 \
  --format srt \
  --output episode_subtitles.srt \
  --language en

# Analyze transcription quality
uv run python -m voicebridge stt confidence analyze session_12345 --detailed
```

### 2. Accessibility Content

```bash
# Convert article to speech with custom voice
uv run python -m voicebridge tts generate \
  --voice en-Alice_woman \
  --output article_audio.wav \
  "$(cat article.txt)"

# Batch convert multiple documents
uv run python -m voicebridge stt batch-transcribe articles/ \
  --output-dir transcripts/ \
  --workers 4
```

### 3. Developer Integration

```bash
# Start TTS daemon for background processing
uv run python -m voicebridge tts daemon start --mode clipboard

# Set up webhook notifications
uv run python -m voicebridge stt webhook add https://api.example.com/transcription-complete

# Real-time transcription with streaming
uv run python -m voicebridge stt realtime \
  --chunk-duration 2.0 \
  --output-format live
```

### 4. Research & Analysis

```bash
# Process interview recordings with resumable capability
uv run python -m voicebridge stt listen-resumable interview.wav \
  --session-name "interview-2024-01-15" \
  --language en

# Export results in multiple formats
uv run python -m voicebridge stt export session session_12345 \
  --format json \
  --include-confidence \
  --output transcript.json
```

## 🛠️ Local Development Setup

### Prerequisites

- **Python 3.10+**
- **uv** (Python package manager)
- **FFmpeg** (for audio processing)
- **CUDA** (optional, for GPU acceleration)

### Installation

```bash
# 1. Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and setup
git clone https://github.com/yourusername/voicebridge.git
cd voicebridge

# 3. Choose your setup:
make prepare        # CPU version
make prepare-cuda   # With CUDA support  
make prepare-tray   # With system tray support

# 4. Install system dependencies
# Ubuntu/Debian:
sudo apt update && sudo apt install ffmpeg

# macOS:
brew install ffmpeg

# Windows (with Chocolatey):
choco install ffmpeg
```

### TTS Setup

VoiceBridge includes comprehensive text-to-speech capabilities powered by VibeVoice.

#### Prerequisites

1. **Install VibeVoice dependencies** (if using local model):

   ```bash
   # Clone and install VibeVoice
   git clone https://github.com/WestZhang/VibeVoice.git
   cd VibeVoice
   pip install -e .
   ```

2. **Voice Samples**: Voice samples are included in `voices/` directory:
   ```
   voices/
   ├── en-Alice_woman.wav
   ├── en-Carter_man.wav
   ├── en-Frank_man.wav
   ├── en-Maya_woman.wav
   ├── en-Patrick.wav
   └── ... (additional voices)
   ```

#### Configuration

VoiceBridge works out-of-the-box with sensible defaults. Configuration can be set via:

1. **Config file** (`~/.config/whisper-cli/config.json`):

   ```json
   {
     "tts_enabled": true,
     "tts_config": {
       "model_path": "aoi-ot/VibeVoice-7B",
       "voice_samples_dir": "voices",
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
   ```

2. **Command-line flags** (override config file):
   ```bash
   # Generate with custom settings
   uv run python -m voicebridge tts generate "Hello world" \
     --voice en-Patrick \
     --streaming \
     --output speech.wav \
     --cfg-scale 1.5 \
     --inference-steps 15
   ```

#### Voice Sample Requirements

- **Format**: WAV (recommended), MP3, FLAC
- **Sample Rate**: 24kHz (recommended), 16kHz-48kHz supported
- **Channels**: Mono (preferred)
- **Duration**: 3-10 seconds
- **Quality**: Clear, single speaker, minimal background noise
- **Naming**: `language-name_gender.wav` (e.g., `en-Alice_woman.wav`)

#### Quick Test

```bash
# Test TTS with default settings
uv run python -m voicebridge tts generate "Hello, this is VoiceBridge text-to-speech!"

# List available voices
uv run python -m voicebridge tts voices

# Show current TTS configuration
uv run python -m voicebridge tts config show
```

### Development Commands

```bash
make help           # Show all available commands
make lint           # Run ruff linting and formatting
make test           # Run all tests with coverage
make test-fast      # Quick tests without coverage
make test-unit      # Run only unit tests (exclude e2e)
make test-e2e       # Run comprehensive end-to-end tests
make test-e2e-smoke # Run quick E2E smoke tests
make test-e2e-stt   # Run STT E2E tests only
make test-e2e-tts   # Run TTS E2E tests only
make test-e2e-audio # Run audio E2E tests only
make test-e2e-gpu   # Run GPU E2E tests only
make test-e2e-api   # Run API E2E tests only
make clean          # Clean cache and temporary files
```

### Configuration

```bash
# Show current STT configuration
uv run python -m voicebridge stt config show

# Set STT configuration values
uv run python -m voicebridge stt config set use_gpu true

# Show TTS configuration
uv run python -m voicebridge tts config show

# Set up profiles for different use cases
uv run python -m voicebridge stt profile save research-setup
uv run python -m voicebridge stt profile load research-setup
```

## 🎮 Usage Guide

### Speech-to-Text (STT) Commands

#### Real-time Recognition
```bash
# Listen with hotkeys (F9 to start/stop)
uv run python -m voicebridge stt listen

# Interactive mode (press 'r' to record)
uv run python -m voicebridge stt interactive

# Global hotkey listener with custom key
uv run python -m voicebridge stt hotkey --key f9 --mode toggle
```

#### File Processing
```bash
# Transcribe single file
uv run python -m voicebridge stt transcribe audio.mp3 --output transcript.txt

# Batch process directory
uv run python -m voicebridge stt batch-transcribe /path/to/audio/ --workers 4

# Long file with resume capability
uv run python -m voicebridge stt listen-resumable large_file.wav --session-name "my-session"

# Real-time streaming
uv run python -m voicebridge stt realtime --chunk-duration 2.0 --output-format live
```

#### Session Management
```bash
# List all sessions
uv run python -m voicebridge stt sessions list

# Resume interrupted session
uv run python -m voicebridge stt sessions resume --session-name "my-session"

# Clean up old sessions
uv run python -m voicebridge stt sessions cleanup

# Delete specific session
uv run python -m voicebridge stt sessions delete session_id
```

#### Advanced Features
```bash
# Add vocabulary words for better recognition
uv run python -m voicebridge stt vocabulary add "technical,terms,here" --type technical

# Export with confidence analysis
uv run python -m voicebridge stt export session session_id --format srt --confidence

# Set up webhooks for notifications
uv run python -m voicebridge stt webhook add https://api.example.com/notify
```

### Text-to-Speech (TTS) Commands

#### Basic Generation
```bash
# Generate speech from text
uv run python -m voicebridge tts generate "Hello, this is VoiceBridge!"

# Use specific voice and save to file
uv run python -m voicebridge tts generate "Hello world" --voice en-Alice_woman --output speech.wav

# List available voices
uv run python -m voicebridge tts voices
```

#### Background Monitoring
```bash
# Monitor clipboard for text changes
uv run python -m voicebridge tts listen-clipboard --streaming

# Monitor text selections (use hotkey to trigger)
uv run python -m voicebridge tts listen-selection

# Start TTS daemon for background processing
uv run python -m voicebridge tts daemon start --mode clipboard
uv run python -m voicebridge tts daemon status
uv run python -m voicebridge tts daemon stop
```

#### Configuration
```bash
# Show TTS settings
uv run python -m voicebridge tts config show

# Configure TTS settings
uv run python -m voicebridge tts config set --default-voice en-Alice_woman --cfg-scale 1.5
```

### Audio Processing

```bash
# Get audio file information
uv run python -m voicebridge audio info audio.mp3

# List supported formats
uv run python -m voicebridge audio formats

# Split large audio file
uv run python -m voicebridge audio split recording.mp3 \
  --method duration \
  --chunk-duration 300

# Enhance audio quality
uv run python -m voicebridge audio preprocess input.wav output.wav \
  --noise-reduction 0.8 \
  --normalize \
  --trim-silence

# Test audio setup
uv run python -m voicebridge audio test
```

### System & Performance

```bash
# Check GPU status and acceleration
uv run python -m voicebridge gpu status

# Benchmark GPU performance
uv run python -m voicebridge gpu benchmark --model base

# View STT performance statistics
uv run python -m voicebridge stt performance stats

# Manage active operations
uv run python -m voicebridge stt operations list
uv run python -m voicebridge stt operations cancel operation_id
```

### API Server

```bash
# Start API server
uv run python -m voicebridge api start --host localhost --port 8000

# Check API status
uv run python -m voicebridge api status

# Get API information
uv run python -m voicebridge api info

# Stop API server
uv run python -m voicebridge api stop
```

## 📋 Complete Command Reference

VoiceBridge uses a hierarchical command structure with five main categories:

### 🎤 `stt` - Speech-to-Text Commands
```
stt listen              # Real-time transcription with hotkeys
stt interactive         # Press-and-hold 'r' to record mode
stt hotkey              # Global hotkey listener
stt transcribe          # Transcribe single audio file
stt batch-transcribe    # Batch process directory
stt listen-resumable    # Long file with resume capability
stt realtime            # Real-time streaming transcription

# Session Management
stt sessions list       # List all sessions
stt sessions resume     # Resume interrupted session
stt sessions cleanup    # Clean up old sessions
stt sessions delete     # Delete specific session

# Advanced Features
stt vocabulary add      # Add custom vocabulary
stt vocabulary remove   # Remove vocabulary
stt vocabulary list     # List vocabulary
stt vocabulary import   # Import from file
stt vocabulary export   # Export to file

stt export session      # Export session data
stt export formats      # List export formats

stt confidence analyze  # Analyze transcription confidence
stt confidence analyze-all # Analyze all sessions

stt postproc config     # Configure post-processing
stt postproc test       # Test post-processing

stt webhook add         # Add webhook notification
stt webhook remove      # Remove webhook
stt webhook list        # List webhooks
stt webhook test        # Test webhook

stt performance stats   # Performance statistics
stt operations list     # List active operations
stt operations cancel   # Cancel operation
stt operations status   # Check operation status

stt config show         # Show configuration
stt config set          # Set configuration

stt profile save        # Save configuration profile
stt profile load        # Load configuration profile
stt profile list        # List profiles
stt profile delete      # Delete profile
```

### 🗣️ `tts` - Text-to-Speech Commands
```
tts generate            # Generate speech from text
tts listen-clipboard    # Monitor clipboard changes
tts listen-selection    # Monitor text selections with hotkey
tts voices              # List available voices

# Daemon Management
tts daemon start        # Start TTS daemon
tts daemon stop         # Stop TTS daemon
tts daemon status       # Check daemon status

# Configuration
tts config show         # Show TTS configuration
tts config set          # Configure TTS settings
```

### 🔊 `audio` - Audio Processing Commands
```
audio info              # Show audio file information
audio formats           # List supported formats
audio split             # Split audio file into chunks
audio preprocess        # Enhance audio quality
audio test              # Test audio setup
```

### 🖥️ `gpu` - GPU and System Commands
```
gpu status              # Show GPU status
gpu benchmark           # Benchmark GPU performance
```

### 🌐 `api` - API Server Management
```
api start               # Start API server
api stop                # Stop API server
api status              # Check API status
api info                # Show API information
```

## 🏗️ Architecture

VoiceBridge follows **hexagonal architecture** principles:

```
voicebridge/
├── domain/          # Core business logic and models
├── ports/           # Interfaces and abstractions
├── adapters/        # External integrations (Whisper, VibeVoice, etc.)
├── services/        # Application services and orchestration
├── cli/             # Command-line interface
└── tests/          # Comprehensive test suite
```

### Key Components

- **Domain Layer**: Core models, configurations, and business rules
- **Ports**: Abstract interfaces for transcription, TTS, audio processing
- **Adapters**: Concrete implementations for Whisper, VibeVoice, FFmpeg
- **Services**: Orchestration, session management, performance monitoring
- **CLI**: Typer-based command interface with sub-commands

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Workflow

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Install** development dependencies: `make install-dev`
4. **Make** your changes following our coding standards
5. **Test** your changes: `make test`
6. **Lint** your code: `make lint`
7. **Commit** your changes: `git commit -m 'Add amazing feature'`
8. **Push** to your branch: `git push origin feature/amazing-feature`
9. **Open** a Pull Request

### Coding Standards

- **Python 3.10+** with comprehensive type hints
- **uv** for fast dependency management and virtual environments
- **Ruff** for linting and formatting (replaces Black and isort)
- **Pytest** for testing with >90% coverage target
- **Hexagonal architecture** for new features and clean separation of concerns
- **Comprehensive documentation** for public APIs and CLI commands
- **E2E testing** for all major CLI workflows and functionality
- **Makefile** for standardized development commands

### Areas for Contribution

- 🎯 **New audio formats** and processing capabilities
- 🌍 **Language support** and localization
- 🔧 **Performance optimizations** and GPU utilization
- 📱 **Platform integrations** (mobile, web interfaces)
- 🧪 **Test coverage** and edge case handling
- 📚 **Documentation** and usage examples
- 🎨 **Voice samples** and TTS improvements

### Reporting Issues

Please use our **issue templates**:

- 🐛 **Bug Report**: Describe the issue with reproduction steps
- 💡 **Feature Request**: Propose new functionality
- 📚 **Documentation**: Report unclear or missing docs
- 🏃 **Performance**: Report slow or resource-intensive operations

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI Whisper** - State-of-the-art speech recognition
- **VibeVoice** - High-quality text-to-speech synthesis
- **FFmpeg** - Comprehensive audio processing
- **Typer** - Modern CLI framework
- **PyTorch** - Machine learning infrastructure
