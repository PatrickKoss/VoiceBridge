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

- **Real-time transcription** with global hotkeys
- **File processing** (MP3, WAV, M4A, FLAC, OGG)
- **Batch transcription** of entire directories
- **GPU acceleration** (CUDA/Metal) for faster processing
- **Resume capability** for interrupted long transcriptions
- **Custom vocabulary** for domain-specific terms
- **Export formats**: JSON, SRT, VTT, plain text, CSV

### 🗣️ Text-to-Speech (TTS)

- **High-quality voice synthesis** with VibeVoice
- **Multiple input modes**: clipboard, text selection, direct input
- **Custom voice samples** with automatic detection
- **Real-time streaming** or complete generation
- **Hotkey controls** for hands-free operation

### 🔧 Advanced Processing

- **Audio enhancement**: noise reduction, normalization, silence trimming
- **Confidence analysis** and quality assessment
- **Session management** with progress tracking
- **Performance monitoring** and GPU benchmarking
- **Webhook integration** for external notifications
- **Profile management** for different use cases

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/voicebridge.git
cd voicebridge

# Set up environment
make prepare

# Or manually:
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Basic Usage

```bash
# Listen for speech and transcribe (hotkey mode)
python main.py listen

# Transcribe an audio file
python main.py transcribe audio.mp3 --output transcript.txt

# Generate speech from text
python main.py tts generate "Hello, this is VoiceBridge!"

# Start clipboard monitoring for TTS
python main.py tts listen-clipboard
```

## 📖 Examples

### 1. Content Creator Workflow

```bash
# Transcribe a podcast episode with timestamps
python main.py transcribe podcast_episode.mp3 \
  --format srt \
  --output episode_subtitles.srt \
  --language en

# Analyze transcription quality
python main.py confidence analyze session_12345 --detailed
```

### 2. Accessibility Content

```bash
# Convert article to speech with custom voice
python main.py tts generate \
  --voice en-Alice_woman \
  --output article_audio.wav \
  "$(cat article.txt)"

# Batch convert multiple documents
python main.py batch-transcribe articles/ \
  --output-dir transcripts/ \
  --workers 4
```

### 3. Developer Integration

```bash
# Start daemon for background processing
python main.py start

# Set up webhook notifications
python main.py webhook add https://api.example.com/transcription-complete

# Real-time transcription with streaming
python main.py realtime \
  --chunk-duration 2.0 \
  --output-format live
```

### 4. Research & Analysis

```bash
# Process interview recordings with confidence analysis
python main.py transcribe interview.wav \
  --session-name "interview-2024-01-15" \
  --language en

# Export results in multiple formats
python main.py export batch \
  --format json \
  --include-confidence \
  --output-dir results/
```

## 🛠️ Local Development Setup

### Prerequisites

- **Python 3.10+**
- **FFmpeg** (for audio processing)
- **CUDA** (optional, for GPU acceleration)

### Installation

```bash
# 1. Clone and setup
git clone https://github.com/yourusername/voicebridge.git
cd voicebridge
make prepare

# 2. Install system dependencies
# Ubuntu/Debian:
sudo apt update && sudo apt install ffmpeg

# macOS:
brew install ffmpeg

# Windows (with Chocolatey):
choco install ffmpeg
```

### TTS Setup (Optional)

```bash
# Install VibeVoice for text-to-speech
python setup_tts.py

# Add voice samples to demo/voices/
# Format: language-name_gender.wav (e.g., en-Alice_woman.wav)
```

### Development Commands

```bash
make help         # Show all available commands
make lint         # Run code formatting and linting
make test         # Run test suite with coverage
make test-fast    # Quick tests without coverage
make clean        # Clean cache and temporary files
```

### Configuration

```bash
# Show current configuration
python main.py config --show

# Enable GPU acceleration
python main.py config --set-key use_gpu --value true

# Set up profiles for different use cases
python main.py profile save research-setup
python main.py profile load research-setup
```

## 🎮 Usage Guide

### Hotkey Mode

```bash
# Start hotkey listener
python main.py hotkey --key f9 --mode toggle

# Available hotkeys:
# F9: Start/stop recording (default)
# F12: Generate TTS from clipboard
# Ctrl+Alt+S: Stop TTS generation
```

### Session Management

```bash
# List all transcription sessions
python main.py sessions list

# Resume interrupted transcription
python main.py sessions resume --session-name "my-session"

# Clean up completed sessions
python main.py sessions cleanup
```

### Audio Processing

```bash
# Split large audio file
python main.py audio split recording.mp3 \
  --method duration \
  --chunk-duration 300

# Enhance audio quality
python main.py audio preprocess input.wav output.wav \
  --noise-reduction 0.8 \
  --normalize -3 \
  --trim-silence
```

### Performance Monitoring

```bash
# Check GPU status
python main.py gpu status

# Benchmark performance
python main.py gpu benchmark --model base

# View performance statistics
python main.py performance stats
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

- **Python 3.10+** with type hints
- **Black** for formatting (88 character limit)
- **Ruff** for linting
- **Pytest** for testing with >90% coverage
- **Hexagonal architecture** for new features
- **Comprehensive documentation** for public APIs

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
