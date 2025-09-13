# Whisper CLI - User Experience Features

This document describes the advanced user experience features implemented in the Whisper CLI.

## 🎯 Interactive Mode

Interactive mode allows you to review and edit transcription results before they're finalized.

### Usage
```bash
# Enable interactive mode with any transcription
whisper-cli listen --interactive

# Works with profiles too
whisper-cli listen --profile podcast --interactive
```

### Features
- **Review**: See the transcription result before accepting
- **Edit**: Open your preferred text editor (via `$EDITOR` environment variable)
- **Diff View**: See exactly what changed when editing
- **Accept/Reject**: Choose to accept, reject, or modify results
- **Cross-platform**: Works on Linux, macOS, and Windows

### Commands in Interactive Mode
- `e` or `edit` - Open text editor to modify transcription
- `a` or `accept` - Accept current transcription (default)
- `r` or `reject` - Reject transcription (return empty)
- `h` or `help` - Show help

## 📋 Configuration Profiles

Save and reuse common settings combinations as named profiles.

### Managing Profiles

#### Save a Profile
```bash
# Save current session settings
whisper-cli profile save podcast-setup --model medium --language en --temperature 0.1 --interactive

# Save with performance tracking
whisper-cli profile save debug-session --model small --performance-tracking
```

#### List Profiles
```bash
whisper-cli profile list
```

#### Load Profile Settings
```bash
# View profile contents
whisper-cli profile load podcast-setup

# Use profile in transcription
whisper-cli listen --profile podcast-setup
```

#### Delete Profile
```bash
whisper-cli profile delete old-setup
```

### Profile Storage
- Profiles are stored in `~/.config/whisper-cli/profiles/`
- Each profile is a JSON file with your settings
- Profiles can include any CLI parameter

## 📊 Logging & Debugging

Enhanced logging with performance metrics and detailed debugging information.

### Performance Tracking
```bash
# Enable performance logging
whisper-cli listen --performance-tracking

# View performance logs
whisper-cli logs --performance

# Follow logs in real-time (Linux/macOS)
whisper-cli logs --follow
```

### Debug Logging
```bash
# Enable debug mode in daemon
whisper-cli config --set-key debug --value true

# View recent logs
whisper-cli logs --lines 100
```

### What Gets Logged
- **Model Loading Times**: How long it takes to load different models
- **Session Duration**: Total time from start to finish
- **Text Metrics**: Length of transcribed text
- **Feature Usage**: Which features were used (interactive, profiles, etc.)
- **Error Details**: Detailed debugging information for troubleshooting

### Log Files
- **Main Log**: `~/.config/whisper-cli/whisper.log`
- **Performance Log**: `~/.config/whisper-cli/performance.log`

## 🔄 Auto-Update Checking

Stay informed about new Whisper releases from OpenAI.

### Check for Updates
```bash
whisper-cli update
```

### What You Get
- Latest release version
- Release date
- Direct link to GitHub release page

### Example Output
```
Checking for updates...
Latest Whisper release: 20250625
Published: 2025-06-26
URL: https://github.com/openai/whisper/releases/tag/v20250625
```

## 🚀 Enhanced Usage Examples

### Complete Workflow Examples

#### Podcast Transcription Setup
```bash
# Create a podcast profile
whisper-cli profile save podcast \
  --model large \
  --language en \
  --temperature 0.0 \
  --interactive \
  --performance-tracking

# Use it for transcription
whisper-cli listen --profile podcast
```

#### Debug Session
```bash
# Create debug profile
whisper-cli profile save debug \
  --model small \
  --performance-tracking

# Enable debug logging
whisper-cli config --set-key debug --value true

# Run transcription
whisper-cli listen --profile debug

# Check logs
whisper-cli logs --performance --lines 20
```

#### Quick Meeting Notes
```bash
# Quick profile for meetings
whisper-cli profile save meeting \
  --model medium \
  --interactive \
  --copy

# Use for quick notes
whisper-cli listen --profile meeting
```

## 📁 File Structure

All user data is stored in `~/.config/whisper-cli/`:

```
~/.config/whisper-cli/
├── config.json          # Main configuration
├── daemon.pid           # Daemon process ID
├── whisper.log          # Main application logs
├── performance.log      # Performance metrics
└── profiles/            # Configuration profiles
    ├── podcast.json
    ├── meeting.json
    └── debug.json
```

## 🎛️ Integration with Existing Features

All new features integrate seamlessly with existing functionality:

- **Daemon Mode**: Profiles and logging work with background daemon
- **Hotkey Mode**: Performance tracking works with global hotkeys  
- **System Tray**: Enhanced logging helps with tray debugging
- **Streaming**: Interactive mode works with real-time transcription
- **Cross-Platform**: All features work on Windows, macOS, and Linux

## 🔧 Configuration Options

New configuration keys available:

- `interactive`: Enable interactive mode by default
- `performance_tracking`: Enable performance logging by default  
- `debug`: Enable detailed debug logging

Set via:
```bash
whisper-cli config --set-key interactive --value true
```

## 📈 Performance Benefits

- **Faster Setup**: Profiles eliminate repetitive parameter entry
- **Better Quality**: Interactive editing improves transcription accuracy
- **Easier Debugging**: Detailed logs help identify and fix issues
- **Stay Updated**: Auto-update checking keeps you informed of improvements
- **Data-Driven**: Performance metrics help optimize your workflow

---

These user experience improvements make Whisper CLI more powerful, flexible, and user-friendly while maintaining its simplicity and performance.