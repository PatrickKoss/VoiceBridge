"""
Refactored CLI application using modular command structure.
"""

import typer

from voicebridge.cli.registry import CommandRegistry


def create_app(command_registry: CommandRegistry) -> typer.Typer:
    """
    Create the main CLI application with modular command structure.

    Args:
        command_registry: Registry containing all command groups

    Returns:
        Configured Typer application
    """
    app = typer.Typer(
        add_completion=False,
        help="VoiceBridge - Comprehensive bidirectional voice-text CLI tool"
    )

    # Speech Recognition Commands
    @app.command()
    def listen(
        model: str | None = typer.Option(None, "--model", "-m", help="Whisper model to use"),
        language: str | None = typer.Option(None, "--language", "-l", help="Language code"),
        initial_prompt: str | None = typer.Option(None, "--prompt", help="Initial prompt"),
        temperature: float = typer.Option(0.0, "--temperature", "-t", help="Temperature"),
        profile: str | None = typer.Option(None, "--profile", "-p", help="Use profile"),
        paste_stream: bool = typer.Option(False, "--paste-stream", help="Paste streaming text"),
        copy_stream: bool = typer.Option(False, "--copy-stream", help="Copy streaming text"),
        paste_final: bool = typer.Option(False, "--paste-final", help="Paste final text"),
        copy_final: bool = typer.Option(True, "--copy-final/--no-copy-final", help="Copy final text"),
        debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
    ):
        """Listen for speech and transcribe it with hotkey control."""
        speech_commands = command_registry.get_command_group('speech')
        speech_commands.listen(
            model, language, initial_prompt, temperature, profile,
            paste_stream, copy_stream, paste_final, copy_final, debug
        )

    @app.command()
    def interactive(
        model: str | None = typer.Option(None, "--model", "-m", help="Whisper model to use"),
        language: str | None = typer.Option(None, "--language", "-l", help="Language code"),
        initial_prompt: str | None = typer.Option(None, "--prompt", help="Initial prompt"),
        temperature: float = typer.Option(0.0, "--temperature", "-t", help="Temperature"),
        profile: str | None = typer.Option(None, "--profile", "-p", help="Use profile"),
        paste_stream: bool = typer.Option(False, "--paste-stream", help="Paste streaming text"),
        copy_stream: bool = typer.Option(False, "--copy-stream", help="Copy streaming text"),
        paste_final: bool = typer.Option(False, "--paste-final", help="Paste final text"),
        copy_final: bool = typer.Option(True, "--copy-final/--no-copy-final", help="Copy final text"),
        debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
    ):
        """Interactive mode with press-and-hold 'r' to record."""
        speech_commands = command_registry.get_command_group('speech')
        speech_commands.interactive(
            model, language, initial_prompt, temperature, profile,
            paste_stream, copy_stream, paste_final, copy_final, debug
        )

    @app.command()
    def hotkey(
        key: str = typer.Option("f9", "--key", help="Hotkey to use"),
        mode: str = typer.Option("toggle", "--mode", help="Hotkey mode: toggle or hold"),
        model: str | None = typer.Option(None, "--model", "-m", help="Whisper model to use"),
        language: str | None = typer.Option(None, "--language", "-l", help="Language code"),
        initial_prompt: str | None = typer.Option(None, "--prompt", help="Initial prompt"),
        temperature: float = typer.Option(0.0, "--temperature", "-t", help="Temperature"),
        profile: str | None = typer.Option(None, "--profile", "-p", help="Use profile"),
        paste_stream: bool = typer.Option(False, "--paste-stream", help="Paste streaming text"),
        copy_stream: bool = typer.Option(False, "--copy-stream", help="Copy streaming text"),
        paste_final: bool = typer.Option(False, "--paste-final", help="Paste final text"),
        copy_final: bool = typer.Option(True, "--copy-final/--no-copy-final", help="Copy final text"),
        debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
    ):
        """Global hotkey listener for speech recognition."""
        speech_commands = command_registry.get_command_group('speech')
        speech_commands.hotkey(
            key, mode, model, language, initial_prompt, temperature, profile,
            paste_stream, copy_stream, paste_final, copy_final, debug
        )

    # File Transcription Commands
    @app.command()
    def transcribe(
        file_path: str = typer.Argument(..., help="Path to audio file"),
        output_path: str | None = typer.Option(None, "--output", "-o", help="Output file path"),
        model: str | None = typer.Option(None, "--model", "-m", help="Whisper model to use"),
        language: str | None = typer.Option(None, "--language", "-l", help="Language code"),
        temperature: float = typer.Option(0.0, "--temperature", "-t", help="Temperature"),
        format_output: str = typer.Option("txt", "--format", "-f", help="Output format"),
    ):
        """Transcribe an audio file."""
        transcription_commands = command_registry.get_command_group('transcription')
        transcription_commands.transcribe_file(
            file_path, output_path, model, language, temperature, format_output
        )

    @app.command(name="batch-transcribe")
    def batch_transcribe(
        input_dir: str = typer.Argument(..., help="Input directory"),
        output_dir: str = typer.Option("transcriptions", "--output-dir", help="Output directory"),
        workers: int = typer.Option(4, "--workers", "-w", help="Number of parallel workers"),
        file_pattern: str | None = typer.Option(None, "--pattern", help="File pattern to match"),
        model: str | None = typer.Option(None, "--model", "-m", help="Whisper model to use"),
    ):
        """Batch transcribe all audio files in a directory."""
        transcription_commands = command_registry.get_command_group('transcription')
        transcription_commands.batch_transcribe(
            input_dir, output_dir, workers, file_pattern, model
        )

    @app.command(name="listen-resumable")
    def listen_resumable(
        file_path: str = typer.Argument(..., help="Path to audio file"),
        session_name: str | None = typer.Option(None, "--session-name", help="Session name"),
        model: str | None = typer.Option(None, "--model", "-m", help="Whisper model to use"),
        language: str | None = typer.Option(None, "--language", "-l", help="Language code"),
        temperature: float = typer.Option(0.0, "--temperature", "-t", help="Temperature"),
        profile: str | None = typer.Option(None, "--profile", "-p", help="Use profile"),
        chunk_size: int = typer.Option(30, "--chunk-size", help="Chunk size in seconds"),
        overlap: int = typer.Option(5, "--overlap", help="Overlap in seconds"),
    ):
        """Transcribe a long audio file with resume capability."""
        transcription_commands = command_registry.get_command_group('transcription')
        transcription_commands.listen_resumable(
            file_path, session_name, model, language, temperature, profile, chunk_size, overlap
        )

    @app.command()
    def realtime(
        chunk_duration: float = typer.Option(2.0, "--chunk-duration", help="Chunk duration in seconds"),
        output_format: str = typer.Option("live", "--output-format", help="Output format: live or segments"),
        model: str | None = typer.Option(None, "--model", "-m", help="Whisper model to use"),
        language: str | None = typer.Option(None, "--language", "-l", help="Language code"),
        temperature: float = typer.Option(0.0, "--temperature", "-t", help="Temperature"),
        profile: str | None = typer.Option(None, "--profile", "-p", help="Use profile"),
        save_audio: bool = typer.Option(False, "--save-audio", help="Save audio to file"),
        output_file: str | None = typer.Option(None, "--output-file", help="Output file for transcription"),
    ):
        """Real-time streaming transcription with live output."""
        transcription_commands = command_registry.get_command_group('transcription')
        transcription_commands.realtime_transcribe(
            chunk_duration, output_format, model, language, temperature, profile, save_audio, output_file
        )

    # Create TTS sub-application
    tts_app = typer.Typer(help="Text-to-Speech commands")
    app.add_typer(tts_app, name="tts")

    @tts_app.command()
    def generate(
        text: str = typer.Argument(..., help="Text to convert to speech"),
        voice: str | None = typer.Option(None, "--voice", help="Voice to use"),
        streaming: bool = typer.Option(False, "--streaming", help="Use streaming mode"),
        output: str | None = typer.Option(None, "--output", "-o", help="Output audio file"),
        play: bool = typer.Option(True, "--play/--no-play", help="Play generated audio"),
        cfg_scale: float | None = typer.Option(None, "--cfg-scale", help="CFG scale"),
        inference_steps: int | None = typer.Option(None, "--inference-steps", help="Inference steps"),
        sample_rate: int | None = typer.Option(None, "--sample-rate", help="Sample rate"),
        use_gpu: bool | None = typer.Option(None, "--use-gpu/--no-gpu", help="Use GPU acceleration"),
    ):
        """Generate TTS from text."""
        tts_commands = command_registry.get_command_group('tts')
        tts_commands.tts_generate(
            text, voice, streaming, output, play, cfg_scale, inference_steps, sample_rate, use_gpu
        )

    @tts_app.command(name="listen-clipboard")
    def tts_listen_clipboard(
        voice: str | None = typer.Option(None, "--voice", help="Voice to use"),
        streaming: bool = typer.Option(False, "--streaming", help="Use streaming mode"),
        auto_play: bool = typer.Option(True, "--auto-play/--no-auto-play", help="Auto play generated audio"),
        output_file: str | None = typer.Option(None, "--output", help="Output file"),
    ):
        """Listen to clipboard changes and generate TTS."""
        tts_commands = command_registry.get_command_group('tts')
        tts_commands.tts_listen_clipboard(voice, streaming, auto_play, output_file)

    @tts_app.command(name="listen-selection")
    def tts_listen_selection(
        voice: str | None = typer.Option(None, "--voice", help="Voice to use"),
        streaming: bool = typer.Option(False, "--streaming", help="Use streaming mode"),
        auto_play: bool = typer.Option(True, "--auto-play/--no-auto-play", help="Auto play generated audio"),
    ):
        """Listen for text selections and generate TTS via hotkey."""
        tts_commands = command_registry.get_command_group('tts')
        tts_commands.tts_listen_selection(voice, streaming, auto_play)

    # TTS Daemon commands
    tts_daemon_app = typer.Typer(help="TTS daemon management")
    tts_app.add_typer(tts_daemon_app, name="daemon")

    @tts_daemon_app.command()
    def start(
        voice: str | None = typer.Option(None, "--voice", help="Voice to use"),
        mode: str = typer.Option("clipboard", "--mode", help="Mode: clipboard or selection"),
        streaming: bool = typer.Option(False, "--streaming", help="Use streaming mode"),
        auto_play: bool = typer.Option(True, "--auto-play/--no-auto-play", help="Auto play generated audio"),
        background: bool = typer.Option(False, "--background", help="Run in background"),
    ):
        """Start TTS daemon."""
        tts_commands = command_registry.get_command_group('tts')
        tts_commands.tts_daemon_start(voice, mode, streaming, auto_play, background)

    @tts_daemon_app.command()
    def stop():
        """Stop TTS daemon."""
        tts_commands = command_registry.get_command_group('tts')
        tts_commands.tts_daemon_stop()

    @tts_daemon_app.command()
    def status():
        """Show TTS daemon status."""
        tts_commands = command_registry.get_command_group('tts')
        tts_commands.tts_daemon_status()

    @tts_app.command()
    def voices():
        """List available TTS voices."""
        tts_commands = command_registry.get_command_group('tts')
        tts_commands.tts_list_voices()

    # TTS Config commands
    tts_config_app = typer.Typer(help="TTS configuration")
    tts_app.add_typer(tts_config_app, name="config")

    @tts_config_app.command()
    def show():
        """Show TTS configuration."""
        tts_commands = command_registry.get_command_group('tts')
        tts_commands.tts_config_show()

    @tts_config_app.command()
    def set(
        default_voice: str | None = typer.Option(None, "--default-voice", help="Default voice"),
        voice_samples_path: str | None = typer.Option(None, "--voice-samples-path", help="Voice samples path"),
        model_path: str | None = typer.Option(None, "--model-path", help="Model path"),
        sample_rate: int | None = typer.Option(None, "--sample-rate", help="Sample rate"),
        cfg_scale: float | None = typer.Option(None, "--cfg-scale", help="CFG scale"),
        inference_steps: int | None = typer.Option(None, "--inference-steps", help="Inference steps"),
        auto_play: bool | None = typer.Option(None, "--auto-play/--no-auto-play", help="Auto play"),
        use_gpu: bool | None = typer.Option(None, "--use-gpu/--no-gpu", help="Use GPU"),
        generate_key: str | None = typer.Option(None, "--generate-key", help="Generate hotkey"),
        stop_key: str | None = typer.Option(None, "--stop-key", help="Stop hotkey"),
        output_mode: str | None = typer.Option(None, "--output-mode", help="Output mode"),
        streaming_mode: str | None = typer.Option(None, "--streaming-mode", help="Streaming mode"),
    ):
        """Configure TTS settings."""
        tts_commands = command_registry.get_command_group('tts')
        tts_commands.tts_config_set(
            default_voice, voice_samples_path, model_path, sample_rate, cfg_scale,
            inference_steps, auto_play, use_gpu, generate_key, stop_key, output_mode, streaming_mode
        )

    # Audio Processing Commands
    audio_app = typer.Typer(help="Audio processing commands")
    app.add_typer(audio_app, name="audio")

    @audio_app.command()
    def info(
        file_path: str = typer.Argument(..., help="Audio file path"),
    ):
        """Show audio file information."""
        audio_commands = command_registry.get_command_group('audio')
        audio_commands.audio_info(file_path)

    @audio_app.command()
    def formats():
        """List supported audio formats."""
        audio_commands = command_registry.get_command_group('audio')
        audio_commands.audio_formats()

    @audio_app.command()
    def split(
        file_path: str = typer.Argument(..., help="Audio file path"),
        output_dir: str = typer.Option("split_audio", "--output-dir", help="Output directory"),
        method: str = typer.Option("duration", "--method", help="Split method: duration, silence, size"),
        chunk_duration: int = typer.Option(300, "--chunk-duration", help="Chunk duration in seconds"),
        silence_threshold: float = typer.Option(0.01, "--silence-threshold", help="Silence threshold"),
        max_size_mb: float = typer.Option(25.0, "--max-size-mb", help="Maximum size in MB"),
    ):
        """Split audio file into chunks."""
        transcription_commands = command_registry.get_command_group('transcription')
        transcription_commands.split_audio(
            file_path, output_dir, method, chunk_duration, silence_threshold, max_size_mb
        )

    @audio_app.command()
    def preprocess(
        input_file: str = typer.Argument(..., help="Input audio file"),
        output_file: str = typer.Argument(..., help="Output audio file"),
        noise_reduction: float = typer.Option(0.0, "--noise-reduction", help="Noise reduction level"),
        normalize: bool = typer.Option(False, "--normalize", help="Normalize audio"),
        trim_silence: bool = typer.Option(False, "--trim-silence", help="Trim silence"),
        silence_threshold: float = typer.Option(0.01, "--silence-threshold", help="Silence threshold"),
        fade_in: float = typer.Option(0.0, "--fade-in", help="Fade in duration"),
        fade_out: float = typer.Option(0.0, "--fade-out", help="Fade out duration"),
    ):
        """Preprocess audio with enhancement."""
        audio_commands = command_registry.get_command_group('audio')
        audio_commands.audio_preprocess(
            input_file, output_file, noise_reduction, normalize, trim_silence,
            silence_threshold, fade_in, fade_out
        )

    @audio_app.command()
    def test():
        """Test audio recording and playback setup."""
        transcription_commands = command_registry.get_command_group('transcription')
        transcription_commands.test_audio_setup()

    # System Commands
    gpu_app = typer.Typer(help="GPU and system commands")
    app.add_typer(gpu_app, name="gpu")

    @gpu_app.command()
    def status():
        """Show GPU status."""
        system_commands = command_registry.get_command_group('system')
        system_commands.gpu_status()

    @gpu_app.command()
    def benchmark(
        model: str = typer.Option("base", "--model", help="Model to benchmark"),
    ):
        """Benchmark GPU performance."""
        system_commands = command_registry.get_command_group('system')
        system_commands.gpu_benchmark(model)

    performance_app = typer.Typer(help="Performance monitoring")
    app.add_typer(performance_app, name="performance")

    @performance_app.command()
    def stats():
        """Show performance statistics."""
        system_commands = command_registry.get_command_group('system')
        system_commands.performance_stats()

    # Session Management
    sessions_app = typer.Typer(help="Session management")
    app.add_typer(sessions_app, name="sessions")

    @sessions_app.command()
    def list():
        """List all sessions."""
        system_commands = command_registry.get_command_group('system')
        system_commands.sessions_list()

    @sessions_app.command()
    def resume(
        session_id: str | None = typer.Option(None, "--session-id", help="Session ID"),
        session_name: str | None = typer.Option(None, "--session-name", help="Session name"),
    ):
        """Resume a session."""
        system_commands = command_registry.get_command_group('system')
        system_commands.sessions_resume(session_id, session_name)

    @sessions_app.command()
    def cleanup():
        """Clean up old sessions."""
        system_commands = command_registry.get_command_group('system')
        system_commands.sessions_cleanup()

    @sessions_app.command()
    def delete(
        session_id: str = typer.Argument(..., help="Session ID to delete"),
    ):
        """Delete a session."""
        system_commands = command_registry.get_command_group('system')
        system_commands.sessions_delete(session_id)

    # Configuration Commands
    config_app = typer.Typer(help="Configuration management")
    app.add_typer(config_app, name="config")

    @config_app.command()
    def show():
        """Show current configuration."""
        config_commands = command_registry.get_command_group('config')
        config_commands.config_show()

    @config_app.command()
    def set(
        key: str = typer.Argument(..., help="Configuration key"),
        value: str = typer.Argument(..., help="Configuration value"),
    ):
        """Set configuration value."""
        config_commands = command_registry.get_command_group('config')
        config_commands.config_set(key, value)

    # Profile management
    profile_app = typer.Typer(help="Profile management")
    app.add_typer(profile_app, name="profile")

    @profile_app.command()
    def save(
        name: str = typer.Argument(..., help="Profile name"),
    ):
        """Save current configuration as profile."""
        config_commands = command_registry.get_command_group('config')
        config_commands.profile_save(name)

    @profile_app.command()
    def load(
        name: str = typer.Argument(..., help="Profile name"),
    ):
        """Load a configuration profile."""
        config_commands = command_registry.get_command_group('config')
        config_commands.profile_load(name)

    @profile_app.command()
    def list():
        """List all profiles."""
        config_commands = command_registry.get_command_group('config')
        config_commands.profile_list()

    @profile_app.command()
    def delete(
        name: str = typer.Argument(..., help="Profile name"),
    ):
        """Delete a profile."""
        config_commands = command_registry.get_command_group('config')
        config_commands.profile_delete(name)

    # Export Commands
    export_app = typer.Typer(help="Export and analysis")
    app.add_typer(export_app, name="export")

    @export_app.command()
    def session(
        session_id: str = typer.Argument(..., help="Session ID"),
        format: str = typer.Option("txt", "--format", help="Export format"),
        output_file: str | None = typer.Option(None, "--output", help="Output file"),
        include_timestamps: bool = typer.Option(False, "--timestamps", help="Include timestamps"),
        include_confidence: bool = typer.Option(False, "--confidence", help="Include confidence scores"),
        timestamp_mode: str = typer.Option("absolute", "--timestamp-mode", help="Timestamp mode"),
    ):
        """Export transcription session."""
        export_commands = command_registry.get_command_group('export')
        export_commands.export_transcription(
            session_id, format, output_file, include_timestamps, include_confidence, timestamp_mode
        )

    @export_app.command()
    def formats():
        """List export formats."""
        export_commands = command_registry.get_command_group('export')
        export_commands.list_export_formats()

    # Confidence Analysis
    confidence_app = typer.Typer(help="Confidence analysis")
    app.add_typer(confidence_app, name="confidence")

    @confidence_app.command()
    def analyze(
        session_id: str = typer.Argument(..., help="Session ID"),
        detailed: bool = typer.Option(False, "--detailed", help="Detailed analysis"),
        threshold: float = typer.Option(0.7, "--threshold", help="Confidence threshold"),
    ):
        """Analyze transcription confidence."""
        export_commands = command_registry.get_command_group('export')
        export_commands.analyze_confidence(session_id, detailed, threshold)

    @confidence_app.command(name="analyze-all")
    def analyze_all(
        detailed: bool = typer.Option(False, "--detailed", help="Detailed analysis"),
        threshold: float = typer.Option(0.7, "--threshold", help="Confidence threshold"),
    ):
        """Analyze confidence for all sessions."""
        export_commands = command_registry.get_command_group('export')
        export_commands.analyze_all_sessions(detailed, threshold)

    return app
