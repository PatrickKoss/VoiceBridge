import typer
from cli.commands import CLICommands


def create_app(commands: CLICommands) -> typer.Typer:
    app = typer.Typer(add_completion=False, help="Whisper hotkey mic-to-text CLI")

    @app.command()
    def listen(
        model: str | None = typer.Option(
            None, "--model", "-m", help="Whisper model to use"
        ),
        language: str | None = typer.Option(
            None, "--language", "-l", help="Language code"
        ),
        initial_prompt: str | None = typer.Option(
            None, "--prompt", help="Initial prompt"
        ),
        temperature: float = typer.Option(
            0.0, "--temperature", "-t", help="Temperature"
        ),
        profile: str | None = typer.Option(None, "--profile", "-p", help="Use profile"),
        paste_stream: bool = typer.Option(
            False, "--paste-stream", help="Paste streaming text"
        ),
        copy_stream: bool = typer.Option(
            False, "--copy-stream", help="Copy streaming text"
        ),
        paste_final: bool = typer.Option(
            False, "--paste-final", help="Paste final text"
        ),
        copy_final: bool = typer.Option(
            True, "--copy-final/--no-copy-final", help="Copy final text"
        ),
        debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
    ):
        """Listen for speech and transcribe it."""
        commands.listen(
            model,
            language,
            initial_prompt,
            temperature,
            profile,
            paste_stream,
            copy_stream,
            paste_final,
            copy_final,
            debug,
        )

    @app.command()
    def hotkey(
        model: str | None = typer.Option(
            None, "--model", "-m", help="Whisper model to use"
        ),
        language: str | None = typer.Option(
            None, "--language", "-l", help="Language code"
        ),
        key: str = typer.Option("f9", "--key", "-k", help="Hotkey to use"),
        mode: str = typer.Option(
            "toggle", "--mode", help="Mode: toggle or push_to_talk"
        ),
        profile: str | None = typer.Option(None, "--profile", "-p", help="Use profile"),
        debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
    ):
        """Global-hotkey-controlled recording for speech transcription."""
        commands.hotkey(model, language, key, mode, profile, debug)

    @app.command()
    def start():
        """Start the whisper daemon."""
        commands.daemon_start()

    @app.command()
    def stop():
        """Stop the whisper daemon."""
        commands.daemon_stop()

    @app.command()
    def status():
        """Show daemon status."""
        commands.daemon_status()

    @app.command()
    def config(
        show: bool = typer.Option(False, "--show", help="Show current config"),
        set_key: str | None = typer.Option(None, "--set-key", help="Set config key"),
        value: str | None = typer.Option(None, "--value", help="Config value to set"),
    ):
        """Manage configuration."""
        if show:
            commands.config_show()
        elif set_key and value:
            commands.config_set(set_key, value)
        elif set_key:
            typer.echo("Error: --value is required when using --set-key", err=True)
            raise typer.Exit(1)
        else:
            typer.echo(
                "Use --show to display config or --set-key key --value val to set a value"
            )

    @app.command()
    def profile(
        save: str | None = typer.Option(None, "--save", help="Save profile"),
        load: str | None = typer.Option(None, "--load", help="Load profile"),
        list_profiles: bool = typer.Option(False, "--list", help="List profiles"),
        delete: str | None = typer.Option(None, "--delete", help="Delete profile"),
    ):
        """Manage configuration profiles."""
        if save:
            commands.profile_save(save)
        elif load:
            commands.profile_load(load)
        elif list_profiles:
            commands.profile_list()
        elif delete:
            commands.profile_delete(delete)
        else:
            commands.profile_list()  # Default to list

    # GPU and Performance Commands
    gpu_app = typer.Typer(help="GPU device management")

    @gpu_app.command("status")
    def gpu_status():
        """Show GPU device status and capabilities."""
        commands.gpu_status()

    @gpu_app.command("benchmark")
    def gpu_benchmark(
        model: str = typer.Option("base", "--model", "-m", help="Model to benchmark"),
    ):
        """Benchmark GPU performance with a model."""
        commands.gpu_benchmark(model)

    app.add_typer(gpu_app, name="gpu")

    # Performance Commands
    performance_app = typer.Typer(help="Performance monitoring and statistics")

    @performance_app.command("stats")
    def performance_stats():
        """Show performance statistics."""
        commands.performance_stats()

    app.add_typer(performance_app, name="performance")

    # Session Management Commands
    sessions_app = typer.Typer(help="Transcription session management")

    @sessions_app.command("list")
    def sessions_list():
        """List all transcription sessions."""
        commands.sessions_list()

    @sessions_app.command("resume")
    def sessions_resume(
        session_id: str | None = typer.Option(
            None, "--session-id", help="Session ID to resume"
        ),
        session_name: str | None = typer.Option(
            None, "--session-name", help="Session name to resume"
        ),
    ):
        """Resume a transcription session."""
        commands.sessions_resume(session_id, session_name)

    @sessions_app.command("cleanup")
    def sessions_cleanup():
        """Clean up completed sessions."""
        commands.sessions_cleanup()

    @sessions_app.command("delete")
    def sessions_delete(
        session_id: str = typer.Argument(..., help="Session ID to delete"),
    ):
        """Delete a specific session."""
        commands.sessions_delete(session_id)

    app.add_typer(sessions_app, name="sessions")

    # Vocabulary Management Commands
    vocabulary_app = typer.Typer(help="Custom vocabulary management")

    @vocabulary_app.command("add")
    def vocabulary_add(
        word: str = typer.Argument(..., help="Word to add to vocabulary"),
        vocabulary_type: str = typer.Option(
            "custom",
            "--type",
            help="Vocabulary type: custom, proper_nouns, technical, domain",
        ),
        domain: str | None = typer.Option(
            None, "--domain", help="Domain name (for domain type)"
        ),
        profile: str = typer.Option(
            "default", "--profile", help="Vocabulary profile name"
        ),
    ):
        """Add a word to custom vocabulary."""
        commands.vocabulary_add(word, vocabulary_type, domain, profile)

    @vocabulary_app.command("remove")
    def vocabulary_remove(
        word: str = typer.Argument(..., help="Word to remove from vocabulary"),
        vocabulary_type: str = typer.Option("custom", "--type", help="Vocabulary type"),
        domain: str | None = typer.Option(
            None, "--domain", help="Domain name (for domain type)"
        ),
        profile: str = typer.Option(
            "default", "--profile", help="Vocabulary profile name"
        ),
    ):
        """Remove a word from custom vocabulary."""
        commands.vocabulary_remove(word, vocabulary_type, domain, profile)

    @vocabulary_app.command("list")
    def vocabulary_list(
        vocabulary_type: str | None = typer.Option(
            None, "--type", help="Show specific vocabulary type"
        ),
        profile: str = typer.Option(
            "default", "--profile", help="Vocabulary profile name"
        ),
    ):
        """List vocabulary words."""
        commands.vocabulary_list(vocabulary_type, profile)

    @vocabulary_app.command("import")
    def vocabulary_import(
        file_path: str = typer.Argument(..., help="Path to vocabulary file"),
        vocabulary_type: str = typer.Option("custom", "--type", help="Vocabulary type"),
        profile: str = typer.Option(
            "default", "--profile", help="Vocabulary profile name"
        ),
    ):
        """Import vocabulary from file."""
        commands.vocabulary_import(file_path, vocabulary_type, profile)

    @vocabulary_app.command("export")
    def vocabulary_export(
        file_path: str = typer.Argument(..., help="Path to export vocabulary"),
        profile: str = typer.Option(
            "default", "--profile", help="Vocabulary profile name"
        ),
    ):
        """Export vocabulary to file."""
        commands.vocabulary_export(file_path, profile)

    app.add_typer(vocabulary_app, name="vocabulary")

    # Post-processing Commands
    postproc_app = typer.Typer(help="Text post-processing configuration")

    @postproc_app.command("config")
    def postproc_config(
        show: bool = typer.Option(
            False, "--show", help="Show current post-processing config"
        ),
        enable_punctuation: bool | None = typer.Option(
            None, "--punctuation/--no-punctuation", help="Enable punctuation cleanup"
        ),
        enable_capitalization: bool | None = typer.Option(
            None, "--capitalization/--no-capitalization", help="Enable capitalization"
        ),
        enable_profanity_filter: bool | None = typer.Option(
            None,
            "--profanity-filter/--no-profanity-filter",
            help="Enable profanity filter",
        ),
        remove_filler_words: bool | None = typer.Option(
            None, "--remove-filler/--keep-filler", help="Remove filler words"
        ),
        profile: str = typer.Option(
            "default", "--profile", help="Post-processing profile name"
        ),
    ):
        """Configure post-processing settings."""
        commands.postprocessing_config(
            show,
            enable_punctuation,
            enable_capitalization,
            enable_profanity_filter,
            remove_filler_words,
            profile,
        )

    @postproc_app.command("test")
    def postproc_test(
        text: str = typer.Argument(..., help="Text to test post-processing on"),
        profile: str = typer.Option(
            "default", "--profile", help="Post-processing profile name"
        ),
    ):
        """Test post-processing on sample text."""
        commands.postprocessing_test(text, profile)

    app.add_typer(postproc_app, name="postproc")

    # Webhook Commands
    webhook_app = typer.Typer(help="Webhook integration management")

    @webhook_app.command("add")
    def webhook_add(
        url: str = typer.Argument(..., help="Webhook URL"),
        event_types: str = typer.Option(
            "transcription_complete", "--events", help="Comma-separated event types"
        ),
        auth_header: str | None = typer.Option(
            None, "--auth", help="Authorization header value"
        ),
    ):
        """Add a webhook endpoint."""
        commands.webhook_add(url, event_types, auth_header)

    @webhook_app.command("remove")
    def webhook_remove(
        url: str = typer.Argument(..., help="Webhook URL to remove"),
    ):
        """Remove a webhook endpoint."""
        commands.webhook_remove(url)

    @webhook_app.command("list")
    def webhook_list():
        """List configured webhooks."""
        commands.webhook_list()

    @webhook_app.command("test")
    def webhook_test(
        url: str = typer.Argument(..., help="Webhook URL to test"),
        event_type: str = typer.Option(
            "transcription_complete", "--event", help="Event type to test"
        ),
    ):
        """Test a webhook endpoint."""
        commands.webhook_test(url, event_type)

    app.add_typer(webhook_app, name="webhook")

    # API Server Commands
    api_app = typer.Typer(help="REST API server management")

    @api_app.command("start")
    def api_start(
        port: int = typer.Option(8000, "--port", help="Port to run API server on"),
        host: str = typer.Option(
            "127.0.0.1", "--host", help="Host to bind API server to"
        ),
        workers: int = typer.Option(1, "--workers", help="Number of worker processes"),
    ):
        """Start the REST API server."""
        commands.api_start(port, host, workers)

    @api_app.command("stop")
    def api_stop():
        """Stop the REST API server."""
        commands.api_stop()

    @api_app.command("status")
    def api_status():
        """Show API server status."""
        commands.api_status()

    app.add_typer(api_app, name="api")

    # Progress and Operations Commands
    operations_app = typer.Typer(help="Operation and progress management")

    @operations_app.command("list")
    def operations_list():
        """List active operations."""
        commands.operations_list()

    @operations_app.command("cancel")
    def operations_cancel(
        operation_id: str = typer.Argument(..., help="Operation ID to cancel"),
    ):
        """Cancel an active operation."""
        commands.operations_cancel(operation_id)

    @operations_app.command("status")
    def operations_status(
        operation_id: str = typer.Argument(..., help="Operation ID to check"),
    ):
        """Check operation status."""
        commands.operations_status(operation_id)

    app.add_typer(operations_app, name="operations")

    # Circuit Breaker Commands
    circuit_app = typer.Typer(help="Circuit breaker management")

    @circuit_app.command("status")
    def circuit_status():
        """Show circuit breaker status for all services."""
        commands.circuit_breaker_status()

    @circuit_app.command("reset")
    def circuit_reset(
        service: str | None = typer.Option(
            None, "--service", help="Service name to reset (all if not specified)"
        ),
    ):
        """Reset circuit breaker for service(s)."""
        commands.circuit_breaker_reset(service)

    @circuit_app.command("stats")
    def circuit_stats():
        """Show circuit breaker statistics."""
        commands.circuit_breaker_stats()

    app.add_typer(circuit_app, name="circuit")

    # Export Commands
    export_app = typer.Typer(help="Export transcription results")

    @export_app.command("session")
    def export_session(
        session_id: str = typer.Argument(..., help="Session ID to export"),
        format: str = typer.Option("txt", "--format", "-f", help="Export format"),
        output_file: str | None = typer.Option(
            None, "--output", "-o", help="Output file path"
        ),
        timestamp_mode: str = typer.Option(
            "sentence", "--timestamps", help="Timestamp mode"
        ),
        include_confidence: bool = typer.Option(
            True, "--confidence/--no-confidence", help="Include confidence scores"
        ),
        include_speaker_info: bool = typer.Option(
            True, "--speakers/--no-speakers", help="Include speaker information"
        ),
    ):
        """Export a specific transcription session."""
        commands.export_transcription(
            session_id,
            format,
            output_file,
            timestamp_mode,
            include_confidence,
            include_speaker_info,
        )

    @export_app.command("batch")
    def batch_export(
        format: str = typer.Option("txt", "--format", "-f", help="Export format"),
        output_dir: str = typer.Option(
            "./exports", "--output-dir", "-o", help="Output directory"
        ),
        timestamp_mode: str = typer.Option(
            "sentence", "--timestamps", help="Timestamp mode"
        ),
        include_confidence: bool = typer.Option(
            True, "--confidence/--no-confidence", help="Include confidence scores"
        ),
        include_speaker_info: bool = typer.Option(
            True, "--speakers/--no-speakers", help="Include speaker information"
        ),
    ):
        """Export all transcription sessions."""
        commands.batch_export_sessions(
            format, output_dir, timestamp_mode, include_confidence, include_speaker_info
        )

    @export_app.command("formats")
    def list_formats():
        """List supported export formats."""
        commands.list_export_formats()

    app.add_typer(export_app, name="export")

    # Confidence Analysis Commands
    confidence_app = typer.Typer(help="Confidence analysis and quality assessment")

    @confidence_app.command("analyze")
    def analyze_confidence(
        session_id: str = typer.Argument(..., help="Session ID to analyze"),
        detailed: bool = typer.Option(
            False, "--detailed", help="Show detailed analysis"
        ),
        flag_threshold: float = typer.Option(
            0.6, "--threshold", help="Flag threshold for review"
        ),
    ):
        """Analyze transcription confidence for a session."""
        commands.analyze_confidence(session_id, detailed, flag_threshold)

    @confidence_app.command("analyze-all")
    def analyze_all(
        detailed: bool = typer.Option(
            False, "--detailed", help="Show detailed analysis for each session"
        ),
        threshold: float = typer.Option(
            0.7, "--threshold", help="Confidence threshold for flagging"
        ),
    ):
        """Analyze confidence for all sessions."""
        commands.analyze_all_sessions(detailed, threshold)

    @confidence_app.command("configure")
    def configure_thresholds(
        high_threshold: float = typer.Option(
            0.9, "--high", help="High confidence threshold"
        ),
        medium_threshold: float = typer.Option(
            0.7, "--medium", help="Medium confidence threshold"
        ),
        low_threshold: float = typer.Option(
            0.5, "--low", help="Low confidence threshold"
        ),
        review_threshold: float = typer.Option(
            0.6, "--review", help="Review threshold"
        ),
    ):
        """Configure confidence analysis thresholds."""
        commands.set_confidence_thresholds(
            high_threshold, medium_threshold, low_threshold, review_threshold
        )

    app.add_typer(confidence_app, name="confidence")

    # Enhanced file transcription command
    @app.command("transcribe")
    def transcribe_file(
        audio_file: str = typer.Argument(..., help="Path to audio file"),
        output_path: str | None = typer.Option(
            None, "--output", "-o", help="Output file path"
        ),
        model: str | None = typer.Option(
            None, "--model", "-m", help="Whisper model to use"
        ),
        language: str | None = typer.Option(
            None, "--language", "-l", help="Language code"
        ),
        temperature: float = typer.Option(
            0.0, "--temperature", help="Temperature for transcription"
        ),
        format_output: str = typer.Option("txt", "--format", help="Output format"),
    ):
        """Transcribe an audio file (supports MP3, WAV, M4A, FLAC, OGG)."""
        commands.transcribe_file(
            audio_file, output_path, model, language, temperature, format_output
        )

    # Audio Processing Commands
    audio_app = typer.Typer(help="Audio processing and manipulation")

    @audio_app.command("info")
    def audio_info(file_path: str = typer.Argument(..., help="Path to audio file")):
        """Show detailed information about an audio file."""
        commands.show_audio_info(file_path)

    @audio_app.command("formats")
    def audio_formats():
        """List supported audio formats."""
        commands.list_audio_formats()

    @audio_app.command("split")
    def audio_split(
        file_path: str = typer.Argument(..., help="Path to audio file to split"),
        output_dir: str = typer.Option(
            "split_audio", "--output-dir", "-o", help="Output directory"
        ),
        method: str = typer.Option(
            "duration", "--method", "-m", help="Split method: duration, silence, size"
        ),
        chunk_duration: int = typer.Option(
            300, "--duration", help="Chunk duration in seconds (for duration method)"
        ),
        silence_threshold: float = typer.Option(
            0.01, "--silence-threshold", help="Silence threshold (for silence method)"
        ),
        max_size_mb: float = typer.Option(
            25.0, "--max-size", help="Maximum file size in MB (for size method)"
        ),
    ):
        """Split audio file into chunks."""
        commands.split_audio(
            file_path,
            output_dir,
            method,
            chunk_duration,
            silence_threshold,
            max_size_mb,
        )

    @audio_app.command("preprocess")
    def audio_preprocess(
        input_path: str = typer.Argument(..., help="Input audio file"),
        output_path: str = typer.Argument(..., help="Output audio file"),
        noise_reduction: float | None = typer.Option(
            None, "--noise-reduction", help="Noise reduction strength (0.0-1.0)"
        ),
        normalize_volume: float | None = typer.Option(
            None, "--normalize", help="Normalize volume to dB level"
        ),
        trim_silence: bool = typer.Option(
            False, "--trim-silence", help="Trim silence from start/end"
        ),
        enhance_speech: bool = typer.Option(
            False, "--enhance-speech", help="Apply speech enhancement filters"
        ),
    ):
        """Preprocess audio file (noise reduction, normalization, etc.)."""
        commands.preprocess_audio(
            input_path,
            output_path,
            noise_reduction,
            normalize_volume,
            trim_silence,
            enhance_speech,
        )

    app.add_typer(audio_app, name="audio")

    # TTS Commands
    tts_app = typer.Typer(help="Text-to-speech commands")

    @tts_app.command("generate")
    def tts_generate(
        text: str = typer.Argument(..., help="Text to convert to speech"),
        voice: str | None = typer.Option(None, "--voice", "-v", help="Voice to use"),
        streaming: bool = typer.Option(False, "--streaming", help="Use streaming mode"),
        output_file: str | None = typer.Option(
            None, "--output", "-o", help="Save audio to file"
        ),
        auto_play: bool = typer.Option(
            True, "--auto-play/--no-auto-play", help="Auto-play generated audio"
        ),
    ):
        """Generate TTS from provided text"""
        commands.tts_generate_from_text(text, voice, streaming, output_file, auto_play)

    @tts_app.command("listen-clipboard")
    def tts_listen_clipboard(
        voice: str | None = typer.Option(None, "--voice", "-v", help="Voice to use"),
        streaming: bool = typer.Option(False, "--streaming", help="Use streaming mode"),
        auto_play: bool = typer.Option(
            True, "--auto-play/--no-auto-play", help="Auto-play generated audio"
        ),
        output_file: str | None = typer.Option(
            None, "--output", "-o", help="Save audio to file"
        ),
    ):
        """Monitor clipboard for text changes and generate TTS"""
        commands.tts_listen_clipboard(voice, streaming, auto_play, output_file)

    @tts_app.command("listen-selection")
    def tts_listen_selection(
        voice: str | None = typer.Option(None, "--voice", "-v", help="Voice to use"),
        streaming: bool = typer.Option(False, "--streaming", help="Use streaming mode"),
        auto_play: bool = typer.Option(
            True, "--auto-play/--no-auto-play", help="Auto-play generated audio"
        ),
    ):
        """Listen for text selections and generate TTS via hotkey"""
        commands.tts_listen_selection(voice, streaming, auto_play)

    # TTS Daemon Commands
    tts_daemon_app = typer.Typer(help="TTS daemon management")

    @tts_daemon_app.command("start")
    def tts_daemon_start(
        voice: str | None = typer.Option(None, "--voice", "-v", help="Voice to use"),
        mode: str = typer.Option(
            "clipboard", "--mode", "-m", help="Mode: clipboard or selection"
        ),
        streaming: bool = typer.Option(False, "--streaming", help="Use streaming mode"),
        auto_play: bool = typer.Option(
            True, "--auto-play/--no-auto-play", help="Auto-play generated audio"
        ),
    ):
        """Start TTS daemon with hotkey support"""
        commands.tts_daemon_start(voice, mode, streaming, auto_play)

    @tts_daemon_app.command("stop")
    def tts_daemon_stop():
        """Stop TTS daemon"""
        commands.tts_daemon_stop()

    @tts_daemon_app.command("status")
    def tts_daemon_status():
        """Check TTS daemon status"""
        commands.tts_daemon_status()

    tts_app.add_typer(tts_daemon_app, name="daemon")

    # TTS Voice Commands
    @tts_app.command("voices")
    def tts_voices():
        """List available TTS voices"""
        commands.tts_list_voices()

    # TTS Config Commands
    tts_config_app = typer.Typer(help="TTS configuration management")

    @tts_config_app.command("show")
    def tts_config_show():
        """Show current TTS configuration"""
        commands.tts_config_show()

    @tts_config_app.command("set")
    def tts_config_set(
        model_path: str | None = typer.Option(
            None, "--model-path", help="Set model path"
        ),
        voice_samples_dir: str | None = typer.Option(
            None, "--voices-dir", help="Set voice samples directory"
        ),
        default_voice: str | None = typer.Option(
            None, "--default-voice", help="Set default voice"
        ),
        cfg_scale: float | None = typer.Option(
            None, "--cfg-scale", help="Set CFG scale"
        ),
        inference_steps: int | None = typer.Option(
            None, "--inference-steps", help="Set inference steps"
        ),
    ):
        """Set TTS configuration options"""
        commands.tts_config_set(
            model_path, voice_samples_dir, default_voice, cfg_scale, inference_steps
        )

    tts_app.add_typer(tts_config_app, name="config")

    app.add_typer(tts_app, name="tts")

    # Batch processing command
    @app.command("batch-transcribe")
    def batch_transcribe(
        input_dir: str = typer.Argument(..., help="Directory containing audio files"),
        output_dir: str = typer.Option(
            "transcriptions", "--output-dir", "-o", help="Output directory"
        ),
        workers: int = typer.Option(
            4, "--workers", "-w", help="Number of parallel workers"
        ),
        file_pattern: str | None = typer.Option(
            None, "--pattern", help="File pattern to match (e.g., '*.mp3')"
        ),
        model: str | None = typer.Option(
            None, "--model", "-m", help="Whisper model to use"
        ),
    ):
        """Batch transcribe all audio files in a directory."""
        commands.batch_transcribe(input_dir, output_dir, workers, file_pattern, model)

    # Real-time transcription with advanced options
    @app.command("realtime")
    def realtime_transcribe(
        chunk_duration: float = typer.Option(
            2.0, "--chunk-duration", help="Chunk duration in seconds"
        ),
        overlap: float = typer.Option(
            0.5, "--overlap", help="Overlap between chunks in seconds"
        ),
        vad_threshold: float = typer.Option(
            0.01, "--vad-threshold", help="Voice activity detection threshold"
        ),
        output_format: str = typer.Option(
            "live", "--output-format", help="Output format: live, segments, complete"
        ),
        model: str | None = typer.Option(
            None, "--model", "-m", help="Whisper model to use"
        ),
    ):
        """Real-time transcription with configurable parameters."""
        commands.realtime_transcribe(
            chunk_duration, overlap, vad_threshold, output_format, model
        )

    return app
