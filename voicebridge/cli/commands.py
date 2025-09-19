import os
from pathlib import Path

import typer

from voicebridge.domain.models import (
    ExportConfig,
    OutputFormat,
    TimestampMode,
    TTSMode,
    TTSOutputMode,
    TTSStreamingMode,
    WhisperConfig,
)
from voicebridge.ports.interfaces import (
    AudioFormatService,
    AudioPreprocessingService,
    AudioSplittingService,
    BatchProcessingService,
    CircuitBreakerService,
    ConfigRepository,
    DaemonService,
    ExportService,
    Logger,
    PerformanceService,
    PostProcessingService,
    ProfileRepository,
    ProgressService,
    RetryService,
    SessionService,
    SystemService,
    TimestampService,
    VocabularyService,
    WebhookService,
)
from voicebridge.services.confidence_service import ConfidenceAnalyzer
from voicebridge.services.resume_service import TranscriptionResumeService
from voicebridge.services.transcription_service import WhisperTranscriptionOrchestrator
from voicebridge.services.tts_service import TTSDaemonService, TTSOrchestrator


class CLICommands:
    def __init__(
        self,
        config_repo: ConfigRepository,
        profile_repo: ProfileRepository,
        daemon_service: DaemonService,
        transcription_orchestrator: WhisperTranscriptionOrchestrator,
        system_service: SystemService,
        logger: Logger,
        session_service: SessionService | None = None,
        performance_service: PerformanceService | None = None,
        resume_service: TranscriptionResumeService | None = None,
        vocabulary_service: VocabularyService | None = None,
        postprocessing_service: PostProcessingService | None = None,
        webhook_service: WebhookService | None = None,
        progress_service: ProgressService | None = None,
        retry_service: RetryService | None = None,
        circuit_breaker_service: CircuitBreakerService | None = None,
        export_service: ExportService | None = None,
        timestamp_service: TimestampService | None = None,
        confidence_analyzer: ConfidenceAnalyzer | None = None,
        audio_format_service: AudioFormatService | None = None,
        audio_preprocessing_service: AudioPreprocessingService | None = None,
        audio_splitting_service: AudioSplittingService | None = None,
        batch_processing_service: BatchProcessingService | None = None,
        # TTS Services
        tts_orchestrator: TTSOrchestrator | None = None,
        tts_daemon_service: TTSDaemonService | None = None,
    ):
        self.config_repo = config_repo
        self.profile_repo = profile_repo
        self.daemon_service = daemon_service
        self.transcription_orchestrator = transcription_orchestrator
        self.system_service = system_service
        self.logger = logger
        self.session_service = session_service
        self.performance_service = performance_service
        self.resume_service = resume_service
        self.vocabulary_service = vocabulary_service
        self.postprocessing_service = postprocessing_service
        self.webhook_service = webhook_service
        self.progress_service = progress_service
        self.retry_service = retry_service
        self.circuit_breaker_service = circuit_breaker_service
        self.export_service = export_service
        self.timestamp_service = timestamp_service
        self.confidence_analyzer = confidence_analyzer
        self.audio_format_service = audio_format_service
        self.audio_preprocessing_service = audio_preprocessing_service
        self.audio_splitting_service = audio_splitting_service
        self.batch_processing_service = batch_processing_service
        # TTS Services
        self.tts_orchestrator = tts_orchestrator
        self.tts_daemon_service = tts_daemon_service

    def listen(
        self,
        model: str | None = None,
        language: str | None = None,
        initial_prompt: str | None = None,
        temperature: float = 0.0,
        profile: str | None = None,
        paste_stream: bool = False,
        copy_stream: bool = False,
        paste_final: bool = False,
        copy_final: bool = True,
        debug: bool = False,
    ):
        """Listen for speech and transcribe it."""
        try:
            config = self._build_config(
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

            typer.echo("Starting transcription... (Press Ctrl+C to stop)")
            result = self.transcription_orchestrator.transcribe_single_recording(config)

            if result:
                typer.echo(f"Transcription: {result}")
            else:
                typer.echo("No speech detected.")

        except KeyboardInterrupt:
            typer.echo("\nTranscription stopped.")
        except Exception as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1) from None

    def hotkey(
        self,
        model: str | None = None,
        language: str | None = None,
        key: str = "f9",
        mode: str = "toggle",
        profile: str | None = None,
        debug: bool = False,
    ):
        """Start hotkey mode for speech transcription."""
        try:
            config = self._build_config(
                model, language, None, 0.0, profile, debug=debug
            )
            config.key = key

            typer.echo(f"Hotkey mode started. Press {key} to record.")
            typer.echo("Press Esc to quit.")

            # This would normally start the hotkey listener
            # For now, we'll just show the configuration
            typer.echo("Hotkey mode would run here with:")
            typer.echo(f"  Model: {config.model_name}")
            typer.echo(f"  Key: {config.key}")
            typer.echo(f"  Mode: {config.mode.value}")

        except Exception as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1) from None

    def daemon_start(self):
        """Start the whisper daemon."""
        try:
            config = self.config_repo.load()
            self.daemon_service.start(config)
            typer.echo("Daemon started successfully.")
        except RuntimeError as e:
            # Check if it's the "already running" error and handle it gracefully
            if "already running" in str(e):
                typer.echo("Daemon is already running")
                return
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1) from None

    def daemon_stop(self):
        """Stop the whisper daemon."""
        try:
            self.daemon_service.stop()
            typer.echo("Daemon stopped successfully.")
        except RuntimeError as e:
            # Check if it's the "not running" error and handle it gracefully
            if "not running" in str(e):
                typer.echo("Daemon is not running")
                return
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1) from None

    def daemon_status(self):
        """Show daemon status."""
        status = self.daemon_service.get_status()

        if status["running"]:
            if "pid" in status:
                typer.echo(f"Daemon is running (PID: {status['pid']})")
            else:
                typer.echo("Daemon is running")
            if "uptime" in status and status["uptime"] != "unknown":
                typer.echo(f"  Uptime: {status['uptime']}")

            # Always show configuration when daemon is running
            config = self.config_repo.load()
            typer.echo("Configuration:")
            config_dict = config.to_dict()
            for key, value in config_dict.items():
                typer.echo(f"  {key}: {value}")
        else:
            typer.echo("Daemon is not running")

    def config_show(self):
        """Show current configuration."""
        config = self.config_repo.load()
        typer.echo("Current configuration:")
        for key, value in config.to_dict().items():
            typer.echo(f"  {key}: {value}")

    def config_set(self, key: str, value: str):
        """Set a configuration value."""
        config = self.config_repo.load()
        config_dict = config.to_dict()

        if key not in config_dict:
            typer.echo(f"Unknown configuration key: {key}", err=True)
            raise typer.Exit(1)

        # Convert string value to appropriate type
        original_type = type(config_dict[key])
        try:
            if original_type is bool:
                value = value.lower() in ("true", "1", "yes", "on")
            elif original_type is float:
                value = float(value)
            elif original_type is int:
                value = int(value)
        except ValueError:
            typer.echo(
                f"Invalid value '{value}' for {key}. Must be a {original_type.__name__}.",
                err=True,
            )
            raise typer.Exit(1) from None

        # Update config
        setattr(config, key, value)
        self.config_repo.save(config)
        typer.echo(f"Set {key} = {value}")

    def profile_save(self, name: str):
        """Save current configuration as a profile."""
        config = self.config_repo.load()
        self.profile_repo.save_profile(name, config)
        typer.echo(f"Profile '{name}' saved.")

    def profile_load(self, name: str):
        """Load a configuration profile."""
        try:
            config = self.profile_repo.load_profile(name)
            self.config_repo.save(config)
            typer.echo(f"Profile '{name}' loaded.")
        except FileNotFoundError as e:
            typer.echo(f"Profile '{name}' not found.", err=True)
            raise typer.Exit(1) from e

    def profile_list(self):
        """List all profiles."""
        profiles = self.profile_repo.list_profiles()
        if profiles:
            typer.echo("Available profiles:")
            for profile in profiles:
                typer.echo(f"  {profile}")
        else:
            typer.echo("No profiles found.")

    def profile_delete(self, name: str):
        """Delete a profile."""
        if self.profile_repo.delete_profile(name):
            typer.echo(f"Profile '{name}' deleted.")
        else:
            typer.echo(f"Profile '{name}' not found.", err=True)
            raise typer.Exit(1)

    def _build_config(
        self,
        model: str | None = None,
        language: str | None = None,
        initial_prompt: str | None = None,
        temperature: float = 0.0,
        profile: str | None = None,
        paste_stream: bool = False,
        copy_stream: bool = False,
        paste_final: bool = False,
        copy_final: bool = True,
        debug: bool = False,
    ) -> WhisperConfig:
        # Start with saved config or profile
        if profile:
            try:
                config = self.profile_repo.load_profile(profile)
            except FileNotFoundError:
                typer.echo(
                    f"Profile '{profile}' not found, using default config.", err=True
                )
                config = self.config_repo.load()
        else:
            config = self.config_repo.load()

        # Override with command line parameters
        if model:
            config.model_name = model
        if language:
            config.language = language
        if initial_prompt:
            config.initial_prompt = initial_prompt
        if temperature != 0.0:
            config.temperature = temperature

        config.paste_stream = paste_stream
        config.copy_stream = copy_stream
        config.paste_final = paste_final
        config.copy_final = copy_final
        config.debug = debug

        return config

    # GPU and Performance Commands
    def gpu_status(self):
        """Show GPU device status and capabilities."""
        gpu_devices = self.system_service.detect_gpu_devices()

        typer.echo("GPU Device Status:")
        for i, gpu in enumerate(gpu_devices):
            typer.echo(f"  Device {i}:")
            typer.echo(f"    Type: {gpu.gpu_type.value}")
            typer.echo(f"    Name: {gpu.device_name}")
            typer.echo(f"    Memory: {gpu.memory_available}MB / {gpu.memory_total}MB")
            if gpu.compute_capability:
                typer.echo(f"    Compute Capability: {gpu.compute_capability}")

    def gpu_benchmark(self, model: str = "base"):
        """Benchmark GPU performance with a model."""
        if not self.performance_service:
            typer.echo("Performance service not available", err=True)
            raise typer.Exit(1)

        typer.echo(f"Benchmarking model '{model}' with GPU acceleration...")

        try:
            # Benchmark with GPU
            gpu_results = self.performance_service.benchmark_model(model, use_gpu=True)
            typer.echo("GPU Results:")
            self._display_benchmark_results(gpu_results)

            # Benchmark with CPU for comparison
            cpu_results = self.performance_service.benchmark_model(model, use_gpu=False)
            typer.echo("\nCPU Results:")
            self._display_benchmark_results(cpu_results)

        except Exception as e:
            typer.echo(f"Benchmark failed: {e}", err=True)
            raise typer.Exit(1) from e

    def performance_stats(self):
        """Show performance statistics."""
        if not self.performance_service:
            typer.echo("Performance service not available", err=True)
            raise typer.Exit(1)

        stats = self.performance_service.get_performance_stats()

        typer.echo("Performance Statistics:")
        typer.echo(f"  Total Operations: {stats['total_operations']}")

        if "system" in stats:
            typer.echo(
                f"  Current Memory Usage: {stats['system']['current_memory_mb']:.1f}MB ({stats['system']['memory_percentage']:.1f}%)"
            )
            typer.echo(f"  Available GPUs: {stats['system']['available_gpus']}")

        if "operations" in stats:
            for operation, op_stats in stats["operations"].items():
                typer.echo(f"\n  {operation.title()}:")
                typer.echo(f"    Count: {op_stats['count']}")
                typer.echo(f"    Avg Duration: {op_stats['avg_duration']:.2f}s")
                typer.echo(
                    f"    GPU Operations: {op_stats['gpu_operations']} ({op_stats['gpu_percentage']:.1f}%)"
                )

                if "avg_memory_mb" in op_stats:
                    typer.echo(f"    Avg Memory: {op_stats['avg_memory_mb']:.1f}MB")

                if "avg_speed_ratio" in op_stats:
                    typer.echo(
                        f"    Avg Speed Ratio: {op_stats['avg_speed_ratio']:.2f}x"
                    )

    # Session Management Commands
    def sessions_list(self):
        """List all transcription sessions."""
        if not self.session_service:
            typer.echo("Session service not available", err=True)
            raise typer.Exit(1)

        sessions = self.session_service.list_sessions()

        if not sessions:
            typer.echo("No sessions found.")
            return

        typer.echo("Transcription Sessions:")
        for session in sessions:
            status = "✓ Completed" if session.is_completed else "⏸ In Progress"
            progress = f"{session.progress_seconds:.1f}s"
            if session.total_duration > 0:
                percent = (session.progress_seconds / session.total_duration) * 100
                progress += f" ({percent:.1f}%)"

            typer.echo(
                f"  {session.session_id[:8]}... - {session.session_name or 'Unnamed'}"
            )
            typer.echo(f"    Status: {status}")
            typer.echo(f"    Progress: {progress}")
            typer.echo(
                f"    Created: {session.created_at.strftime('%Y-%m-%d %H:%M:%S')}"
            )

    def sessions_resume(
        self, session_id: str | None = None, session_name: str | None = None
    ):
        """Resume a transcription session."""
        if not self.resume_service or not self.session_service:
            typer.echo("Resume service not available", err=True)
            raise typer.Exit(1)

        # Find session by ID or name
        if session_name:
            session = self.session_service.find_session_by_name(session_name)
            if not session:
                typer.echo(f"Session '{session_name}' not found", err=True)
                raise typer.Exit(1)
            session_id = session.session_id
        elif session_id:
            try:
                session = self.session_service.load_session(session_id)
            except FileNotFoundError as e:
                typer.echo(f"Session '{session_id}' not found", err=True)
                raise typer.Exit(1) from e
        else:
            typer.echo("Must specify either --session-id or --session-name", err=True)
            raise typer.Exit(1)

        config = self.config_repo.load()

        try:
            typer.echo(f"Resuming transcription session: {session_id[:8]}...")

            results = list(self.resume_service.resume_transcription(session_id, config))

            if results:
                final_text = " ".join(result.text for result in results)
                typer.echo(f"Resumed transcription: {final_text}")
            else:
                typer.echo("No additional content transcribed.")

        except Exception as e:
            typer.echo(f"Resume failed: {e}", err=True)
            raise typer.Exit(1) from e

    def sessions_cleanup(self):
        """Clean up completed sessions."""
        if not self.session_service:
            typer.echo("Session service not available", err=True)
            raise typer.Exit(1)

        count = self.session_service.cleanup_completed_sessions()
        typer.echo(f"Cleaned up {count} completed sessions.")

    def sessions_delete(self, session_id: str):
        """Delete a specific session."""
        if not self.session_service:
            typer.echo("Session service not available", err=True)
            raise typer.Exit(1)

        if self.session_service.delete_session(session_id):
            typer.echo(f"Session {session_id} deleted.")
        else:
            typer.echo(f"Session {session_id} not found.", err=True)
            raise typer.Exit(1)

    def listen_resumable(
        self,
        audio_file: str,
        session_name: str | None = None,
        model: str | None = None,
        language: str | None = None,
        chunk_size: int = 30,
        max_memory: int = 1024,
    ):
        """Start a resumable transcription of an audio file."""
        if not self.resume_service:
            typer.echo("Resume service not available", err=True)
            raise typer.Exit(1)

        if not Path(audio_file).exists():
            typer.echo(f"Audio file not found: {audio_file}", err=True)
            raise typer.Exit(1)

        config = self.config_repo.load()
        if model:
            config.model_name = model
        if language:
            config.language = language
        config.chunk_size = chunk_size
        config.max_memory_mb = max_memory

        try:
            typer.echo(f"Starting resumable transcription: {audio_file}")
            if session_name:
                typer.echo(f"Session name: {session_name}")

            results = list(
                self.resume_service.create_resumable_transcription(
                    audio_file, config, session_name
                )
            )

            if results:
                final_text = " ".join(result.text for result in results)
                typer.echo(f"Transcription completed: {final_text}")
            else:
                typer.echo("No content transcribed.")

        except Exception as e:
            typer.echo(f"Transcription failed: {e}", err=True)
            raise typer.Exit(1) from e

    def _display_benchmark_results(self, results: dict):
        """Display benchmark results in a formatted way."""
        if "error" in results:
            typer.echo(f"  Error: {results['error']}")
            return

        if "results" in results:
            r = results["results"]
            typer.echo(f"  Model Load Time: {r.get('model_load_time', 0):.2f}s")
            typer.echo(f"  Inference Time: {r.get('inference_time', 0):.2f}s")
            typer.echo(f"  Memory Usage: {r.get('memory_usage_mb', 0):.1f}MB")
            if r.get("gpu_memory_mb"):
                typer.echo(f"  GPU Memory: {r['gpu_memory_mb']:.1f}MB")
            typer.echo(f"  Device: {r.get('device_used', 'unknown')}")

    # Vocabulary Management Commands
    def vocabulary_add(
        self,
        word: str,
        vocabulary_type: str = "custom",
        domain: str = None,
        profile: str = "default",
    ):
        if not self.vocabulary_service:
            typer.echo("Vocabulary service not available", err=True)
            raise typer.Exit(1)

        from adapters.vocabulary import VocabularyAdapter

        adapter = VocabularyAdapter()
        config = adapter.load_vocabulary_config(profile)

        if vocabulary_type == "custom":
            if word not in config.custom_words:
                config.custom_words.append(word)
        elif vocabulary_type == "proper_nouns":
            if word not in config.proper_nouns:
                config.proper_nouns.append(word)
        elif vocabulary_type == "technical":
            if word not in config.technical_jargon:
                config.technical_jargon.append(word)
        elif vocabulary_type == "domain":
            if not domain:
                typer.echo("Domain name required for domain vocabulary type", err=True)
                raise typer.Exit(1)
            if domain not in config.domain_terms:
                config.domain_terms[domain] = []
            if word not in config.domain_terms[domain]:
                config.domain_terms[domain].append(word)
        else:
            typer.echo(f"Unknown vocabulary type: {vocabulary_type}", err=True)
            raise typer.Exit(1)

        adapter.save_vocabulary_config(config, profile)
        typer.echo(
            f"Added '{word}' to {vocabulary_type} vocabulary (profile: {profile})"
        )

    def vocabulary_remove(
        self,
        word: str,
        vocabulary_type: str = "custom",
        domain: str = None,
        profile: str = "default",
    ):
        if not self.vocabulary_service:
            typer.echo("Vocabulary service not available", err=True)
            raise typer.Exit(1)

        from adapters.vocabulary import VocabularyAdapter

        adapter = VocabularyAdapter()
        config = adapter.load_vocabulary_config(profile)

        removed = False
        if vocabulary_type == "custom" and word in config.custom_words:
            config.custom_words.remove(word)
            removed = True
        elif vocabulary_type == "proper_nouns" and word in config.proper_nouns:
            config.proper_nouns.remove(word)
            removed = True
        elif vocabulary_type == "technical" and word in config.technical_jargon:
            config.technical_jargon.remove(word)
            removed = True
        elif vocabulary_type == "domain" and domain and domain in config.domain_terms:
            if word in config.domain_terms[domain]:
                config.domain_terms[domain].remove(word)
                removed = True

        if removed:
            adapter.save_vocabulary_config(config, profile)
            typer.echo(
                f"Removed '{word}' from {vocabulary_type} vocabulary (profile: {profile})"
            )
        else:
            typer.echo(f"Word '{word}' not found in {vocabulary_type} vocabulary")

    def vocabulary_list(self, vocabulary_type: str = None, profile: str = "default"):
        if not self.vocabulary_service:
            typer.echo("Vocabulary service not available", err=True)
            raise typer.Exit(1)

        from adapters.vocabulary import VocabularyAdapter

        adapter = VocabularyAdapter()
        config = adapter.load_vocabulary_config(profile)

        typer.echo(f"Vocabulary for profile: {profile}")

        if not vocabulary_type or vocabulary_type == "custom":
            typer.echo(f"Custom words ({len(config.custom_words)}):")
            for word in sorted(config.custom_words):
                typer.echo(f"  - {word}")

        if not vocabulary_type or vocabulary_type == "proper_nouns":
            typer.echo(f"Proper nouns ({len(config.proper_nouns)}):")
            for word in sorted(config.proper_nouns):
                typer.echo(f"  - {word}")

        if not vocabulary_type or vocabulary_type == "technical":
            typer.echo(f"Technical jargon ({len(config.technical_jargon)}):")
            for word in sorted(config.technical_jargon):
                typer.echo(f"  - {word}")

        if not vocabulary_type or vocabulary_type == "domain":
            typer.echo("Domain terms:")
            for domain, terms in config.domain_terms.items():
                typer.echo(f"  {domain} ({len(terms)}):")
                for word in sorted(terms):
                    typer.echo(f"    - {word}")

    def vocabulary_import(
        self, file_path: str, vocabulary_type: str = "custom", profile: str = "default"
    ):
        if not self.vocabulary_service:
            typer.echo("Vocabulary service not available", err=True)
            raise typer.Exit(1)

        from adapters.vocabulary import VocabularyAdapter

        adapter = VocabularyAdapter()
        words = adapter.import_vocabulary_from_file(file_path, vocabulary_type)

        if not words:
            typer.echo(f"No words imported from {file_path}", err=True)
            raise typer.Exit(1)

        config = adapter.load_vocabulary_config(profile)

        if vocabulary_type == "custom":
            config.custom_words.extend(
                word for word in words if word not in config.custom_words
            )
        elif vocabulary_type == "proper_nouns":
            config.proper_nouns.extend(
                word for word in words if word not in config.proper_nouns
            )
        elif vocabulary_type == "technical":
            config.technical_jargon.extend(
                word for word in words if word not in config.technical_jargon
            )

        adapter.save_vocabulary_config(config, profile)
        typer.echo(
            f"Imported {len(words)} words from {file_path} to {vocabulary_type} vocabulary"
        )

    def vocabulary_export(self, file_path: str, profile: str = "default"):
        if not self.vocabulary_service:
            typer.echo("Vocabulary service not available", err=True)
            raise typer.Exit(1)

        from adapters.vocabulary import VocabularyAdapter

        adapter = VocabularyAdapter()
        config = adapter.load_vocabulary_config(profile)
        adapter.export_vocabulary_to_file(config, file_path)
        typer.echo(f"Exported vocabulary to {file_path}")

    # Post-processing Commands
    def postprocessing_config(
        self,
        show: bool = False,
        enable_punctuation: bool = None,
        enable_capitalization: bool = None,
        enable_profanity_filter: bool = None,
        remove_filler_words: bool = None,
        profile: str = "default",
    ):
        if not self.postprocessing_service:
            typer.echo("Post-processing service not available", err=True)
            raise typer.Exit(1)

        import json
        from pathlib import Path

        from domain.models import PostProcessingConfig

        config_dir = Path.home() / ".config" / "whisper-cli" / "postprocessing"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / f"{profile}.json"

        # Load existing config
        if config_file.exists():
            with open(config_file) as f:
                data = json.load(f)
                config = PostProcessingConfig(**data)
        else:
            config = PostProcessingConfig()

        if show:
            typer.echo(f"Post-processing configuration for profile: {profile}")
            typer.echo(f"  Punctuation cleanup: {config.enable_punctuation_cleanup}")
            typer.echo(f"  Capitalization: {config.enable_capitalization}")
            typer.echo(f"  Profanity filter: {config.enable_profanity_filter}")
            typer.echo(f"  Remove filler words: {config.remove_filler_words}")
            typer.echo(f"  Sentence segmentation: {config.sentence_segmentation}")
            typer.echo(f"  Text normalization: {config.text_normalization}")
            return

        # Update config
        updated = False
        if enable_punctuation is not None:
            config.enable_punctuation_cleanup = enable_punctuation
            updated = True
        if enable_capitalization is not None:
            config.enable_capitalization = enable_capitalization
            updated = True
        if enable_profanity_filter is not None:
            config.enable_profanity_filter = enable_profanity_filter
            updated = True
        if remove_filler_words is not None:
            config.remove_filler_words = remove_filler_words
            updated = True

        if updated:
            data = {
                "enable_punctuation_cleanup": config.enable_punctuation_cleanup,
                "enable_capitalization": config.enable_capitalization,
                "enable_profanity_filter": config.enable_profanity_filter,
                "custom_replacements": config.custom_replacements,
                "sentence_segmentation": config.sentence_segmentation,
                "text_normalization": config.text_normalization,
                "remove_filler_words": config.remove_filler_words,
                "filler_words": config.filler_words,
            }

            with open(config_file, "w") as f:
                json.dump(data, f, indent=2)

            typer.echo(f"Updated post-processing configuration for profile: {profile}")
        else:
            typer.echo("No configuration changes specified")

    def postprocessing_test(self, text: str, profile: str = "default"):
        if not self.postprocessing_service:
            typer.echo("Post-processing service not available", err=True)
            raise typer.Exit(1)

        import json
        from pathlib import Path

        from domain.models import PostProcessingConfig

        config_dir = Path.home() / ".config" / "whisper-cli" / "postprocessing"
        config_file = config_dir / f"{profile}.json"

        if config_file.exists():
            with open(config_file) as f:
                data = json.load(f)
                config = PostProcessingConfig(**data)
        else:
            config = PostProcessingConfig()

        processed_text = self.postprocessing_service.process_text(text, config)

        typer.echo("Original text:")
        typer.echo(f"  {text}")
        typer.echo("Processed text:")
        typer.echo(f"  {processed_text}")

    # Webhook Commands
    def webhook_add(
        self,
        url: str,
        event_types: str = "transcription_complete",
        auth_header: str = None,
    ):
        if not self.webhook_service:
            typer.echo("Webhook service not available", err=True)
            raise typer.Exit(1)

        import json
        from pathlib import Path

        from domain.models import EventType

        # Parse event types
        try:
            events = [EventType(event.strip()) for event in event_types.split(",")]
        except ValueError as e:
            typer.echo(f"Invalid event type: {e}", err=True)
            raise typer.Exit(1) from e

        # Load webhook config
        config_dir = Path.home() / ".config" / "whisper-cli"
        config_dir.mkdir(parents=True, exist_ok=True)
        webhook_file = config_dir / "webhooks.json"

        if webhook_file.exists():
            with open(webhook_file) as f:
                config_data = json.load(f)
        else:
            config_data = {
                "webhook_urls": [],
                "event_types": [],
                "authentication": {},
                "timeout_seconds": 30,
                "retry_attempts": 3,
            }

        # Add webhook
        if url not in config_data["webhook_urls"]:
            config_data["webhook_urls"].append(url)

        if auth_header:
            config_data["authentication"]["Authorization"] = auth_header

        with open(webhook_file, "w") as f:
            json.dump(config_data, f, indent=2)

        self.webhook_service.register_webhook(url, events)
        typer.echo(f"Added webhook: {url}")

    def webhook_remove(self, url: str):
        if not self.webhook_service:
            typer.echo("Webhook service not available", err=True)
            raise typer.Exit(1)

        import json
        from pathlib import Path

        config_dir = Path.home() / ".config" / "whisper-cli"
        webhook_file = config_dir / "webhooks.json"

        if not webhook_file.exists():
            typer.echo("No webhooks configured", err=True)
            raise typer.Exit(1)

        with open(webhook_file) as f:
            config_data = json.load(f)

        if url in config_data["webhook_urls"]:
            config_data["webhook_urls"].remove(url)
            with open(webhook_file, "w") as f:
                json.dump(config_data, f, indent=2)
            typer.echo(f"Removed webhook: {url}")
        else:
            typer.echo(f"Webhook not found: {url}")

    def webhook_list(self):
        if not self.webhook_service:
            typer.echo("Webhook service not available", err=True)
            raise typer.Exit(1)

        import json
        from pathlib import Path

        config_dir = Path.home() / ".config" / "whisper-cli"
        webhook_file = config_dir / "webhooks.json"

        if not webhook_file.exists():
            typer.echo("No webhooks configured")
            return

        with open(webhook_file) as f:
            config_data = json.load(f)

        typer.echo("Configured webhooks:")
        for url in config_data.get("webhook_urls", []):
            typer.echo(f"  - {url}")

    def webhook_test(self, url: str, event_type: str = "transcription_complete"):
        if not self.webhook_service:
            typer.echo("Webhook service not available", err=True)
            raise typer.Exit(1)

        import uuid
        from datetime import datetime

        from domain.models import EventType, WebhookEvent

        try:
            event_enum = EventType(event_type)
        except ValueError as e:
            typer.echo(f"Invalid event type: {event_type}", err=True)
            raise typer.Exit(1) from e

        test_event = WebhookEvent(
            event_type=event_enum,
            timestamp=datetime.now(),
            operation_id=str(uuid.uuid4()),
            data={"test": True, "message": "Test webhook event"},
        )

        success = self.webhook_service.send_webhook(test_event, url)
        if success:
            typer.echo(f"Successfully sent test webhook to {url}")
        else:
            typer.echo(f"Failed to send test webhook to {url}", err=True)

    # API Server Commands
    def api_start(self, port: int = 8000, host: str = "127.0.0.1", workers: int = 1):
        typer.echo(f"Starting API server on {host}:{port} with {workers} workers")
        typer.echo("Note: API server functionality requires additional implementation")

    def api_stop(self):
        typer.echo("Stopping API server")
        typer.echo("Note: API server functionality requires additional implementation")

    def api_status(self):
        typer.echo("API server status: Not implemented")

    # Operations Commands
    def operations_list(self):
        if not self.progress_service:
            typer.echo("Progress service not available", err=True)
            raise typer.Exit(1)

        active_ops = self.progress_service.list_active_operations()

        if not active_ops:
            typer.echo("No active operations")
            return

        typer.echo("Active operations:")
        for tracker in active_ops:
            typer.echo(
                f"  {tracker.operation_id}: {tracker.operation_type} ({tracker.status}) - {tracker.current_progress * 100:.1f}%"
            )

    def operations_cancel(self, operation_id: str):
        if not self.progress_service:
            typer.echo("Progress service not available", err=True)
            raise typer.Exit(1)

        self.progress_service.cancel_operation(operation_id)
        typer.echo(f"Cancelled operation: {operation_id}")

    def operations_status(self, operation_id: str):
        if not self.progress_service:
            typer.echo("Progress service not available", err=True)
            raise typer.Exit(1)

        tracker = self.progress_service.get_tracker(operation_id)
        if not tracker:
            typer.echo(f"Operation not found: {operation_id}", err=True)
            raise typer.Exit(1)

        typer.echo(f"Operation: {operation_id}")
        typer.echo(f"  Type: {tracker.operation_type}")
        typer.echo(f"  Status: {tracker.status}")
        typer.echo(f"  Progress: {tracker.current_progress * 100:.1f}%")
        typer.echo(f"  Current step: {tracker.current_step}")
        if tracker.eta_seconds:
            typer.echo(f"  ETA: {tracker.eta_seconds:.0f} seconds")

    # Circuit Breaker Commands
    def circuit_breaker_status(self):
        if not self.circuit_breaker_service:
            typer.echo("Circuit breaker service not available", err=True)
            raise typer.Exit(1)

        stats = (
            self.circuit_breaker_service.get_stats()
            if hasattr(self.circuit_breaker_service, "get_stats")
            else {}
        )

        if not stats:
            typer.echo("No circuit breaker states available")
            return

        typer.echo("Circuit breaker status:")
        for service_name, state in stats.items():
            typer.echo(
                f"  {service_name}: {state['state']} (failures: {state.get('failure_count', 0)})"
            )

    def circuit_breaker_reset(self, service: str = None):
        if not self.circuit_breaker_service:
            typer.echo("Circuit breaker service not available", err=True)
            raise typer.Exit(1)

        if service:
            self.circuit_breaker_service.reset(service)
            typer.echo(f"Reset circuit breaker for service: {service}")
        else:
            typer.echo("Reset all circuit breakers (not implemented)")

    def circuit_breaker_stats(self):
        if not self.circuit_breaker_service:
            typer.echo("Circuit breaker service not available", err=True)
            raise typer.Exit(1)

        typer.echo("Circuit breaker statistics:")
        typer.echo("  (Statistics functionality requires implementation)")

    def export_transcription(
        self,
        session_id: str,
        format: str = "txt",
        output_file: str | None = None,
        timestamp_mode: str = "sentence",
        include_confidence: bool = True,
        include_speaker_info: bool = True,
    ):
        """Export a transcription session to various formats."""
        if not self.export_service:
            typer.echo("Export service not available", err=True)
            raise typer.Exit(1)

        if not self.session_service:
            typer.echo("Session service not available", err=True)
            raise typer.Exit(1)

        try:
            session = self.session_service.load_session(session_id)
        except Exception as e:
            typer.echo(f"Failed to load session: {e}", err=True)
            raise typer.Exit(1) from e

        try:
            output_format = OutputFormat(format)
        except ValueError as e:
            typer.echo(f"Unsupported format: {format}", err=True)
            typer.echo(f"Supported formats: {[f.value for f in OutputFormat]}")
            raise typer.Exit(1) from e

        try:
            ts_mode = TimestampMode(timestamp_mode)
        except ValueError as e:
            typer.echo(f"Unsupported timestamp mode: {timestamp_mode}", err=True)
            typer.echo(f"Supported modes: {[m.value for m in TimestampMode]}")
            raise typer.Exit(1) from e

        config = ExportConfig(
            format=output_format,
            timestamp_mode=ts_mode,
            include_confidence=include_confidence,
            include_speaker_info=include_speaker_info,
            output_file=output_file,
        )

        # Create a mock transcription result from session
        from domain.models import TranscriptionResult

        result = TranscriptionResult(
            text=session.transcribed_text,
            duration=session.total_duration,
            language="en",  # Default, should be detected
        )

        try:
            if output_file:
                success = self.export_service.export_to_file(result, config)
                if success:
                    typer.echo(f"Exported to: {output_file}")
                else:
                    typer.echo("Export failed", err=True)
                    raise typer.Exit(1)
            else:
                content = self.export_service.export_transcription(result, config)
                typer.echo(content)

        except Exception as e:
            typer.echo(f"Export failed: {e}", err=True)
            raise typer.Exit(1) from e

    def list_export_formats(self):
        """List all supported export formats."""
        if not self.export_service:
            typer.echo("Export service not available", err=True)
            raise typer.Exit(1)

        formats = self.export_service.get_supported_formats()
        typer.echo("Supported export formats:")
        for fmt in formats:
            typer.echo(f"  - {fmt.value}")

        typer.echo("\nSupported timestamp modes:")
        for mode in TimestampMode:
            typer.echo(f"  - {mode.value}")

    def batch_export_sessions(
        self,
        format: str = "txt",
        output_dir: str = "./exports",
        timestamp_mode: str = "sentence",
        include_confidence: bool = True,
        include_speaker_info: bool = True,
    ):
        """Export all sessions to specified format."""
        if not self.export_service:
            typer.echo("Export service not available", err=True)
            raise typer.Exit(1)

        if not self.session_service:
            typer.echo("Session service not available", err=True)
            raise typer.Exit(1)

        try:
            output_format = OutputFormat(format)
            ts_mode = TimestampMode(timestamp_mode)
        except ValueError as e:
            typer.echo(f"Invalid parameter: {e}", err=True)
            raise typer.Exit(1) from e

        sessions = self.session_service.list_sessions()
        if not sessions:
            typer.echo("No sessions found")
            return

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        exported_count = 0
        for session in sessions:
            if not session.transcribed_text:
                continue

            filename = f"session_{session.session_id}.{format}"
            output_file = output_path / filename

            config = ExportConfig(
                format=output_format,
                timestamp_mode=ts_mode,
                include_confidence=include_confidence,
                include_speaker_info=include_speaker_info,
                output_file=str(output_file),
            )

            from domain.models import TranscriptionResult

            result = TranscriptionResult(
                text=session.transcribed_text,
                duration=session.total_duration,
                language="en",
            )

            try:
                success = self.export_service.export_to_file(result, config)
                if success:
                    exported_count += 1
                    typer.echo(f"Exported: {filename}")
            except Exception as e:
                typer.echo(f"Failed to export {session.session_id}: {e}", err=True)

        typer.echo(
            f"Exported {exported_count}/{len(sessions)} sessions to {output_dir}"
        )

    def analyze_confidence(
        self,
        session_id: str,
        detailed: bool = False,
        flag_threshold: float = 0.6,
    ):
        """Analyze transcription confidence and quality."""
        if not self.confidence_analyzer:
            typer.echo("Confidence analyzer not available", err=True)
            raise typer.Exit(1)

        if not self.session_service:
            typer.echo("Session service not available", err=True)
            raise typer.Exit(1)

        try:
            session = self.session_service.load_session(session_id)
        except Exception as e:
            typer.echo(f"Failed to load session: {e}", err=True)
            raise typer.Exit(1) from e

        # Create mock transcription result for analysis
        from domain.models import TranscriptionResult

        result = TranscriptionResult(
            text=session.transcribed_text,
            duration=session.total_duration,
            language="en",
        )

        analysis = self.confidence_analyzer.analyze_confidence(result)
        summary = self.confidence_analyzer.get_review_summary(analysis)
        typer.echo(summary)

        if detailed and analysis.quality_flags:
            typer.echo("\nDetailed Issues:")
            for i, flag in enumerate(analysis.quality_flags, 1):
                typer.echo(f"{i}. {flag.description} (severity: {flag.severity.value})")

    def analyze_all_sessions(self, detailed: bool = False, threshold: float = 0.7):
        """Analyze confidence for all sessions."""
        if not self.confidence_analyzer:
            typer.echo("Confidence analyzer not available", err=True)
            raise typer.Exit(1)

        if not self.session_service:
            typer.echo("Session service not available", err=True)
            raise typer.Exit(1)

        sessions = self.session_service.list_sessions()
        if not sessions:
            typer.echo("No sessions found")
            return

        low_confidence_sessions = []

        for session in sessions:
            if not session.transcribed_text:
                continue

            from domain.models import TranscriptionResult

            result = TranscriptionResult(
                text=session.transcribed_text,
                duration=session.total_duration,
                language="en",
            )

            analysis = self.confidence_analyzer.analyze_confidence(result)

            if analysis.overall_confidence < threshold:
                low_confidence_sessions.append((session, analysis))

            if detailed:
                typer.echo(f"\nSession {session.session_id}:")
                typer.echo(f"  Confidence: {analysis.overall_confidence:.2%}")
                typer.echo(f"  Issues: {len(analysis.quality_flags)}")
                if analysis.segments_needing_review:
                    typer.echo(
                        f"  Segments needing review: {len(analysis.segments_needing_review)}"
                    )

        typer.echo(
            f"\nSummary: {len(low_confidence_sessions)}/{len(sessions)} sessions below {threshold:.0%} confidence"
        )

        if low_confidence_sessions:
            typer.echo("\nSessions needing review:")
            for session, analysis in low_confidence_sessions:
                typer.echo(
                    f"  - {session.session_id}: {analysis.overall_confidence:.2%}"
                )

    def set_confidence_thresholds(
        self,
        high_threshold: float = 0.9,
        medium_threshold: float = 0.7,
        low_threshold: float = 0.5,
        review_threshold: float = 0.6,
    ):
        """Configure confidence analysis thresholds."""
        if not self.confidence_analyzer:
            typer.echo("Confidence analyzer not available", err=True)
            raise typer.Exit(1)

        self.confidence_analyzer.high_threshold = high_threshold
        self.confidence_analyzer.medium_threshold = medium_threshold
        self.confidence_analyzer.low_threshold = low_threshold
        self.confidence_analyzer.review_threshold = review_threshold

        typer.echo("Confidence thresholds updated:")
        typer.echo(f"  High: {high_threshold:.1%}")
        typer.echo(f"  Medium: {medium_threshold:.1%}")
        typer.echo(f"  Low: {low_threshold:.1%}")
        typer.echo(f"  Review: {review_threshold:.1%}")

    # Audio Processing Commands

    def transcribe_file(
        self,
        file_path: str,
        output_path: str | None = None,
        model: str | None = None,
        language: str | None = None,
        temperature: float = 0.0,
        format_output: str = "txt",
    ):
        """Transcribe an audio file (supports MP3, WAV, M4A, FLAC, OGG)."""
        if not self.audio_format_service:
            typer.echo("Audio format service not available", err=True)
            raise typer.Exit(1)

        input_file = Path(file_path)
        if not input_file.exists():
            typer.echo(f"File not found: {file_path}", err=True)
            raise typer.Exit(1)

        if not self.audio_format_service.is_supported_format(input_file):
            typer.echo(f"Unsupported file format: {input_file.suffix}", err=True)
            supported = ", ".join(self.audio_format_service.get_supported_formats())
            typer.echo(f"Supported formats: {supported}")
            raise typer.Exit(1)

        # Load config and override with parameters
        config = self.config_repo.load()
        if model:
            config.model_name = model
        if language:
            config.language = language
        config.temperature = temperature

        # Convert to WAV if needed
        temp_wav = None
        try:
            if input_file.suffix.lower() != ".wav":
                temp_wav = input_file.parent / f"temp_{input_file.stem}.wav"
                typer.echo(f"Converting {input_file.name} to WAV...")

                if not self.audio_format_service.convert_to_wav(input_file, temp_wav):
                    typer.echo("Failed to convert audio file", err=True)
                    raise typer.Exit(1)

                audio_file = temp_wav
            else:
                audio_file = input_file

            # Read and transcribe
            with open(audio_file, "rb") as f:
                audio_data = f.read()

            typer.echo("Transcribing...")
            result = self.transcription_orchestrator.transcription_service.transcribe(
                audio_data, config
            )

            # Output result
            output_text = result.text.strip()
            if output_path:
                output_file = Path(output_path)
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(output_text)
                typer.echo(f"Transcription saved to: {output_file}")
            else:
                typer.echo(output_text)

        finally:
            # Clean up temp file
            if temp_wav and temp_wav.exists():
                temp_wav.unlink()

    def batch_transcribe(
        self,
        input_dir: str,
        output_dir: str = "transcriptions",
        workers: int = 4,
        file_pattern: str | None = None,
        model: str | None = None,
    ):
        """Batch transcribe all audio files in a directory."""
        if not self.batch_processing_service:
            typer.echo("Batch processing service not available", err=True)
            raise typer.Exit(1)

        input_path = Path(input_dir)
        output_path = Path(output_dir)

        if not input_path.exists():
            typer.echo(f"Input directory not found: {input_dir}", err=True)
            raise typer.Exit(1)

        # Load config
        config = self.config_repo.load()
        if model:
            config.model_name = model

        # Set file patterns
        patterns = [file_pattern] if file_pattern else None

        typer.echo(f"Processing directory: {input_path}")
        typer.echo(f"Output directory: {output_path}")
        typer.echo(f"Workers: {workers}")

        # Estimate processing time
        files = self.batch_processing_service.get_processable_files(
            input_path, patterns
        )
        estimated_time = self.batch_processing_service.estimate_batch_time(files)

        typer.echo(f"Found {len(files)} files to process")
        typer.echo(f"Estimated time: {estimated_time / 60:.1f} minutes")

        # Process files
        results = self.batch_processing_service.process_directory(
            input_path, output_path, config, workers, patterns
        )

        # Summary
        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful

        typer.echo("\nBatch processing complete:")
        typer.echo(f"  Successful: {successful}")
        typer.echo(f"  Failed: {failed}")
        typer.echo(f"  Output directory: {output_path}")

    def split_audio(
        self,
        file_path: str,
        output_dir: str = "split_audio",
        method: str = "duration",
        chunk_duration: int = 300,
        silence_threshold: float = 0.01,
        max_size_mb: float = 25.0,
    ):
        """Split audio file into chunks."""
        if not self.audio_splitting_service:
            typer.echo("Audio splitting service not available", err=True)
            raise typer.Exit(1)

        input_file = Path(file_path)
        output_path = Path(output_dir)

        if not input_file.exists():
            typer.echo(f"File not found: {file_path}", err=True)
            raise typer.Exit(1)

        typer.echo(f"Splitting {input_file.name} by {method}...")

        try:
            if method == "duration":
                chunks = self.audio_splitting_service.split_by_duration(
                    input_file, chunk_duration, output_path
                )
            elif method == "silence":
                chunks = self.audio_splitting_service.split_by_silence(
                    input_file, silence_threshold, output_path
                )
            elif method == "size":
                chunks = self.audio_splitting_service.split_by_size(
                    input_file, max_size_mb, output_path
                )
            else:
                typer.echo(f"Unknown split method: {method}", err=True)
                typer.echo("Available methods: duration, silence, size")
                raise typer.Exit(1)

            typer.echo(f"Created {len(chunks)} chunks:")
            for chunk in chunks:
                typer.echo(f"  {chunk.name}")

        except Exception as e:
            typer.echo(f"Error splitting audio: {e}", err=True)
            raise typer.Exit(1) from e

    def preprocess_audio(
        self,
        input_path: str,
        output_path: str,
        noise_reduction: float | None = None,
        normalize_volume: float | None = None,
        trim_silence: bool = False,
        enhance_speech: bool = False,
    ):
        """Preprocess audio file (noise reduction, normalization, etc.)."""
        if not self.audio_preprocessing_service:
            typer.echo("Audio preprocessing service not available", err=True)
            raise typer.Exit(1)

        input_file = Path(input_path)
        output_file = Path(output_path)

        if not input_file.exists():
            typer.echo(f"Input file not found: {input_path}", err=True)
            raise typer.Exit(1)

        try:
            if enhance_speech:
                typer.echo("Applying speech enhancement...")
                success = self.audio_preprocessing_service.enhance_speech(
                    input_file, output_file
                )
            else:
                # Apply individual filters
                current_input = input_file
                temp_files = []

                if noise_reduction is not None:
                    typer.echo(
                        f"Applying noise reduction (strength: {noise_reduction})..."
                    )
                    temp_output = output_file.parent / f"temp_nr_{output_file.name}"
                    if not self.audio_preprocessing_service.reduce_noise(
                        current_input, temp_output, noise_reduction
                    ):
                        raise RuntimeError("Noise reduction failed")
                    temp_files.append(temp_output)
                    current_input = temp_output

                if normalize_volume is not None:
                    typer.echo(f"Normalizing volume to {normalize_volume} dB...")
                    temp_output = output_file.parent / f"temp_norm_{output_file.name}"
                    if not self.audio_preprocessing_service.normalize_volume(
                        current_input, temp_output, normalize_volume
                    ):
                        raise RuntimeError("Volume normalization failed")
                    temp_files.append(temp_output)
                    current_input = temp_output

                if trim_silence:
                    typer.echo("Trimming silence...")
                    temp_output = output_file.parent / f"temp_trim_{output_file.name}"
                    if not self.audio_preprocessing_service.trim_silence(
                        current_input, temp_output
                    ):
                        raise RuntimeError("Silence trimming failed")
                    temp_files.append(temp_output)
                    current_input = temp_output

                # Move final result to output path
                if current_input != input_file:
                    current_input.rename(output_file)
                    temp_files.remove(current_input)
                    success = True
                else:
                    # No processing applied, copy file
                    import shutil

                    shutil.copy2(input_file, output_file)
                    success = True

                # Clean up temp files
                for temp_file in temp_files:
                    if temp_file.exists():
                        temp_file.unlink()

            if success:
                typer.echo(f"Preprocessed audio saved to: {output_file}")
            else:
                typer.echo("Audio preprocessing failed", err=True)
                raise typer.Exit(1)

        except Exception as e:
            typer.echo(f"Error preprocessing audio: {e}", err=True)
            raise typer.Exit(1) from e

    def show_audio_info(self, file_path: str):
        """Show detailed information about an audio file."""
        if not self.audio_format_service:
            typer.echo("Audio format service not available", err=True)
            raise typer.Exit(1)

        audio_file = Path(file_path)
        if not audio_file.exists():
            typer.echo(f"File not found: {file_path}", err=True)
            raise typer.Exit(1)

        info = self.audio_format_service.get_audio_info(audio_file)

        if not info:
            typer.echo("Could not read audio file information", err=True)
            raise typer.Exit(1)

        typer.echo(f"Audio file: {audio_file.name}")
        typer.echo(f"Format: {info.get('format', 'Unknown')}")
        typer.echo(f"Duration: {info.get('duration', 0):.2f} seconds")
        typer.echo(f"Sample rate: {info.get('sample_rate', 0)} Hz")
        typer.echo(f"Channels: {info.get('channels', 0)}")
        typer.echo(f"Bitrate: {info.get('bitrate', 0)} bps")
        typer.echo(f"Size: {info.get('size', 0) / (1024 * 1024):.2f} MB")
        if info.get("codec"):
            typer.echo(f"Codec: {info.get('codec')}")

    def list_audio_formats(self):
        """List supported audio formats."""
        if not self.audio_format_service:
            typer.echo("Audio format service not available", err=True)
            raise typer.Exit(1)

        formats = self.audio_format_service.get_supported_formats()
        typer.echo("Supported audio formats:")
        for fmt in formats:
            typer.echo(f"  .{fmt}")

    def realtime_transcribe(
        self,
        chunk_duration: float = 2.0,
        overlap: float = 0.5,
        vad_threshold: float = 0.01,
        output_format: str = "live",
        model: str | None = None,
    ):
        """Real-time transcription with configurable parameters."""
        from services.realtime_transcription import RealtimeTranscriptionService

        # Load config
        config = self.config_repo.load()
        if model:
            config.model_name = model

        # Create realtime service
        realtime_service = RealtimeTranscriptionService(
            self.transcription_orchestrator.audio_recorder,
            self.transcription_orchestrator.transcription_service,
            self.logger,
        )

        typer.echo("Starting real-time transcription...")
        typer.echo(f"Chunk duration: {chunk_duration}s")
        typer.echo(f"VAD threshold: {vad_threshold}")
        typer.echo(f"Output format: {output_format}")
        typer.echo("Press Ctrl+C to stop")

        try:
            for result in realtime_service.transcribe_realtime(
                config, chunk_duration, overlap, vad_threshold, output_format
            ):
                if result:
                    if output_format == "live":
                        typer.echo(f"\r{result['text']}", nl=False)
                    else:
                        typer.echo(result["text"])

        except KeyboardInterrupt:
            typer.echo("\nStopping real-time transcription...")
        finally:
            realtime_service.stop()

    # TTS Commands

    def tts_generate_from_text(
        self,
        text: str,
        voice: str | None = None,
        streaming: bool = False,
        output_file: str | None = None,
        auto_play: bool = True,
        cfg_scale: float | None = None,
        inference_steps: int | None = None,
        sample_rate: int | None = None,
        use_gpu: bool | None = None,
    ):
        """Generate TTS from provided text"""
        if not self.tts_orchestrator:
            typer.echo("TTS not available", err=True)
            raise typer.Exit(1)

        config = self.config_repo.load()
        tts_config = config.tts_config

        # Override config with command parameters
        if voice:
            tts_config.default_voice = voice
        if output_file:
            tts_config.output_file_path = output_file
            tts_config.output_mode = (
                TTSOutputMode.SAVE_FILE if not auto_play else TTSOutputMode.BOTH
            )
        if cfg_scale is not None:
            tts_config.cfg_scale = cfg_scale
        if inference_steps is not None:
            tts_config.inference_steps = inference_steps
        if sample_rate is not None:
            tts_config.sample_rate = sample_rate
        if use_gpu is not None:
            tts_config.use_gpu = use_gpu
        tts_config.streaming_mode = (
            TTSStreamingMode.STREAMING if streaming else TTSStreamingMode.NON_STREAMING
        )
        tts_config.auto_play = auto_play

        typer.echo(
            f"Generating TTS for: {text[:100]}{'...' if len(text) > 100 else ''}"
        )

        try:
            success = self.tts_orchestrator.generate_tts_from_text(text, tts_config)
            if success:
                typer.echo("✓ TTS generation completed")
            else:
                typer.echo("✗ TTS generation failed", err=True)
                raise typer.Exit(1)
        except Exception as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1) from e

    def tts_listen_clipboard(
        self,
        voice: str | None = None,
        streaming: bool = False,
        auto_play: bool = True,
        output_file: str | None = None,
    ):
        """Listen to clipboard changes and generate TTS"""
        if not self.tts_orchestrator:
            typer.echo("TTS not available", err=True)
            raise typer.Exit(1)

        config = self.config_repo.load()
        tts_config = config.tts_config

        # Override config with command parameters
        tts_config.tts_mode = TTSMode.CLIPBOARD
        if voice:
            tts_config.default_voice = voice
        if output_file:
            tts_config.output_file_path = output_file
            tts_config.output_mode = (
                TTSOutputMode.SAVE_FILE if not auto_play else TTSOutputMode.BOTH
            )
        tts_config.streaming_mode = (
            TTSStreamingMode.STREAMING if streaming else TTSStreamingMode.NON_STREAMING
        )
        tts_config.auto_play = auto_play

        typer.echo("Starting TTS clipboard monitoring...")
        typer.echo(f"Voice: {tts_config.default_voice}")
        typer.echo(f"Mode: {'Streaming' if streaming else 'Non-streaming'}")
        typer.echo("Copy text to clipboard to generate TTS. Press Ctrl+C to stop.")

        try:
            self.tts_orchestrator.start_tts_mode(tts_config)

            # Keep the process running
            import time

            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            typer.echo("\nStopping clipboard monitoring...")
        finally:
            self.tts_orchestrator.stop_tts()

    def tts_listen_selection(
        self,
        voice: str | None = None,
        streaming: bool = False,
        auto_play: bool = True,
    ):
        """Listen for text selections and generate TTS via hotkey"""
        if not self.tts_orchestrator:
            typer.echo("TTS not available", err=True)
            raise typer.Exit(1)

        config = self.config_repo.load()
        tts_config = config.tts_config

        # Override config with command parameters
        tts_config.tts_mode = TTSMode.MOUSE
        if voice:
            tts_config.default_voice = voice
        tts_config.streaming_mode = (
            TTSStreamingMode.STREAMING if streaming else TTSStreamingMode.NON_STREAMING
        )
        tts_config.auto_play = auto_play

        typer.echo("Starting TTS selection mode...")
        typer.echo(f"Voice: {tts_config.default_voice}")
        typer.echo(f"Generate hotkey: {tts_config.tts_generate_key}")
        typer.echo(f"Stop hotkey: {tts_config.tts_stop_key}")
        typer.echo(
            "Select text and press hotkey to generate TTS. Press Ctrl+C to stop."
        )

        try:
            self.tts_orchestrator.start_tts_mode(tts_config)

            # Keep the process running
            import time

            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            typer.echo("\nStopping selection monitoring...")
        finally:
            self.tts_orchestrator.stop_tts()

    def tts_daemon_start(
        self,
        voice: str | None = None,
        mode: str = "clipboard",
        streaming: bool = False,
        auto_play: bool = True,
        background: bool = False,
    ):
        """Start TTS daemon with hotkey support"""
        if not self.tts_daemon_service:
            typer.echo("TTS daemon not available", err=True)
            raise typer.Exit(1)

        # Check if daemon is already running
        if self.tts_daemon_service.is_daemon_running():
            typer.echo("TTS daemon is already running", err=True)
            raise typer.Exit(1)

        config = self.config_repo.load()
        tts_config = config.tts_config

        # Set mode
        if mode == "clipboard":
            tts_config.tts_mode = TTSMode.CLIPBOARD
        elif mode == "selection":
            tts_config.tts_mode = TTSMode.MOUSE
        else:
            typer.echo(
                f"Invalid mode: {mode}. Use 'clipboard' or 'selection'", err=True
            )
            raise typer.Exit(1)

        # Override config
        if voice:
            tts_config.default_voice = voice
        tts_config.streaming_mode = (
            TTSStreamingMode.STREAMING if streaming else TTSStreamingMode.NON_STREAMING
        )
        tts_config.auto_play = auto_play

        # Check if we're being called recursively in background mode
        if background and os.environ.get("VOICEBRIDGE_NO_BACKGROUND"):
            background = False  # Disable background mode for recursive call

        if background:
            # For background mode, use subprocess with detachment
            import subprocess
            import sys

            cmd = [
                sys.executable,
                "-m",
                "voicebridge",
                "tts",
                "daemon",
                "start",
                "--mode",
                mode,
            ]
            if streaming:
                cmd.append("--streaming")
            if not auto_play:
                cmd.append("--no-auto-play")
            if voice:
                cmd.extend(["--voice", voice])

            # Start daemon in background, suppressing background flag
            env = os.environ.copy()
            env["VOICEBRIDGE_NO_BACKGROUND"] = (
                "1"  # Flag to prevent recursive background calls
            )

            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env=env,
            )

            # Give it a moment to start
            import time

            time.sleep(3)

            # Check if it actually started
            if self.tts_daemon_service.is_daemon_running():
                typer.echo("TTS daemon started in background")
            else:
                typer.echo("Failed to start daemon in background", err=True)
                raise typer.Exit(1)
            return

        try:
            self.tts_daemon_service.start_daemon(tts_config)
            typer.echo("TTS daemon started successfully")
            typer.echo(f"Mode: {mode}")
            typer.echo(f"Voice: {tts_config.default_voice}")
            typer.echo(f"Generate hotkey: {tts_config.tts_generate_key}")
            typer.echo(f"Stop hotkey: {tts_config.tts_stop_key}")
            typer.echo("Press Ctrl+C to stop the daemon")

            # Keep daemon running
            import time

            try:
                while self.tts_daemon_service.is_running:
                    time.sleep(1)
            except KeyboardInterrupt:
                typer.echo("\nStopping TTS daemon...")
                self.tts_daemon_service.stop_daemon()
                typer.echo("TTS daemon stopped")

        except Exception as e:
            typer.echo(f"Failed to start TTS daemon: {e}", err=True)
            raise typer.Exit(1) from e

    def tts_daemon_stop(self):
        """Stop TTS daemon"""
        if not self.tts_daemon_service:
            typer.echo("TTS daemon not available", err=True)
            raise typer.Exit(1)

        # Try to stop daemon using PID file
        if self.tts_daemon_service.stop_daemon_by_pid():
            typer.echo("TTS daemon stopped")
        else:
            typer.echo("No running TTS daemon found")

    def tts_daemon_status(self):
        """Check TTS daemon status"""
        if not self.tts_daemon_service:
            typer.echo("TTS daemon not available", err=True)
            raise typer.Exit(1)

        status_info = self.tts_daemon_service.get_status()
        typer.echo(f"TTS daemon status: {status_info['status']}")

        if status_info["status"] == "running":
            if "mode" in status_info:
                typer.echo(f"Mode: {status_info['mode']}")
            if "voice" in status_info:
                typer.echo(f"Voice: {status_info['voice']}")
            if "generate_key" in status_info:
                typer.echo(f"Generate hotkey: {status_info['generate_key']}")
            if "stop_key" in status_info:
                typer.echo(f"Stop hotkey: {status_info['stop_key']}")

    def tts_list_voices(self):
        """List available TTS voices"""
        if not self.tts_orchestrator:
            typer.echo("TTS not available", err=True)
            raise typer.Exit(1)

        config = self.config_repo.load()
        voices = self.tts_orchestrator.list_available_voices(config.tts_config)

        if not voices:
            typer.echo("No voices found")
            return

        typer.echo(f"Available voices ({len(voices)} total):")
        for name, voice_info in voices.items():
            display_name = voice_info.display_name or name
            details = []
            if voice_info.language:
                details.append(f"lang: {voice_info.language}")
            if voice_info.gender:
                details.append(f"gender: {voice_info.gender}")

            detail_str = f" ({', '.join(details)})" if details else ""
            typer.echo(f"  {name}: {display_name}{detail_str}")

    def tts_config_show(self):
        """Show current TTS configuration"""
        config = self.config_repo.load()
        tts_config = config.tts_config

        typer.echo("Current TTS Configuration:")
        typer.echo(f"  Enabled: {config.tts_enabled}")
        typer.echo(f"  Model path: {tts_config.model_path}")
        typer.echo(f"  Voice samples dir: {tts_config.voice_samples_dir}")
        typer.echo(f"  Default voice: {tts_config.default_voice}")
        typer.echo(f"  Mode: {tts_config.tts_mode.value}")
        typer.echo(f"  Streaming: {tts_config.streaming_mode.value}")
        typer.echo(f"  Output mode: {tts_config.output_mode.value}")
        typer.echo(f"  CFG scale: {tts_config.cfg_scale}")
        typer.echo(f"  Inference steps: {tts_config.inference_steps}")
        typer.echo("  Hotkeys:")
        typer.echo(f"    Generate: {tts_config.tts_generate_key}")
        typer.echo(f"    Stop: {tts_config.tts_stop_key}")

    def tts_config_set(
        self,
        model_path: str | None = None,
        voice_samples_dir: str | None = None,
        default_voice: str | None = None,
        cfg_scale: float | None = None,
        inference_steps: int | None = None,
        sample_rate: int | None = None,
        use_gpu: bool | None = None,
        auto_play: bool | None = None,
    ):
        """Set TTS configuration options"""
        config = self.config_repo.load()
        tts_config = config.tts_config

        updated = False

        if model_path:
            tts_config.model_path = model_path
            updated = True
            typer.echo(f"Set model path: {model_path}")

        if voice_samples_dir:
            tts_config.voice_samples_dir = voice_samples_dir
            updated = True
            typer.echo(f"Set voice samples dir: {voice_samples_dir}")

        if default_voice:
            tts_config.default_voice = default_voice
            updated = True
            typer.echo(f"Set default voice: {default_voice}")

        if cfg_scale is not None:
            tts_config.cfg_scale = cfg_scale
            updated = True
            typer.echo(f"Set CFG scale: {cfg_scale}")

        if inference_steps is not None:
            tts_config.inference_steps = inference_steps
            updated = True
            typer.echo(f"Set inference steps: {inference_steps}")

        if sample_rate is not None:
            tts_config.sample_rate = sample_rate
            updated = True
            typer.echo(f"Set sample rate: {sample_rate}")

        if use_gpu is not None:
            tts_config.use_gpu = use_gpu
            updated = True
            typer.echo(f"Set GPU usage: {use_gpu}")

        if auto_play is not None:
            tts_config.auto_play = auto_play
            updated = True
            typer.echo(f"Set auto-play: {auto_play}")

        if updated:
            self.config_repo.save(config)
            typer.echo("Configuration saved")
        else:
            typer.echo("No changes made")

    # TTS Command Aliases (for test compatibility)
    def tts_generate(
        self,
        text: str,
        voice: str | None = None,
        streaming: bool = False,
        output_file: str | None = None,
    ):
        """Generate TTS from text (alias for tts_generate_from_text)"""
        typer.echo(f"Generating TTS for: {text}")
        if not self.tts_orchestrator:
            typer.echo("TTS orchestrator not available", err=True)
            raise typer.Exit(1)

        config = self.config_repo.load()
        tts_config = config.tts_config

        # Override with parameters
        if voice:
            tts_config.default_voice = voice
        tts_config.streaming_mode = (
            TTSStreamingMode.STREAMING if streaming else TTSStreamingMode.NON_STREAMING
        )
        if output_file:
            tts_config.output_file_path = output_file
            tts_config.output_mode = TTSOutputMode.SAVE_FILE
        else:
            tts_config.output_mode = TTSOutputMode.PLAY_AUDIO

        result = self.tts_orchestrator.generate_and_play_speech(
            text, tts_config, voice, output_file
        )
        return result

    def tts_voices(self):
        """List available TTS voices (alias for tts_list_voices)"""
        return self.tts_list_voices()
