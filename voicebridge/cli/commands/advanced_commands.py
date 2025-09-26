import typer

from voicebridge.cli.commands.base import BaseCommands
from voicebridge.cli.utils.command_helpers import (
    display_error,
    display_info,
    display_progress,
)


class AdvancedCommands(BaseCommands):
    """Commands for advanced features like vocabulary, webhooks, and post-processing."""

    def vocabulary_add(
        self,
        words: str,
        vocabulary_type: str = "custom",
        profile: str = "default",
        weight: float = 1.0,
    ):
        """Add words to vocabulary for improved recognition."""
        if not self.vocabulary_management_service:
            display_error("Vocabulary service not available")
            return

        try:
            word_list = [word.strip() for word in words.split(",") if word.strip()]

            if not word_list:
                display_error("No valid words provided")
                return

            display_progress(
                f"Adding {len(word_list)} words to {vocabulary_type} vocabulary..."
            )

            success = self.vocabulary_management_service.add_words(
                word_list, vocabulary_type, profile, weight
            )

            if success:
                display_progress("Words added to vocabulary", finished=True)
                typer.echo(f"  Profile: {profile}")
                typer.echo(f"  Type: {vocabulary_type}")
                typer.echo(f"  Words: {', '.join(word_list)}")
            else:
                display_error("Failed to add words to vocabulary")

        except Exception as e:
            display_error(f"Error adding vocabulary: {e}")

    def vocabulary_remove(
        self,
        words: str,
        vocabulary_type: str = "custom",
        profile: str = "default",
    ):
        """Remove words from vocabulary."""
        if not self.vocabulary_management_service:
            display_error("Vocabulary service not available")
            return

        try:
            word_list = [word.strip() for word in words.split(",") if word.strip()]

            if not word_list:
                display_error("No valid words provided")
                return

            display_progress(
                f"Removing {len(word_list)} words from {vocabulary_type} vocabulary..."
            )

            success = self.vocabulary_management_service.remove_words(
                word_list, vocabulary_type, profile
            )

            if success:
                display_progress("Words removed from vocabulary", finished=True)
            else:
                display_error("Failed to remove words from vocabulary")

        except Exception as e:
            display_error(f"Error removing vocabulary: {e}")

    def vocabulary_list(
        self, vocabulary_type: str | None = None, profile: str = "default"
    ):
        """List vocabulary words."""
        if not self.vocabulary_management_service:
            display_error("Vocabulary service not available")
            return

        try:
            vocabularies = self.vocabulary_management_service.list_vocabularies(
                vocabulary_type, profile
            )

            if not vocabularies:
                display_info("No vocabulary entries found")
                return

            typer.echo(f"Vocabulary for profile '{profile}':")
            typer.echo("=" * 50)

            for vocab_type, words in vocabularies.items():
                typer.echo(f"\n{vocab_type.upper()} ({len(words)} words):")

                if isinstance(words, dict):
                    # Words with weights/metadata
                    for word, info in words.items():
                        weight = (
                            info.get("weight", 1.0) if isinstance(info, dict) else info
                        )
                        typer.echo(f"  {word} (weight: {weight})")
                else:
                    # Simple word list
                    for word in words:
                        typer.echo(f"  {word}")

        except Exception as e:
            display_error(f"Error listing vocabulary: {e}")

    def vocabulary_import(
        self,
        file_path: str,
        vocabulary_type: str = "custom",
        profile: str = "default",
        format: str = "txt",
    ):
        """Import vocabulary from file."""
        if not self.vocabulary_management_service:
            display_error("Vocabulary service not available")
            return

        try:
            from pathlib import Path

            input_file = Path(file_path)

            if not input_file.exists():
                display_error(f"File not found: {file_path}")
                return

            display_progress(f"Importing vocabulary from {input_file.name}...")

            success = self.vocabulary_management_service.import_vocabulary(
                str(input_file), vocabulary_type, profile, format
            )

            if success:
                display_progress("Vocabulary imported successfully", finished=True)
            else:
                display_error("Failed to import vocabulary")

        except Exception as e:
            display_error(f"Error importing vocabulary: {e}")

    def vocabulary_export(self, file_path: str, profile: str = "default"):
        """Export vocabulary to file."""
        if not self.vocabulary_management_service:
            display_error("Vocabulary service not available")
            return

        try:
            display_progress(f"Exporting vocabulary to {file_path}...")

            success = self.vocabulary_management_service.export_vocabulary(
                file_path, profile
            )

            if success:
                display_progress(f"Vocabulary exported to {file_path}", finished=True)
            else:
                display_error("Failed to export vocabulary")

        except Exception as e:
            display_error(f"Error exporting vocabulary: {e}")

    def postprocessing_config(
        self,
        enable_spell_check: bool | None = None,
        enable_grammar_check: bool | None = None,
        enable_punctuation: bool | None = None,
        enable_capitalization: bool | None = None,
        custom_rules: str | None = None,
        profile: str = "default",
    ):
        """Configure post-processing settings."""
        if not self.postprocessing_service:
            display_error("Post-processing service not available")
            return

        try:
            config_updates = {}

            if enable_spell_check is not None:
                config_updates["spell_check"] = enable_spell_check
            if enable_grammar_check is not None:
                config_updates["grammar_check"] = enable_grammar_check
            if enable_punctuation is not None:
                config_updates["punctuation"] = enable_punctuation
            if enable_capitalization is not None:
                config_updates["capitalization"] = enable_capitalization
            if custom_rules is not None:
                config_updates["custom_rules"] = custom_rules.split(",")

            if not config_updates:
                # Show current config
                config = self.postprocessing_service.get_config(profile)
                typer.echo(f"Post-processing configuration for '{profile}':")
                typer.echo(f"  Spell check: {config.get('spell_check', False)}")
                typer.echo(f"  Grammar check: {config.get('grammar_check', False)}")
                typer.echo(f"  Punctuation: {config.get('punctuation', True)}")
                typer.echo(f"  Capitalization: {config.get('capitalization', True)}")
                typer.echo(f"  Custom rules: {len(config.get('custom_rules', []))}")
                return

            success = self.postprocessing_service.update_config(profile, config_updates)

            if success:
                display_progress("Post-processing configuration updated", finished=True)
                for key, value in config_updates.items():
                    typer.echo(f"  {key}: {value}")
            else:
                display_error("Failed to update post-processing configuration")

        except Exception as e:
            display_error(f"Error configuring post-processing: {e}")

    def postprocessing_test(self, text: str, profile: str = "default"):
        """Test post-processing on sample text."""
        if not self.postprocessing_service:
            display_error("Post-processing service not available")
            return

        try:
            display_progress("Testing post-processing...")

            result = self.postprocessing_service.process_text(text, profile)

            if result:
                typer.echo("\nPost-processing Test Results:")
                typer.echo("=" * 50)
                typer.echo(f"Original:  {text}")
                typer.echo(f"Processed: {result.get('text', text)}")

                # Show what was changed
                changes = result.get("changes", [])
                if changes:
                    typer.echo(f"\nChanges made ({len(changes)}):")
                    for change in changes:
                        change_type = change.get("type", "unknown")
                        old_text = change.get("original", "")
                        new_text = change.get("corrected", "")
                        typer.echo(f"  {change_type}: '{old_text}' -> '{new_text}'")
                else:
                    typer.echo("\nNo changes were made.")
            else:
                display_error("Post-processing test failed")

        except Exception as e:
            display_error(f"Error testing post-processing: {e}")

    def webhook_add(
        self,
        url: str,
        events: str = "transcription_complete",
        secret: str | None = None,
        timeout: int = 30,
        retry_count: int = 3,
    ):
        """Add a webhook for event notifications."""
        if not self.webhook_service:
            display_error("Webhook service not available")
            return

        try:
            event_list = [event.strip() for event in events.split(",") if event.strip()]

            display_progress(f"Adding webhook: {url}")

            webhook_config = {
                "url": url,
                "events": event_list,
                "secret": secret,
                "timeout": timeout,
                "retry_count": retry_count,
            }

            success = self.webhook_service.add_webhook(webhook_config)

            if success:
                display_progress("Webhook added successfully", finished=True)
                typer.echo(f"  URL: {url}")
                typer.echo(f"  Events: {', '.join(event_list)}")
                typer.echo(f"  Timeout: {timeout}s")
                typer.echo(f"  Retries: {retry_count}")
            else:
                display_error("Failed to add webhook")

        except Exception as e:
            display_error(f"Error adding webhook: {e}")

    def webhook_remove(self, url: str):
        """Remove a webhook."""
        if not self.webhook_service:
            display_error("Webhook service not available")
            return

        try:
            success = self.webhook_service.remove_webhook(url)

            if success:
                display_progress(f"Webhook removed: {url}", finished=True)
            else:
                display_error(f"Webhook not found: {url}")

        except Exception as e:
            display_error(f"Error removing webhook: {e}")

    def webhook_list(self):
        """List all configured webhooks."""
        if not self.webhook_service:
            display_error("Webhook service not available")
            return

        try:
            webhooks = self.webhook_service.list_webhooks()

            if not webhooks:
                display_info("No webhooks configured")
                return

            typer.echo("Configured Webhooks:")
            typer.echo("=" * 80)
            typer.echo(f"{'URL':<40} {'Events':<25} {'Status':<10} {'Last Used'}")
            typer.echo("-" * 80)

            for webhook in webhooks:
                url = webhook.get("url", "")[:40]
                events = ", ".join(webhook.get("events", []))[:25]
                status = webhook.get("status", "Unknown")
                last_used = webhook.get("last_used", "Never")[:20]

                typer.echo(f"{url:<40} {events:<25} {status:<10} {last_used}")

        except Exception as e:
            display_error(f"Error listing webhooks: {e}")

    def webhook_test(self, url: str, event_type: str = "transcription_complete"):
        """Test a webhook with sample data."""
        if not self.webhook_service:
            display_error("Webhook service not available")
            return

        try:
            display_progress(f"Testing webhook: {url}")

            # Create test payload
            test_payload = {
                "event_type": event_type,
                "timestamp": "test_timestamp",
                "session_id": "test_session",
                "data": {
                    "text": "This is a test transcription.",
                    "confidence": 0.95,
                    "language": "en",
                },
            }

            result = self.webhook_service.test_webhook(url, test_payload)

            if result and result.get("success"):
                display_progress("Webhook test successful", finished=True)
                typer.echo(f"  Response code: {result.get('status_code')}")
                typer.echo(f"  Response time: {result.get('response_time', 0):.2f}s")

                response_data = result.get("response_data")
                if response_data:
                    typer.echo(f"  Response: {response_data}")
            else:
                error_msg = (
                    result.get("error", "Unknown error") if result else "Test failed"
                )
                display_error(f"Webhook test failed: {error_msg}")

        except Exception as e:
            display_error(f"Error testing webhook: {e}")

    def api_start(self, port: int = 8000, host: str = "127.0.0.1", workers: int = 1):
        """Start the REST API server."""
        display_info("REST API functionality not implemented yet")
        display_info(f"Would start API server on {host}:{port} with {workers} workers")

    def api_stop(self):
        """Stop the REST API server."""
        display_info("REST API functionality not implemented yet")

    def api_status(self):
        """Show REST API server status."""
        display_info("REST API functionality not implemented yet")

    def operations_management_start(self):
        """Start the operations management service."""
        if not self.progress_service:
            display_error("Progress service not available")
            return

        try:
            success = self.progress_service.start_monitoring()

            if success:
                display_progress("Operations management started", finished=True)
            else:
                display_error("Failed to start operations management")

        except Exception as e:
            display_error(f"Error starting operations management: {e}")

    def operations_management_stop(self):
        """Stop the operations management service."""
        if not self.progress_service:
            display_error("Progress service not available")
            return

        try:
            success = self.progress_service.stop_monitoring()

            if success:
                display_progress("Operations management stopped", finished=True)
            else:
                display_error("Failed to stop operations management")

        except Exception as e:
            display_error(f"Error stopping operations management: {e}")

    def operations_management_status(self):
        """Show operations management status."""
        if not self.progress_service:
            display_error("Progress service not available")
            return

        try:
            status = self.progress_service.get_monitoring_status()

            typer.echo("Operations Management Status:")
            typer.echo(f"  Active: {status.get('active', False)}")
            typer.echo(f"  Active operations: {status.get('active_operations', 0)}")
            typer.echo(f"  Total processed: {status.get('total_processed', 0)}")
            typer.echo(f"  Uptime: {status.get('uptime', 'Unknown')}")

        except Exception as e:
            display_error(f"Error getting operations management status: {e}")

    def run_diagnostics(self):
        """Run system diagnostics and health checks."""
        typer.echo("Running VoiceBridge Diagnostics...")
        typer.echo("=" * 50)

        # Check core services
        checks = [
            ("Config Repository", self.config_repo is not None),
            ("Profile Repository", self.profile_repo is not None),
            ("System Service", self.system_service is not None),
            ("Transcription Service", self.transcription_orchestrator is not None),
            ("Audio Format Service", self.audio_format_service is not None),
            ("TTS Service", self.tts_orchestrator is not None),
            ("Session Service", self.session_service is not None),
            ("Export Service", self.export_service is not None),
        ]

        for check_name, check_result in checks:
            status = "✓" if check_result else "✗"
            typer.echo(f"  {status} {check_name}")

        # System health checks
        typer.echo("\nSystem Health:")

        if self.system_service:
            try:
                gpu_devices = self.system_service.detect_gpu_devices()
                gpu_info = gpu_devices[0] if gpu_devices else None
                if gpu_info and gpu_info.gpu_type.value != "none":
                    typer.echo("  ✓ GPU available")
                else:
                    typer.echo("  ⚠ GPU not available (using CPU)")
            except Exception:
                typer.echo("  ✗ Error checking GPU status")

        # Check configuration
        try:
            self.config_repo.load()
            typer.echo("  ✓ Configuration loaded successfully")
        except Exception:
            typer.echo("  ✗ Error loading configuration")

        # Check dependencies
        typer.echo("\nDependency Checks:")
        dependencies = [
            ("torch", "PyTorch"),
            ("whisper", "OpenAI Whisper"),
            ("typer", "Typer CLI"),
            ("pynput", "Input handling"),
        ]

        for module, description in dependencies:
            try:
                __import__(module)
                typer.echo(f"  ✓ {description}")
            except ImportError:
                typer.echo(f"  ✗ {description} (missing)")

        typer.echo("\nDiagnostics completed.")

    def benchmark_full_system(self):
        """Run comprehensive system benchmark."""
        if not self.system_service:
            display_error("System service not available")
            return

        display_info("Starting comprehensive system benchmark...")
        display_info("This may take several minutes...")

        try:
            results = self.system_service.run_full_benchmark()

            if results:
                typer.echo("\nSystem Benchmark Results:")
                typer.echo("=" * 50)

                # CPU benchmark
                cpu_results = results.get("cpu", {})
                typer.echo("CPU Performance:")
                typer.echo(f"  Score: {cpu_results.get('score', 0):.1f}")
                typer.echo(f"  Test duration: {cpu_results.get('duration', 0):.1f}s")

                # GPU benchmark
                gpu_results = results.get("gpu", {})
                if gpu_results:
                    typer.echo("\nGPU Performance:")
                    typer.echo(f"  Score: {gpu_results.get('score', 0):.1f}")
                    typer.echo(
                        f"  Memory usage: {gpu_results.get('memory_usage', 0):.1f}%"
                    )

                # Audio processing benchmark
                audio_results = results.get("audio", {})
                if audio_results:
                    typer.echo("\nAudio Processing:")
                    typer.echo(
                        f"  Real-time factor: {audio_results.get('real_time_factor', 0):.2f}x"
                    )
                    typer.echo(
                        f"  Quality score: {audio_results.get('quality_score', 0):.1f}/10"
                    )

                # Overall rating
                overall_rating = results.get("overall_rating", "Unknown")
                typer.echo(f"\nOverall System Rating: {overall_rating}")

                # Recommendations
                recommendations = results.get("recommendations", [])
                if recommendations:
                    typer.echo("\nRecommendations:")
                    for rec in recommendations:
                        typer.echo(f"  • {rec}")
            else:
                display_error("Benchmark failed to complete")

        except Exception as e:
            display_error(f"Error running benchmark: {e}")
