import typer

from voicebridge.cli.commands.base import BaseCommands
from voicebridge.cli.utils.command_helpers import (
    display_error,
    display_info,
    display_progress,
)


class ConfigCommands(BaseCommands):
    """Commands for configuration and profile management."""

    def config_show(self):
        """Show current configuration."""
        try:
            config = self.config_repo.load()

            typer.echo("Current Configuration:")
            typer.echo(f"  Model: {config.model_name}")
            typer.echo(f"  Language: {config.language or 'Auto-detect'}")
            typer.echo(f"  Temperature: {config.temperature}")
            typer.echo(f"  Use GPU: {config.use_gpu}")
            typer.echo(f"  Hotkey: {getattr(config, 'key', 'ctrl+f2')}")
            typer.echo(f"  Copy final: {config.copy_final}")
            typer.echo(f"  Paste final: {config.paste_final}")

            # Show TTS config if available
            if hasattr(config, "tts_config") and config.tts_config:
                tts = config.tts_config
                typer.echo("\nTTS Configuration:")
                typer.echo(f"  Default voice: {tts.default_voice}")
                typer.echo(f"  Auto play: {tts.auto_play}")
                typer.echo(f"  Use GPU: {tts.use_gpu}")
                typer.echo(f"  Sample rate: {tts.sample_rate}")
                typer.echo(f"  Generate hotkey: {tts.tts_generate_key}")
                typer.echo(f"  Stop hotkey: {tts.tts_stop_key}")

        except Exception as e:
            display_error(f"Error loading configuration: {e}")

    def config_set(self, key: str, value: str):
        """Set a configuration value."""
        try:
            config = self.config_repo.load()

            # Handle different value types
            if key in [
                "use_gpu",
                "copy_final",
                "paste_final",
                "copy_stream",
                "paste_stream",
            ]:
                # Boolean values
                bool_value = value.lower() in ("true", "1", "yes", "on")
                setattr(config, key, bool_value)
            elif key in ["temperature"]:
                # Float values
                setattr(config, key, float(value))
            elif key in ["sample_rate", "inference_steps"]:
                # Integer values
                setattr(config, key, int(value))
            else:
                # String values
                setattr(config, key, value)

            self.config_repo.save(config)
            display_progress(f"Configuration updated: {key} = {value}", finished=True)

        except ValueError:
            display_error(f"Invalid value for {key}: {value}")
        except Exception as e:
            display_error(f"Error setting configuration: {e}")

    def profile_save(self, name: str):
        """Save current configuration as a profile."""
        try:
            config = self.config_repo.load()
            success = self.profile_repo.save_profile(name, config)

            if success:
                display_progress(f"Profile '{name}' saved", finished=True)
            else:
                display_error(f"Failed to save profile '{name}'")

        except Exception as e:
            display_error(f"Error saving profile: {e}")

    def profile_load(self, name: str):
        """Load a configuration profile."""
        try:
            profile_config = self.profile_repo.load_profile(name)

            if profile_config:
                self.config_repo.save(profile_config)
                display_progress(f"Profile '{name}' loaded", finished=True)
            else:
                display_error(f"Profile '{name}' not found")

        except Exception as e:
            display_error(f"Error loading profile: {e}")

    def profile_list(self):
        """List all available profiles."""
        try:
            profiles = self.profile_repo.list_profiles()

            if not profiles:
                display_info("No profiles found")
                return

            typer.echo("Available Profiles:")
            for profile_name in profiles:
                typer.echo(f"  {profile_name}")

        except Exception as e:
            display_error(f"Error listing profiles: {e}")

    def profile_delete(self, name: str):
        """Delete a configuration profile."""
        try:
            # Confirm deletion
            typer.confirm(
                f"Delete profile '{name}'? This cannot be undone.", abort=True
            )

            success = self.profile_repo.delete_profile(name)

            if success:
                display_progress(f"Profile '{name}' deleted", finished=True)
            else:
                display_error(f"Profile '{name}' not found")

        except typer.Abort:
            display_info("Profile deletion cancelled")
        except Exception as e:
            display_error(f"Error deleting profile: {e}")

    def profile_copy(self, source: str, target: str):
        """Copy a profile to a new name."""
        try:
            source_config = self.profile_repo.load_profile(source)

            if not source_config:
                display_error(f"Source profile '{source}' not found")
                return

            success = self.profile_repo.save_profile(target, source_config)

            if success:
                display_progress(
                    f"Profile copied: '{source}' -> '{target}'", finished=True
                )
            else:
                display_error("Failed to copy profile")

        except Exception as e:
            display_error(f"Error copying profile: {e}")

    def profile_rename(self, old_name: str, new_name: str):
        """Rename a profile."""
        try:
            # Get the profile config
            profile_config = self.profile_repo.get_profile(old_name)

            if not profile_config:
                display_error(f"Profile '{old_name}' not found")
                return

            # Save with new name
            success = self.profile_repo.save_profile(new_name, profile_config)

            if success:
                # Delete old profile
                self.profile_repo.delete_profile(old_name)
                display_progress(
                    f"Profile renamed: '{old_name}' -> '{new_name}'", finished=True
                )
            else:
                display_error("Failed to rename profile")

        except Exception as e:
            display_error(f"Error renaming profile: {e}")

    def profile_export(self, name: str, file_path: str):
        """Export a profile to a file."""
        try:
            profile_config = self.profile_repo.load_profile(name)

            if not profile_config:
                display_error(f"Profile '{name}' not found")
                return

            success = self.profile_repo.export_profile(name, file_path)

            if success:
                display_progress(
                    f"Profile '{name}' exported to {file_path}", finished=True
                )
            else:
                display_error("Failed to export profile")

        except Exception as e:
            display_error(f"Error exporting profile: {e}")

    def profile_import(self, file_path: str, name: str | None = None):
        """Import a profile from a file."""
        try:
            success = self.profile_repo.import_profile(file_path, name)

            if success:
                profile_name = name or "imported_profile"
                display_progress(f"Profile imported as '{profile_name}'", finished=True)
            else:
                display_error(f"Failed to import profile from {file_path}")

        except Exception as e:
            display_error(f"Error importing profile: {e}")

    def config_reset(self):
        """Reset configuration to defaults."""
        try:
            # Confirm reset
            typer.confirm(
                "Reset configuration to defaults? This will overwrite your current settings.",
                abort=True,
            )

            success = self.config_repo.reset_to_defaults()

            if success:
                display_progress("Configuration reset to defaults", finished=True)
            else:
                display_error("Failed to reset configuration")

        except typer.Abort:
            display_info("Configuration reset cancelled")
        except Exception as e:
            display_error(f"Error resetting configuration: {e}")

    def config_backup(self, file_path: str | None = None):
        """Backup current configuration."""
        try:
            if not file_path:
                import datetime

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = f"voicebridge_config_backup_{timestamp}.json"

            success = self.config_repo.backup_config(file_path)

            if success:
                display_progress(
                    f"Configuration backed up to {file_path}", finished=True
                )
            else:
                display_error("Failed to backup configuration")

        except Exception as e:
            display_error(f"Error backing up configuration: {e}")

    def config_restore(self, file_path: str):
        """Restore configuration from backup."""
        try:
            # Confirm restore
            typer.confirm(
                f"Restore configuration from {file_path}? This will overwrite your current settings.",
                abort=True,
            )

            success = self.config_repo.restore_config(file_path)

            if success:
                display_progress(
                    f"Configuration restored from {file_path}", finished=True
                )
            else:
                display_error(f"Failed to restore configuration from {file_path}")

        except typer.Abort:
            display_info("Configuration restore cancelled")
        except Exception as e:
            display_error(f"Error restoring configuration: {e}")

    def config_validate(self):
        """Validate current configuration."""
        try:
            config = self.config_repo.load()

            # Perform validation checks
            issues = []

            # Check model availability
            if hasattr(config, "model_name"):
                # This would need access to model service
                pass

            # Check GPU settings
            if config.use_gpu and self.system_service:
                gpu_devices = self.system_service.detect_gpu_devices()
                gpu_info = gpu_devices[0] if gpu_devices else None
                if not gpu_info or not gpu_info.is_available:
                    issues.append("GPU acceleration enabled but no GPU available")

            # Check TTS configuration
            if hasattr(config, "tts_config") and config.tts_config:
                tts = config.tts_config
                if tts.voice_samples_path:
                    from pathlib import Path

                    if not Path(tts.voice_samples_path).exists():
                        issues.append(
                            f"TTS voice samples path not found: {tts.voice_samples_path}"
                        )

            # Report results
            if issues:
                typer.echo("Configuration Issues Found:")
                for issue in issues:
                    typer.echo(f"  ⚠️ {issue}")
            else:
                display_progress("Configuration is valid", finished=True)

        except Exception as e:
            display_error(f"Error validating configuration: {e}")
