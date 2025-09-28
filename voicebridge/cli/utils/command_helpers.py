from pathlib import Path
from typing import Any

import typer


def validate_file_path(file_path: str, must_exist: bool = True) -> Path:
    """Validate and return a Path object for a file."""
    path = Path(file_path)
    if must_exist and not path.exists():
        typer.echo(f"Error: File not found: {file_path}", err=True)
        raise typer.Exit(1)
    return path


def validate_directory_path(dir_path: str, must_exist: bool = True) -> Path:
    """Validate and return a Path object for a directory."""
    path = Path(dir_path)
    if must_exist and not path.exists():
        typer.echo(f"Error: Directory not found: {dir_path}", err=True)
        raise typer.Exit(1)
    if must_exist and not path.is_dir():
        typer.echo(f"Error: Path is not a directory: {dir_path}", err=True)
        raise typer.Exit(1)
    return path


def build_whisper_config(
    model: str | None = None,
    language: str | None = None,
    initial_prompt: str | None = None,
    temperature: float = 0.0,
    use_gpu: bool | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Build a Whisper configuration dictionary from command options."""
    config = {}
    if model is not None:
        config["model_name"] = model
    if language is not None:
        config["language"] = language
    if initial_prompt is not None:
        config["initial_prompt"] = initial_prompt
    if temperature != 0.0:
        config["temperature"] = temperature
    if use_gpu is not None:
        config["use_gpu"] = use_gpu

    # Add any additional kwargs
    config.update(kwargs)
    return config


def handle_profile_config(profile_name: str | None, config_repo, profile_repo):
    """Load profile configuration if specified."""
    if profile_name:
        try:
            profile_config = profile_repo.load_profile(profile_name)
            if profile_config:
                return profile_config
            else:
                typer.echo(
                    f"Warning: Profile '{profile_name}' not found, using default config"
                )
        except Exception as e:
            typer.echo(f"Error loading profile '{profile_name}': {e}", err=True)

    return config_repo.load()


def display_progress(message: str, finished: bool = False):
    """Display progress message with appropriate formatting."""
    if finished:
        typer.echo(f"✓ {message}")
    else:
        typer.echo(f"→ {message}")


def display_error(message: str, exit_code: int = 1):
    """Display error message and exit."""
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(exit_code)


def display_warning(message: str):
    """Display warning message."""
    typer.echo(f"Warning: {message}", err=True)


def display_info(message: str):
    """Display informational message."""
    typer.echo(f"Info: {message}")


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def format_file_size(bytes_size: int) -> str:
    """Format file size in bytes to human readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"
