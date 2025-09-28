import typer


# Common parameter definitions
def common_whisper_options():
    """Common Whisper model and transcription options."""
    return {
        "model": typer.Option(None, "--model", "-m", help="Whisper model to use"),
        "language": typer.Option(None, "--language", "-l", help="Language code"),
        "initial_prompt": typer.Option(None, "--prompt", help="Initial prompt"),
        "temperature": typer.Option(0.0, "--temperature", "-t", help="Temperature"),
        "profile": typer.Option(None, "--profile", "-p", help="Use profile"),
        "max_memory": typer.Option(
            0, "--max-memory", help="Memory limit in MB (0 = auto-detect)"
        ),
        "debug": typer.Option(False, "--debug", help="Enable debug logging"),
    }


def common_output_options():
    """Common output and clipboard options."""
    return {
        "paste_stream": typer.Option(
            False, "--paste-stream", help="Paste streaming text"
        ),
        "copy_stream": typer.Option(False, "--copy-stream", help="Copy streaming text"),
        "paste_final": typer.Option(False, "--paste-final", help="Paste final text"),
        "copy_final": typer.Option(
            True, "--copy-final/--no-copy-final", help="Copy final text"
        ),
    }


def common_file_options():
    """Common file input/output options."""
    return {
        "output": typer.Option(None, "--output", "-o", help="Output file path"),
        "format": typer.Option("txt", "--format", "-f", help="Output format"),
    }


def common_tts_options():
    """Common TTS options."""
    return {
        "voice": typer.Option(None, "--voice", help="Voice to use for TTS"),
        "output": typer.Option(None, "--output", "-o", help="Output audio file"),
        "play": typer.Option(True, "--play/--no-play", help="Play generated audio"),
        "streaming": typer.Option(False, "--streaming", help="Use streaming mode"),
    }


def common_batch_options():
    """Common batch processing options."""
    return {
        "workers": typer.Option(
            4, "--workers", "-w", help="Number of parallel workers"
        ),
        "recursive": typer.Option(
            False, "--recursive", "-r", help="Process directories recursively"
        ),
        "output_dir": typer.Option(".", "--output-dir", help="Output directory"),
    }
