"""Audio recorder factory that automatically selects the best recorder for the environment."""

from voicebridge.ports.interfaces import AudioRecorder


def create_audio_recorder() -> AudioRecorder:
    """Create the best audio recorder for the current environment."""

    # Check if we're in WSL
    try:
        with open("/proc/version") as f:
            if "microsoft" in f.read().lower():
                # We're in WSL, try to use WSL audio recorder
                try:
                    from voicebridge.adapters.wsl_audio import WSLAudioRecorder

                    wsl_recorder = WSLAudioRecorder()
                    # Test if Windows FFmpeg is available
                    if wsl_recorder._windows_ffmpeg_path:
                        devices = wsl_recorder.list_devices()
                        if devices:
                            print(
                                f"WSL detected with {len(devices)} Windows audio devices available."
                            )
                            return wsl_recorder
                        else:
                            # Even if no devices are listed, try WSL recorder anyway
                            # Device enumeration might fail due to permissions, but recording might work
                            print(
                                "WSL detected but no Windows audio devices found during enumeration."
                            )
                            print(
                                "Attempting to use WSL audio recorder anyway (device enumeration may fail due to permissions)."
                            )
                            print(
                                "Please check Windows Privacy Settings > Microphone > Allow desktop apps to access your microphone"
                            )
                            return wsl_recorder
                except Exception as e:
                    print(f"WSL audio setup failed: {e}")
    except Exception:
        pass

    # Fall back to regular FFmpeg audio recorder
    from voicebridge.adapters.audio import FFmpegAudioRecorder

    return FFmpegAudioRecorder()
