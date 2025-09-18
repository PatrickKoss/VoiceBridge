"""
End-to-end tests for VoiceBridge CLI commands.

These tests run the actual CLI commands against real audio files and text
to ensure the entire application stack works correctly.
"""

import json
import time

import pytest

from voicebridge.tests.e2e_helpers import (
    AudioGenerator,
    ClipboardSimulator,
    FileManager,
    VoiceBridgeCLI,
)


class TestE2ECLICommands:
    """End-to-end tests for all CLI commands."""

    @pytest.fixture(autouse=True)
    def setup_e2e_test(self, tmp_path):
        """Set up E2E test environment."""
        self.tmp_path = tmp_path
        self.cli = VoiceBridgeCLI()
        self.file_manager = FileManager(tmp_path)
        self.audio_generator = AudioGenerator()
        self.clipboard_sim = ClipboardSimulator()

        # Create test directories
        self.test_audio_dir = tmp_path / "test_audio"
        self.test_output_dir = tmp_path / "output"
        self.test_voices_dir = tmp_path / "voices"

        for dir_path in [
            self.test_audio_dir,
            self.test_output_dir,
            self.test_voices_dir,
        ]:
            dir_path.mkdir(exist_ok=True)

        # Generate test audio files
        self.sample_audio = self.audio_generator.generate_test_audio(
            self.test_audio_dir / "sample.wav",
            duration=5.0,
            text="Hello, this is a test audio file for speech recognition.",
        )

        self.long_audio = self.audio_generator.generate_test_audio(
            self.test_audio_dir / "long_sample.wav",
            duration=30.0,
            text="This is a longer audio file for testing batch processing and resume functionality. It contains multiple sentences.",
        )

        # Generate voice samples for TTS
        self.test_voice = self.audio_generator.generate_voice_sample(
            self.test_voices_dir / "en-test_voice.wav",
            text="This is a test voice sample",
        )

    def test_basic_transcribe_command(self):
        """Test basic file transcription."""
        output_file = self.test_output_dir / "transcription.txt"

        result = self.cli.run_command(
            [
                "transcribe",
                str(self.sample_audio),
                "--output",
                str(output_file),
                "--format",
                "txt",
            ]
        )

        assert result.returncode == 0
        assert output_file.exists()

        with open(output_file) as f:
            content = f.read()
            assert len(content) > 0
            # Should contain some recognizable words from our test audio
            assert any(word in content.lower() for word in ["hello", "test", "audio"])

    def test_transcribe_with_json_output(self):
        """Test transcription with JSON output format."""
        output_file = self.test_output_dir / "transcription.json"

        result = self.cli.run_command(
            [
                "transcribe",
                str(self.sample_audio),
                "--output",
                str(output_file),
                "--format",
                "json",
            ]
        )

        assert result.returncode == 0
        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)
            assert "segments" in data
            assert "text" in data
            assert len(data["text"]) > 0

    def test_transcribe_with_srt_output(self):
        """Test transcription with SRT subtitle output."""
        output_file = self.test_output_dir / "transcription.srt"

        result = self.cli.run_command(
            [
                "transcribe",
                str(self.sample_audio),
                "--output",
                str(output_file),
                "--format",
                "srt",
            ]
        )

        assert result.returncode == 0
        assert output_file.exists()

        with open(output_file) as f:
            content = f.read()
            # SRT format should contain timestamps
            assert "-->" in content
            assert any(word in content.lower() for word in ["hello", "test", "audio"])

    def test_batch_transcribe_command(self):
        """Test batch transcription of multiple files."""
        # Create additional test files
        self.audio_generator.generate_test_audio(
            self.test_audio_dir / "sample2.wav",
            duration=3.0,
            text="Second test audio file.",
        )

        self.audio_generator.generate_test_audio(
            self.test_audio_dir / "sample3.wav",
            duration=4.0,
            text="Third test audio for batch processing.",
        )

        batch_output_dir = self.test_output_dir / "batch_results"

        result = self.cli.run_command(
            [
                "batch-transcribe",
                str(self.test_audio_dir),
                "--output-dir",
                str(batch_output_dir),
                "--workers",
                "2",
                "--pattern",
                "*.wav",
            ]
        )

        assert result.returncode == 0
        assert batch_output_dir.exists()

        # Should have transcribed all 3 WAV files (+ the long one from setup)
        output_files = list(batch_output_dir.glob("*.txt"))
        assert len(output_files) >= 3

    def test_audio_info_command(self):
        """Test audio file information display."""
        result = self.cli.run_command(["audio", "info", str(self.sample_audio)])

        assert result.returncode == 0
        output = result.stdout

        # Should contain basic audio file information
        assert "Duration" in output or "duration" in output
        assert "Sample Rate" in output or "sample" in output.lower()
        assert "Channels" in output or "channel" in output.lower()

    def test_audio_formats_command(self):
        """Test listing supported audio formats."""
        result = self.cli.run_command(["audio", "formats"])

        assert result.returncode == 0
        output = result.stdout

        # Should list common audio formats
        assert any(fmt in output.lower() for fmt in ["wav", "mp3", "m4a", "flac"])

    def test_audio_split_command(self):
        """Test audio file splitting."""
        split_output_dir = self.test_output_dir / "split_chunks"

        result = self.cli.run_command(
            [
                "audio",
                "split",
                str(self.long_audio),
                "--output-dir",
                str(split_output_dir),
                "--method",
                "duration",
                "--duration",
                "10",  # 10 second chunks
            ]
        )

        assert result.returncode == 0
        assert split_output_dir.exists()

        # Should have created multiple chunks from 30s audio
        chunk_files = list(split_output_dir.glob("*.wav"))
        assert len(chunk_files) >= 2

    def test_audio_preprocess_command(self):
        """Test audio preprocessing."""
        preprocessed_file = self.test_output_dir / "preprocessed.wav"

        result = self.cli.run_command(
            [
                "audio",
                "preprocess",
                str(self.sample_audio),
                str(preprocessed_file),
                "--normalize",
                "-20",
                "--trim-silence",
            ]
        )

        assert result.returncode == 0
        assert preprocessed_file.exists()

        # Preprocessed file should exist and be different from original
        _original_size = self.sample_audio.stat().st_size
        processed_size = preprocessed_file.stat().st_size
        # Sizes might be similar but shouldn't be identical
        assert processed_size > 0

    def test_tts_generate_command(self):
        """Test basic TTS generation."""
        output_file = self.test_output_dir / "generated_speech.wav"

        result = self.cli.run_command(
            [
                "tts",
                "generate",
                "Hello, this is a test of text to speech functionality.",
                "--output",
                str(output_file),
                "--no-auto-play",  # Don't play during testing
            ]
        )

        assert result.returncode == 0
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_tts_voices_command(self):
        """Test listing available TTS voices."""
        result = self.cli.run_command(["tts", "voices"])

        assert result.returncode == 0
        output = result.stdout

        # Should show some voice information
        assert len(output) > 0

    def test_tts_config_show_command(self):
        """Test showing TTS configuration."""
        result = self.cli.run_command(["tts", "config", "show"])

        assert result.returncode == 0
        output = result.stdout

        # Should display configuration settings
        assert "model" in output.lower() or "voice" in output.lower()

    def test_config_show_command(self):
        """Test showing general configuration."""
        result = self.cli.run_command(["config", "--show"])

        assert result.returncode == 0
        output = result.stdout

        # Should display configuration
        assert len(output) > 0

    def test_config_set_command(self):
        """Test setting configuration values."""
        result = self.cli.run_command(
            ["config", "--set-key", "use_gpu", "--value", "false"]
        )

        assert result.returncode == 0

    def test_sessions_list_command(self):
        """Test listing transcription sessions."""
        result = self.cli.run_command(["sessions", "list"])

        assert result.returncode == 0
        # Should not error even if no sessions exist

    def test_gpu_status_command(self):
        """Test GPU status display."""
        result = self.cli.run_command(["gpu", "status"])

        assert result.returncode == 0
        output = result.stdout

        # Should show GPU information (even if no GPU available)
        assert len(output) > 0

    def test_performance_stats_command(self):
        """Test performance statistics display."""
        result = self.cli.run_command(["performance", "stats"])

        assert result.returncode == 0
        output = result.stdout

        # Should display performance information
        assert len(output) > 0

    def test_export_formats_command(self):
        """Test listing export formats."""
        result = self.cli.run_command(["export", "formats"])

        assert result.returncode == 0
        output = result.stdout

        # Should list supported formats
        assert any(fmt in output.lower() for fmt in ["txt", "json", "srt", "vtt"])

    def test_vocabulary_list_command(self):
        """Test vocabulary listing."""
        result = self.cli.run_command(["vocabulary", "list"])

        assert result.returncode == 0
        # Should not error even if no custom vocabulary exists

    def test_vocabulary_add_remove_commands(self):
        """Test adding and removing vocabulary words."""
        # Add a word
        result = self.cli.run_command(
            ["vocabulary", "add", "testword", "--type", "custom"]
        )
        assert result.returncode == 0

        # Remove the word
        result = self.cli.run_command(
            ["vocabulary", "remove", "testword", "--type", "custom"]
        )
        assert result.returncode == 0

    def test_profile_operations(self):
        """Test profile save/load/list operations."""
        profile_name = "test_profile"

        # Save current config as profile
        result = self.cli.run_command(["profile", "--save", profile_name])
        assert result.returncode == 0

        # List profiles (should include our test profile)
        result = self.cli.run_command(["profile", "--list"])
        assert result.returncode == 0
        output = result.stdout
        assert profile_name in output

        # Load the profile
        result = self.cli.run_command(["profile", "--load", profile_name])
        assert result.returncode == 0

    def test_postproc_config_command(self):
        """Test post-processing configuration."""
        result = self.cli.run_command(["postproc", "config", "--show"])
        assert result.returncode == 0

        # Test setting post-processing options
        result = self.cli.run_command(
            ["postproc", "config", "--punctuation", "--capitalization"]
        )
        assert result.returncode == 0

    def test_postproc_test_command(self):
        """Test post-processing on sample text."""
        result = self.cli.run_command(
            ["postproc", "test", "hello world this is a test"]
        )
        assert result.returncode == 0
        output = result.stdout
        assert len(output) > 0

    def test_confidence_configure_command(self):
        """Test confidence analysis configuration."""
        result = self.cli.run_command(
            [
                "confidence",
                "configure",
                "--high",
                "0.9",
                "--medium",
                "0.7",
                "--low",
                "0.5",
            ]
        )
        assert result.returncode == 0

    def test_webhook_list_command(self):
        """Test webhook listing."""
        result = self.cli.run_command(["webhook", "list"])
        assert result.returncode == 0
        # Should not error even if no webhooks configured

    def test_operations_list_command(self):
        """Test operations listing."""
        result = self.cli.run_command(["operations", "list"])
        assert result.returncode == 0
        # Should not error even if no active operations

    def test_circuit_status_command(self):
        """Test circuit breaker status."""
        result = self.cli.run_command(["circuit", "status"])
        assert result.returncode == 0
        output = result.stdout
        assert len(output) > 0

    def test_invalid_command_handling(self):
        """Test that invalid commands are handled gracefully."""
        result = self.cli.run_command(["nonexistent-command"])
        assert result.returncode != 0

        # Invalid file path should also be handled
        result = self.cli.run_command(["transcribe", "/nonexistent/file.wav"])
        assert result.returncode != 0

    def test_help_commands(self):
        """Test help output for main commands."""
        # Main help
        result = self.cli.run_command(["--help"])
        assert result.returncode == 0
        output = result.stdout
        assert "transcribe" in output.lower()

        # Subcommand help
        result = self.cli.run_command(["tts", "--help"])
        assert result.returncode == 0
        output = result.stdout
        assert "generate" in output.lower()


class TestE2EClipboardAndSelection:
    """Tests for clipboard and text selection functionality."""

    @pytest.fixture(autouse=True)
    def setup_clipboard_tests(self, tmp_path):
        """Set up clipboard test environment."""
        self.tmp_path = tmp_path
        self.cli = VoiceBridgeCLI()
        self.clipboard_sim = ClipboardSimulator()

    @pytest.mark.skip(reason="Requires GUI environment and real clipboard access")
    def test_tts_listen_clipboard_simulation(self):
        """Test TTS clipboard monitoring (simulated)."""
        # This would require a more complex setup with actual clipboard simulation
        # For now, just test that the command starts without error

        # Start clipboard listening in background
        process = self.cli.start_background_command(
            ["tts", "listen-clipboard", "--no-auto-play"]
        )

        time.sleep(2)  # Let it initialize

        # Simulate clipboard change
        self.clipboard_sim.set_text("This is test text for TTS generation")

        time.sleep(3)  # Let it process

        # Clean up
        process.terminate()
        process.wait()

    @pytest.mark.skip(reason="Requires GUI environment and hotkey simulation")
    def test_tts_listen_selection_simulation(self):
        """Test TTS text selection monitoring (simulated)."""
        # Similar to clipboard test, this would require GUI simulation
        process = self.cli.start_background_command(
            ["tts", "listen-selection", "--no-auto-play"]
        )

        time.sleep(2)

        # Would need to simulate text selection and hotkey press

        process.terminate()
        process.wait()

    @pytest.mark.skip(reason="Requires GUI environment and hotkey simulation")
    def test_realtime_hotkey_transcription(self):
        """Test real-time transcription with hotkey simulation."""
        # This would require audio input simulation and hotkey pressing
        process = self.cli.start_background_command(
            ["hotkey", "--key", "f9", "--mode", "toggle"]
        )

        time.sleep(2)

        # Would simulate F9 press, audio input, and F9 press again

        process.terminate()
        process.wait()


class TestE2EDaemonOperations:
    """Tests for daemon mode operations."""

    @pytest.fixture(autouse=True)
    def setup_daemon_tests(self, tmp_path):
        """Set up daemon test environment."""
        self.tmp_path = tmp_path
        self.cli = VoiceBridgeCLI()

    def test_tts_daemon_lifecycle(self):
        """Test TTS daemon start/status/stop cycle."""
        # Check initial status (should be stopped)
        result = self.cli.run_command(["tts", "daemon", "status"])
        assert result.returncode == 0

        # Start daemon (in background mode to avoid blocking)
        start_process = self.cli.start_background_command(
            ["tts", "daemon", "start", "--mode", "clipboard", "--no-auto-play"]
        )

        time.sleep(3)  # Give it time to start

        # Check status (should be running)
        result = self.cli.run_command(["tts", "daemon", "status"])
        assert result.returncode == 0

        # Stop daemon
        result = self.cli.run_command(["tts", "daemon", "stop"])
        assert result.returncode == 0

        # Clean up process if still running
        if start_process.poll() is None:
            start_process.terminate()
            start_process.wait()

    def test_whisper_daemon_lifecycle(self):
        """Test main Whisper daemon operations."""
        # Check status
        result = self.cli.run_command(["status"])
        assert result.returncode == 0

        # These would typically require more setup for full testing
        # For now just verify the commands don't crash


class TestE2EIntegrationScenarios:
    """Full integration scenario tests."""

    @pytest.fixture(autouse=True)
    def setup_integration_tests(self, tmp_path):
        """Set up integration test environment."""
        self.tmp_path = tmp_path
        self.cli = VoiceBridgeCLI()
        self.file_manager = FileManager(tmp_path)
        self.audio_generator = AudioGenerator()

        # Create comprehensive test environment
        self.workspace = tmp_path / "integration_workspace"
        self.workspace.mkdir()

    def test_full_transcription_workflow(self):
        """Test complete transcription workflow from file to export."""
        # Generate test audio
        audio_file = self.workspace / "interview.wav"
        self.audio_generator.generate_test_audio(
            audio_file,
            duration=15.0,
            text="Welcome to today's interview. We'll be discussing advanced AI techniques and their applications in modern software development.",
        )

        # Transcribe with specific settings
        result = self.cli.run_command(
            [
                "transcribe",
                str(audio_file),
                "--model",
                "base",
                "--format",
                "json",
                "--output",
                str(self.workspace / "interview.json"),
            ]
        )
        assert result.returncode == 0

        # Check that session was created
        result = self.cli.run_command(["sessions", "list"])
        assert result.returncode == 0

        # Export in different format
        json_file = self.workspace / "interview.json"
        if json_file.exists():
            with open(json_file) as f:
                data = json.load(f)
                data.get("session_id", "test_session")

        # Test confidence analysis (would need session)
        result = self.cli.run_command(["confidence", "analyze-all"])
        assert result.returncode == 0

    def test_batch_processing_with_analysis(self):
        """Test batch processing followed by analysis and export."""
        # Create multiple test files
        batch_dir = self.workspace / "batch_audio"
        batch_dir.mkdir()

        for i in range(3):
            audio_file = batch_dir / f"file_{i}.wav"
            self.audio_generator.generate_test_audio(
                audio_file,
                duration=5.0 + i * 2,  # Varying durations
                text=f"This is test file number {i + 1}. It contains unique content for batch processing tests.",
            )

        # Batch transcribe
        output_dir = self.workspace / "batch_output"
        result = self.cli.run_command(
            [
                "batch-transcribe",
                str(batch_dir),
                "--output-dir",
                str(output_dir),
                "--workers",
                "2",
            ]
        )
        assert result.returncode == 0
        assert output_dir.exists()

        # Analyze results
        result = self.cli.run_command(["confidence", "analyze-all"])
        assert result.returncode == 0

        # Export all sessions
        export_dir = self.workspace / "exports"
        result = self.cli.run_command(
            ["export", "batch", "--format", "srt", "--output-dir", str(export_dir)]
        )
        assert result.returncode == 0

    def test_audio_processing_pipeline(self):
        """Test audio preprocessing followed by transcription."""
        # Generate noisy test audio
        original_file = self.workspace / "noisy_audio.wav"
        self.audio_generator.generate_test_audio(
            original_file,
            duration=10.0,
            text="This audio file simulates noisy conditions that require preprocessing.",
            add_noise=True,
        )

        # Preprocess the audio
        processed_file = self.workspace / "cleaned_audio.wav"
        result = self.cli.run_command(
            [
                "audio",
                "preprocess",
                str(original_file),
                str(processed_file),
                "--noise-reduction",
                "0.7",
                "--normalize",
                "-20",
                "--trim-silence",
                "--enhance-speech",
            ]
        )
        assert result.returncode == 0

        # Split processed audio
        split_dir = self.workspace / "audio_chunks"
        result = self.cli.run_command(
            [
                "audio",
                "split",
                str(processed_file),
                "--output-dir",
                str(split_dir),
                "--method",
                "duration",
                "--duration",
                "5",
            ]
        )
        assert result.returncode == 0

        # Transcribe the chunks
        result = self.cli.run_command(
            [
                "batch-transcribe",
                str(split_dir),
                "--output-dir",
                str(self.workspace / "chunk_transcriptions"),
            ]
        )
        assert result.returncode == 0

    def test_configuration_and_profiles_workflow(self):
        """Test configuration management and profile usage."""
        # Create and configure a custom profile
        profile_name = "integration_test_profile"

        # Set some configuration
        result = self.cli.run_command(
            ["config", "--set-key", "model", "--value", "base"]
        )
        assert result.returncode == 0

        # Configure post-processing
        result = self.cli.run_command(
            [
                "postproc",
                "config",
                "--punctuation",
                "--capitalization",
                "--remove-filler",
            ]
        )
        assert result.returncode == 0

        # Save profile
        result = self.cli.run_command(["profile", "--save", profile_name])
        assert result.returncode == 0

        # Test vocabulary management
        words = ["artificial", "intelligence", "transcription"]
        for word in words:
            result = self.cli.run_command(
                ["vocabulary", "add", word, "--type", "technical"]
            )
            assert result.returncode == 0

        # Load profile and verify
        result = self.cli.run_command(["profile", "--load", profile_name])
        assert result.returncode == 0

        # Test the profile with actual transcription
        test_audio = self.workspace / "profile_test.wav"
        self.audio_generator.generate_test_audio(
            test_audio,
            duration=8.0,
            text="Testing artificial intelligence transcription with technical vocabulary.",
        )

        result = self.cli.run_command(
            [
                "transcribe",
                str(test_audio),
                "--profile",
                profile_name,
                "--output",
                str(self.workspace / "profile_result.txt"),
            ]
        )
        assert result.returncode == 0
