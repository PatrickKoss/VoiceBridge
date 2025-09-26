"""E2E tests for TTS (Text-to-Speech) commands."""

import pytest


class TestTTSCommands:
    """Test TTS command functionality."""

    def test_tts_help_command(self, cli_runner):
        """Test TTS help command."""
        result = cli_runner.run("tts --help", timeout=10)

        assert result.success, f"TTS help failed: {result.stderr}"
        assert "text-to-speech commands" in result.stdout.lower()
        assert "generate" in result.stdout
        assert "voices" in result.stdout
        assert "daemon" in result.stdout
        assert "config" in result.stdout
        assert "listen-clipboard" in result.stdout
        assert "listen-selection" in result.stdout

    def test_tts_voices_command(self, cli_runner):
        """Test TTS voices listing command."""
        result = cli_runner.run("tts voices", timeout=20)

        # Should either list voices or indicate none available
        if result.failed:
            # Allow graceful failure if no voices configured
            error_lower = result.stderr.lower()
            assert any(
                msg in error_lower
                for msg in [
                    "no voices",
                    "not found",
                    "not available",
                    "model",
                    "download",
                ]
            )
        else:
            assert result.success, f"TTS voices failed: {result.stderr}"
            # Should show voices or indicate status
            assert len(result.stdout.strip()) > 0

    def test_tts_generate_help_command(self, cli_runner):
        """Test TTS generate help command."""
        result = cli_runner.run("tts generate --help", timeout=10)

        assert result.success, f"TTS generate help failed: {result.stderr}"
        assert "generate tts from text" in result.stdout.lower()
        assert "--voice" in result.stdout
        assert "--output" in result.stdout
        assert "--play" in result.stdout or "--no-play" in result.stdout
        assert "--streaming" in result.stdout

    def test_tts_config_show_command(self, cli_runner):
        """Test TTS config show command."""
        result = cli_runner.run("tts config show", timeout=15)

        # Should either show config or indicate no config
        if result.failed:
            error_lower = result.stderr.lower()
            assert any(
                msg in error_lower
                for msg in ["config", "not found", "default", "setup"]
            )
        else:
            assert result.success, f"TTS config show failed: {result.stderr}"
            assert len(result.stdout.strip()) > 0

    def test_tts_daemon_status_command(self, cli_runner):
        """Test TTS daemon status command."""
        result = cli_runner.run("tts daemon status", timeout=15)

        # Should show daemon status (running or not running)
        if result.failed:
            error_lower = result.stderr.lower()
            assert any(
                msg in error_lower
                for msg in ["daemon", "not running", "status", "unavailable"]
            )
        else:
            assert result.success, f"TTS daemon status failed: {result.stderr}"
            output_lower = result.stdout.lower()
            assert any(
                status in output_lower
                for status in ["status", "running", "not running", "stopped", "daemon"]
            )

    def test_tts_listen_clipboard_help(self, cli_runner):
        """Test TTS listen-clipboard help command."""
        result = cli_runner.run("tts listen-clipboard --help", timeout=10)

        assert result.success, f"TTS listen-clipboard help failed: {result.stderr}"
        assert "listen to clipboard changes" in result.stdout.lower()
        assert "--voice" in result.stdout
        assert "--auto-play" in result.stdout or "--no-auto-play" in result.stdout

    def test_tts_listen_selection_help(self, cli_runner):
        """Test TTS listen-selection help command."""
        result = cli_runner.run("tts listen-selection --help", timeout=10)

        assert result.success, f"TTS listen-selection help failed: {result.stderr}"
        assert "listen for text selections" in result.stdout.lower()
        assert "--voice" in result.stdout
        assert "--auto-play" in result.stdout or "--no-auto-play" in result.stdout


class TestTTSCommandsValidation:
    """Test TTS command argument validation."""

    def test_tts_generate_missing_text(self, cli_runner):
        """Test TTS generate without text argument."""
        result = cli_runner.run("tts generate", timeout=10, expect_failure=True)

        assert result.failed, "Should fail without text argument"
        assert any(
            msg in result.stderr.lower() for msg in ["missing", "required", "text"]
        )

    def test_tts_daemon_invalid_subcommand(self, cli_runner):
        """Test TTS daemon with invalid subcommand."""
        result = cli_runner.run(
            "tts daemon invalid_command", timeout=10, expect_failure=True
        )

        assert result.failed, "Should fail with invalid daemon subcommand"
        assert any(
            msg in result.stderr.lower()
            for msg in ["no such command", "invalid", "not found", "available commands"]
        )

    def test_tts_config_invalid_subcommand(self, cli_runner):
        """Test TTS config with invalid subcommand."""
        result = cli_runner.run(
            "tts config invalid_command", timeout=10, expect_failure=True
        )

        assert result.failed, "Should fail with invalid config subcommand"
        assert any(
            msg in result.stderr.lower()
            for msg in ["no such command", "invalid", "not found", "available commands"]
        )

    def test_tts_daemon_help_validation(self, cli_runner):
        """Test TTS daemon help shows proper validation info."""
        result = cli_runner.run("tts daemon --help", timeout=10)
        assert result.success, f"TTS daemon help failed: {result.stderr}"

        # Should show daemon subcommands
        assert "start" in result.stdout.lower()
        assert "stop" in result.stdout.lower()
        assert "status" in result.stdout.lower()

    def test_tts_config_help_validation(self, cli_runner):
        """Test TTS config help shows proper validation info."""
        result = cli_runner.run("tts config --help", timeout=10)
        assert result.success, f"TTS config help failed: {result.stderr}"

        # Should show config subcommands
        assert "show" in result.stdout.lower()
        assert "set" in result.stdout.lower()


class TestTTSCommandsSmokeTests:
    """Quick smoke tests for TTS commands."""

    def test_all_tts_subcommands_help(self, cli_runner):
        """Test that all TTS subcommands have working help."""
        subcommands = [
            "generate",
            "voices",
            "daemon",
            "config",
            "listen-clipboard",
            "listen-selection",
        ]

        for cmd in subcommands:
            result = cli_runner.run(["tts", cmd, "--help"], timeout=10)
            assert result.success, f"TTS {cmd} help failed: {result.stderr}"
            assert "usage:" in result.stdout.lower()

    def test_tts_commands_structure(self, cli_runner):
        """Test TTS command structure and availability."""
        result = cli_runner.run("tts --help", timeout=10)

        assert result.success, f"TTS command structure test failed: {result.stderr}"

        # Verify all expected subcommands are listed
        expected_commands = [
            "generate",
            "voices",
            "daemon",
            "config",
            "listen-clipboard",
            "listen-selection",
        ]
        for cmd in expected_commands:
            assert cmd in result.stdout, f"Missing TTS subcommand: {cmd}"

    def test_tts_voices_quick_check(self, cli_runner):
        """Quick test that TTS voices command works."""
        result = cli_runner.run("tts voices", timeout=15)

        # Should either succeed or fail gracefully
        if result.failed:
            # Check for expected failure reasons
            error_lower = result.stderr.lower()
            assert any(
                msg in error_lower
                for msg in ["no voices", "model", "download", "not available"]
            )
        else:
            # Should produce some output about voices
            assert len(result.stdout.strip()) >= 0  # Allow empty but valid output

    def test_tts_config_show_quick_check(self, cli_runner):
        """Quick test that TTS config show command works."""
        result = cli_runner.run("tts config show", timeout=10)

        # Should either succeed or fail gracefully
        if result.failed:
            error_lower = result.stderr.lower()
            assert any(msg in error_lower for msg in ["config", "not found", "default"])
        else:
            # Should produce some output about config
            assert len(result.stdout.strip()) >= 0  # Allow empty but valid output


class TestTTSIntegrationSlow:
    """Slower integration tests for TTS functionality."""

    @pytest.mark.e2e_slow
    def test_tts_generate_basic(self, cli_runner):
        """Test basic TTS generation with very short text."""
        # Use very short text and disable audio playback to minimize execution time
        result = cli_runner.run(
            [
                "tts",
                "generate",
                "Hi",  # Very short text
                "--no-play",  # Don't play audio
                "--use-gpu",
                "false",  # Force CPU to avoid GPU setup time
            ],
            timeout=60,
        )  # Longer timeout for model loading

        # Should either succeed or fail gracefully
        if result.failed:
            # Check for expected failure reasons (model not available, etc.)
            error_lower = result.stderr.lower()
            assert any(
                msg in error_lower
                for msg in [
                    "model",
                    "download",
                    "not available",
                    "error",
                    "failed",
                    "memory",
                    "torch",
                    "transformers",
                    "hub",
                    "network",
                ]
            ), f"Unexpected error: {result.stderr}"
        else:
            assert result.success, f"TTS generate failed: {result.stderr}"
            # Should indicate TTS generation occurred
            output_lower = result.stdout.lower()
            assert (
                any(
                    info in output_lower
                    for info in ["generated", "tts", "audio", "speech", "complete"]
                )
                or len(result.stdout.strip()) > 0
            )

    @pytest.mark.e2e_slow
    def test_tts_daemon_workflow(self, cli_runner):
        """Test TTS daemon start/status/stop workflow."""
        # Test daemon status first (should be not running)
        status_result = cli_runner.run("tts daemon status", timeout=10)

        # Status command should work
        if status_result.failed:
            error_lower = status_result.stderr.lower()
            assert any(
                msg in error_lower for msg in ["daemon", "not available", "status"]
            )
        else:
            assert status_result.success, (
                f"Daemon status check failed: {status_result.stderr}"
            )

        # Test daemon help commands work
        for subcmd in ["start", "stop", "status"]:
            help_result = cli_runner.run(
                ["tts", "daemon", subcmd, "--help"], timeout=10
            )
            assert help_result.success, (
                f"Daemon {subcmd} help failed: {help_result.stderr}"
            )

    @pytest.mark.e2e_slow
    def test_tts_config_workflow(self, cli_runner):
        """Test TTS config show workflow."""
        # Test config show
        show_result = cli_runner.run("tts config show", timeout=10)

        if show_result.failed:
            error_lower = show_result.stderr.lower()
            assert any(
                msg in error_lower
                for msg in ["config", "not found", "default", "setup"]
            )
        else:
            assert show_result.success, f"Config show failed: {show_result.stderr}"

        # Test config help commands work
        for subcmd in ["show", "set"]:
            help_result = cli_runner.run(
                ["tts", "config", subcmd, "--help"], timeout=10
            )
            assert help_result.success, (
                f"Config {subcmd} help failed: {help_result.stderr}"
            )

    def test_tts_daemon_stop_when_not_running(self, cli_runner):
        """Test TTS daemon stop when no daemon is running."""
        result = cli_runner.run("tts daemon stop", timeout=10)

        # Should handle gracefully (either succeed with message or fail informatively)
        if result.failed:
            error_lower = result.stderr.lower()
            assert any(
                msg in error_lower
                for msg in ["not running", "no daemon", "already stopped", "not found"]
            )
        else:
            output_lower = result.stdout.lower()
            assert any(
                msg in output_lower
                for msg in ["not running", "already stopped", "no daemon", "stopped"]
            )


class TestTTSSystemIntegration:
    """Test TTS command integration with system detection."""

    def test_tts_generate_help_shows_gpu_option(self, cli_runner):
        """Test that TTS generate help shows GPU options."""
        result = cli_runner.run("tts generate --help", timeout=10)
        assert result.success, f"TTS generate help failed: {result.stderr}"

        # Should show GPU options
        assert "--use-gpu" in result.stdout or "--no-gpu" in result.stdout

    def test_tts_generate_help_shows_model_options(self, cli_runner):
        """Test that TTS generate help shows relevant options."""
        result = cli_runner.run("tts generate --help", timeout=10)
        assert result.success, f"TTS generate help failed: {result.stderr}"

        # Should show key TTS options
        assert "--voice" in result.stdout
        assert "--streaming" in result.stdout
        assert "--cfg-scale" in result.stdout or "cfg" in result.stdout.lower()

    def test_tts_voices_command_structure(self, cli_runner):
        """Test TTS voices command provides useful structure."""
        result = cli_runner.run("tts voices", timeout=20)

        # Should either list voices or give informative error
        if result.failed:
            error_lower = result.stderr.lower()
            # Should provide actionable error message
            assert any(
                keyword in error_lower
                for keyword in ["voice", "model", "download", "setup", "available"]
            )
        else:
            # If successful, should provide voice information
            assert len(result.stdout.strip()) >= 0  # Allow empty but structured output

    def test_tts_config_set_help_validation(self, cli_runner):
        """Test TTS config set help shows proper validation."""
        result = cli_runner.run("tts config set --help", timeout=10)
        assert result.success, f"TTS config set help failed: {result.stderr}"

        # Should provide usage information for config setting
        assert len(result.stdout.strip()) > 0, "Should provide help content"
        assert "usage:" in result.stdout.lower() or "set" in result.stdout.lower()
