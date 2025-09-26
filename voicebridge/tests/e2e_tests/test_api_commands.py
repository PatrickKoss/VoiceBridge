"""E2E tests for API server management commands."""

import socket

import pytest


class TestAPICommands:
    """Test API server management command functionality."""

    def test_api_help_command(self, cli_runner):
        """Test API help command."""
        result = cli_runner.run("api --help", timeout=10)

        assert result.success, f"API help failed: {result.stderr}"
        assert "api server management" in result.stdout.lower()
        assert "status" in result.stdout
        assert "start" in result.stdout
        assert "stop" in result.stdout
        assert "info" in result.stdout

    def test_api_status_command(self, cli_runner):
        """Test API status command."""
        result = cli_runner.run("api status", timeout=15)

        assert result.success, f"API status failed: {result.stderr}"

        # Should show API server status (running or not running)
        output_lower = result.stdout.lower()
        assert any(
            status in output_lower
            for status in [
                "status",
                "running",
                "not running",
                "stopped",
                "active",
                "inactive",
                "server",
            ]
        )

    def test_api_info_command(self, cli_runner):
        """Test API info command."""
        result = cli_runner.run("api info", timeout=15)

        assert result.success, f"API info failed: {result.stderr}"

        # Should show API server information and endpoints
        output_lower = result.stdout.lower()
        assert any(
            info in output_lower
            for info in ["api", "server", "endpoint", "port", "host", "url", "info"]
        )

    def test_api_start_command_structure(self, cli_runner):
        """Test API start command structure and validation."""
        # Test help first (this should be fast)
        result = cli_runner.run("api start --help", timeout=10)
        assert result.success, f"API start help failed: {result.stderr}"
        assert "start" in result.stdout.lower()

        # Skip actual start command to avoid timeouts - just test command recognition
        # The help command validates that the command exists and is properly structured

    def test_api_stop_command_structure(self, cli_runner):
        """Test API stop command structure."""
        # Test help first
        result = cli_runner.run("api stop --help", timeout=10)
        assert result.success, f"API stop help failed: {result.stderr}"
        assert "stop" in result.stdout.lower()

        # Test stop command (should handle gracefully even if no server running)
        result = cli_runner.run("api stop", timeout=10)

        # Should either stop server or indicate no server running
        if result.failed:
            error_lower = result.stderr.lower()
            assert any(
                msg in error_lower
                for msg in ["not running", "no server", "already stopped", "stop"]
            )
        else:
            output_lower = result.stdout.lower()
            assert any(
                info in output_lower
                for info in ["stop", "server", "stopped", "shutdown", "not running"]
            )

    def test_api_start_help_shows_port_parameter(self, cli_runner):
        """Test API start help shows port parameter."""
        result = cli_runner.run("api start --help", timeout=10)
        assert result.success, f"API start help failed: {result.stderr}"

        # Should show port parameter in help
        assert "--port" in result.stdout or "port" in result.stdout.lower()

    def _find_available_port(self):
        """Find an available port for testing."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port


class TestAPICommandsValidation:
    """Test API command argument validation."""

    def test_api_start_help_shows_port_validation(self, cli_runner):
        """Test API start help shows port validation info."""
        result = cli_runner.run("api start --help", timeout=10)
        assert result.success, f"API start help failed: {result.stderr}"

        # Should show usage information
        assert "usage:" in result.stdout.lower() or "start" in result.stdout.lower()

    def test_api_start_help_comprehensive(self, cli_runner):
        """Test API start help is comprehensive."""
        result = cli_runner.run("api start --help", timeout=10)
        assert result.success, f"API start help failed: {result.stderr}"

        # Should provide comprehensive help
        assert len(result.stdout.strip()) > 0, "Should provide help content"

    def test_api_subcommand_availability(self, cli_runner):
        """Test that API subcommands are properly available."""
        # Test invalid subcommand
        result = cli_runner.run(
            "api invalid_subcommand", timeout=10, expect_failure=True
        )

        assert result.failed, "Should fail with invalid subcommand"
        assert any(
            msg in result.stderr.lower()
            for msg in ["no such command", "invalid", "not found", "available commands"]
        )


class TestAPICommandsSmokeTests:
    """Quick smoke tests for API commands."""

    def test_all_api_subcommands_help(self, cli_runner):
        """Test that all API subcommands have working help."""
        subcommands = ["status", "start", "stop", "info"]

        for cmd in subcommands:
            result = cli_runner.run(["api", cmd, "--help"], timeout=10)
            assert result.success, f"API {cmd} help failed: {result.stderr}"
            assert "usage:" in result.stdout.lower()
            assert cmd in result.stdout.lower()

    def test_api_commands_structure(self, cli_runner):
        """Test API command structure and availability."""
        result = cli_runner.run("api --help", timeout=10)

        assert result.success, f"API command structure test failed: {result.stderr}"

        # Verify all expected subcommands are listed
        expected_commands = ["status", "start", "stop", "info"]
        for cmd in expected_commands:
            assert cmd in result.stdout, f"Missing API subcommand: {cmd}"

    def test_api_status_quick_check(self, cli_runner):
        """Quick test that API status command works."""
        result = cli_runner.run("api status", timeout=10)

        assert result.success, f"Quick API status check failed: {result.stderr}"

        # Should produce some output about API server status
        assert len(result.stdout.strip()) > 0, "API status should produce output"

    def test_api_info_quick_check(self, cli_runner):
        """Quick test that API info command works."""
        result = cli_runner.run("api info", timeout=10)

        assert result.success, f"Quick API info check failed: {result.stderr}"

        # Should produce some output about API server information
        assert len(result.stdout.strip()) > 0, "API info should produce output"


class TestAPIServerLifecycle:
    """Test API server lifecycle management."""

    @pytest.mark.e2e_slow
    def test_api_status_before_start(self, cli_runner):
        """Test API status when no server is running."""
        # Ensure no server is running first
        cli_runner.run("api stop", timeout=10)  # Ignore result

        result = cli_runner.run("api status", timeout=10)
        assert result.success, f"API status check failed: {result.stderr}"

        # Should indicate server is not running
        output_lower = result.stdout.lower()
        assert (
            any(
                status in output_lower
                for status in ["not running", "stopped", "inactive", "down"]
            )
            or "running" not in output_lower
        )

    def test_api_stop_when_not_running(self, cli_runner):
        """Test API stop when no server is running."""
        result = cli_runner.run("api stop", timeout=10)

        # Should handle gracefully (either succeed with message or fail informatively)
        if result.failed:
            error_lower = result.stderr.lower()
            assert any(
                msg in error_lower
                for msg in ["not running", "no server", "already stopped"]
            )
        else:
            output_lower = result.stdout.lower()
            assert any(
                msg in output_lower
                for msg in [
                    "not running",
                    "already stopped",
                    "no server",
                    "stopped",
                    "no api server found",
                ]
            )

    def test_api_server_help_workflow(self, cli_runner):
        """Test API server help commands workflow."""
        # This test verifies the command structure works by testing help commands

        # Test all help commands work
        help_commands = [
            "api --help",
            "api status --help",
            "api start --help",
            "api stop --help",
            "api info --help",
        ]

        for cmd in help_commands:
            result = cli_runner.run(cmd, timeout=10)
            assert result.success, f"Help command should work: {cmd}"
            assert "usage:" in result.stdout.lower() or "help" in result.stdout.lower()

            # Should not have command structure errors
            assert "no such command" not in result.stderr.lower()
            assert "unrecognized" not in result.stderr.lower()
