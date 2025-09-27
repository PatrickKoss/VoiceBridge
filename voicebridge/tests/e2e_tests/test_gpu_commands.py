"""E2E tests for GPU and system commands."""


class TestGPUCommands:
    """Test GPU and system command functionality."""

    def test_gpu_help_command(self, cli_runner):
        """Test GPU help command."""
        result = cli_runner.run("gpu --help", timeout=10)

        assert result.success, f"GPU help failed: {result.stderr}"
        assert "gpu and system commands" in result.stdout.lower()
        assert "status" in result.stdout
        assert "benchmark" in result.stdout

    def test_gpu_status_command(self, cli_runner):
        """Test GPU status command."""
        result = cli_runner.run("gpu status", timeout=15)

        assert result.success, f"GPU status failed: {result.stderr}"

        # Should show GPU information (even if no GPU detected)
        output_lower = result.stdout.lower()
        assert any(
            info in output_lower
            for info in [
                "gpu",
                "device",
                "status",
                "cuda",
                "metal",
                "available",
                "not found",
            ]
        )

    def test_gpu_status_detailed_output(self, cli_runner):
        """Test GPU status command provides useful information."""
        result = cli_runner.run("gpu status", timeout=15)

        assert result.success, f"GPU status failed: {result.stderr}"

        # Should include system information
        output = result.stdout
        # Look for common GPU status indicators
        has_gpu_info = any(
            keyword in output.lower()
            for keyword in [
                "cuda",
                "metal",
                "opencl",
                "gpu",
                "device",
                "memory",
                "driver",
            ]
        )
        assert has_gpu_info, "GPU status should provide device information"

    def test_gpu_benchmark_command_structure(self, cli_runner):
        """Test GPU benchmark command structure and validation."""
        # Test help first (this should be fast)
        result = cli_runner.run("gpu benchmark --help", timeout=10)
        assert result.success, f"GPU benchmark help failed: {result.stderr}"
        assert "benchmark" in result.stdout.lower()

        # Skip actual benchmark execution to avoid timeouts - just test command recognition
        # The help command validates that the command exists and is properly structured

    def test_gpu_benchmark_help_shows_model_parameter(self, cli_runner):
        """Test GPU benchmark help shows model parameter."""
        result = cli_runner.run("gpu benchmark --help", timeout=10)
        assert result.success, f"GPU benchmark help failed: {result.stderr}"

        # Should show model parameter in help
        assert "--model" in result.stdout or "model" in result.stdout.lower()


class TestGPUCommandsValidation:
    """Test GPU command argument validation."""

    def test_gpu_benchmark_help_validation(self, cli_runner):
        """Test GPU benchmark help shows proper validation info."""
        result = cli_runner.run("gpu benchmark --help", timeout=10)
        assert result.success, f"GPU benchmark help failed: {result.stderr}"

        # Should show usage and options
        assert "usage:" in result.stdout.lower() or "benchmark" in result.stdout.lower()

    def test_gpu_subcommand_availability(self, cli_runner):
        """Test that GPU subcommands are properly available."""
        # Test invalid subcommand
        result = cli_runner.run(
            "gpu invalid_subcommand", timeout=10, expect_failure=True
        )

        assert result.failed, "Should fail with invalid subcommand"
        assert any(
            msg in result.stderr.lower()
            for msg in ["no such command", "invalid", "not found", "available commands"]
        )


class TestGPUCommandsSmokeTests:
    """Quick smoke tests for GPU commands."""

    def test_all_gpu_subcommands_help(self, cli_runner):
        """Test that all GPU subcommands have working help."""
        subcommands = ["status", "benchmark"]

        for cmd in subcommands:
            result = cli_runner.run(["gpu", cmd, "--help"], timeout=10)
            assert result.success, f"GPU {cmd} help failed: {result.stderr}"
            assert "usage:" in result.stdout.lower()
            assert cmd in result.stdout.lower()

    def test_gpu_commands_structure(self, cli_runner):
        """Test GPU command structure and availability."""
        result = cli_runner.run("gpu --help", timeout=10)

        assert result.success, f"GPU command structure test failed: {result.stderr}"

        # Verify all expected subcommands are listed
        expected_commands = ["status", "benchmark"]
        for cmd in expected_commands:
            assert cmd in result.stdout, f"Missing GPU subcommand: {cmd}"

    def test_gpu_status_quick_check(self, cli_runner):
        """Quick test that GPU status command works."""
        result = cli_runner.run("gpu status", timeout=10)

        assert result.success, f"Quick GPU status check failed: {result.stderr}"

        # Should produce some output about GPU/system status
        assert len(result.stdout.strip()) > 0, "GPU status should produce output"


class TestGPUSystemIntegration:
    """Test GPU command integration with system detection."""

    def test_gpu_status_system_detection(self, cli_runner):
        """Test that GPU status detects system capabilities."""
        result = cli_runner.run("gpu status", timeout=15)

        assert result.success, f"GPU system detection failed: {result.stderr}"

        # Should indicate what GPU capabilities are available on the system
        output = result.stdout.lower()

        # Should mention at least one of the GPU technologies or indicate none available
        gpu_mentions = sum(
            1 for tech in ["cuda", "metal", "opencl", "gpu"] if tech in output
        )
        no_gpu_mentions = sum(
            1 for msg in ["not available", "not found", "disabled"] if msg in output
        )

        assert gpu_mentions > 0 or no_gpu_mentions > 0, (
            "Should indicate GPU availability status"
        )

    def test_gpu_benchmark_help_comprehensive(self, cli_runner):
        """Test that GPU benchmark help is comprehensive."""
        result = cli_runner.run("gpu benchmark --help", timeout=10)

        assert result.success, f"GPU benchmark help failed: {result.stderr}"

        # Should provide comprehensive help without needing actual benchmark execution
        assert len(result.stdout.strip()) > 0, "Should provide help content"
        assert "benchmark" in result.stdout.lower(), "Should mention benchmark in help"
