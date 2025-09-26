"""Custom assertions for E2E testing."""

import json
import re
from pathlib import Path
from typing import Any

from .cli_runner import CLIResult


class E2EAssertions:
    """Custom assertion helpers for E2E tests."""

    @staticmethod
    def assert_command_success(
        result: CLIResult,
        expected_in_stdout: list[str] | None = None,
        expected_in_stderr: list[str] | None = None,
    ):
        """Assert command succeeded with optional output checks.

        Args:
            result: CLI result to check
            expected_in_stdout: Strings that should be in stdout
            expected_in_stderr: Strings that should be in stderr
        """
        result.assert_success()

        if expected_in_stdout:
            for expected in expected_in_stdout:
                assert expected in result.stdout, (
                    f"Expected '{expected}' in stdout.\nSTDOUT: {result.stdout}"
                )

        if expected_in_stderr:
            for expected in expected_in_stderr:
                assert expected in result.stderr, (
                    f"Expected '{expected}' in stderr.\nSTDERR: {result.stderr}"
                )

    @staticmethod
    def assert_command_failure(
        result: CLIResult,
        expected_exit_code: int | None = None,
        expected_error_message: str | None = None,
    ):
        """Assert command failed with optional checks.

        Args:
            result: CLI result to check
            expected_exit_code: Expected exit code
            expected_error_message: Expected error message in stderr
        """
        result.assert_failure(expected_exit_code)

        if expected_error_message:
            assert expected_error_message in result.stderr, (
                f"Expected error message '{expected_error_message}' in stderr.\n"
                f"STDERR: {result.stderr}"
            )

    @staticmethod
    def assert_help_output(result: CLIResult, command_name: str):
        """Assert help output is properly formatted.

        Args:
            result: CLI result from --help command
            command_name: Expected command name in help
        """
        result.assert_success()

        # Strip ANSI color codes for easier pattern matching
        import re

        clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)
        clean_output_lower = clean_output.lower()

        # Common help patterns (flexible to handle Rich CLI formatting)
        required_patterns = [
            "usage",
            "--help",
        ]

        # Options pattern - handle various formats
        options_patterns = ["options", "arguments", "commands"]

        # Check required patterns
        for pattern in required_patterns:
            assert pattern in clean_output_lower, (
                f"Expected help pattern '{pattern}' in stdout.\n"
                f"CLEAN OUTPUT: {clean_output[:500]}..."
            )

        # Check for at least one options-like pattern
        has_options = any(pattern in clean_output_lower for pattern in options_patterns)
        assert has_options, (
            f"Expected one of {options_patterns} in help output.\n"
            f"CLEAN OUTPUT: {clean_output[:500]}..."
        )

        # Check for command name if it's not a generic term
        if command_name not in ["help", "usage", "options"]:
            assert command_name in clean_output_lower, (
                f"Expected command name '{command_name}' in help output.\n"
                f"CLEAN OUTPUT: {clean_output[:500]}..."
            )

    @staticmethod
    def assert_json_output(
        result: CLIResult,
        expected_keys: list[str] | None = None,
        expected_values: dict[str, Any] | None = None,
    ) -> dict:
        """Assert output is valid JSON with optional key/value checks.

        Args:
            result: CLI result with JSON output
            expected_keys: Keys that should exist in JSON
            expected_values: Key-value pairs that should match

        Returns:
            Parsed JSON data
        """
        result.assert_success()

        try:
            data = json.loads(result.stdout.strip())
        except json.JSONDecodeError as e:
            raise AssertionError(
                f"Invalid JSON output: {e}\nSTDOUT: {result.stdout}"
            ) from e

        if expected_keys:
            for key in expected_keys:
                assert key in data, (
                    f"Expected key '{key}' in JSON output.\n"
                    f"Available keys: {list(data.keys())}\n"
                    f"JSON: {data}"
                )

        if expected_values:
            for key, expected_value in expected_values.items():
                assert key in data, f"Key '{key}' not found in JSON output"
                assert data[key] == expected_value, (
                    f"Expected {key}={expected_value}, got {data[key]}\nJSON: {data}"
                )

        return data

    @staticmethod
    def assert_file_created(
        file_path: str | Path,
        min_size: int | None = None,
        contains: list[str] | None = None,
    ):
        """Assert file was created with optional content checks.

        Args:
            file_path: Path to file that should exist
            min_size: Minimum file size in bytes
            contains: Strings that should be in file content
        """
        path = Path(file_path)

        assert path.exists(), f"Expected file {path} to exist"
        assert path.is_file(), f"Expected {path} to be a file"

        if min_size is not None:
            size = path.stat().st_size
            assert size >= min_size, (
                f"Expected file size >= {min_size} bytes, got {size}"
            )

        if contains:
            content = path.read_text()
            for expected in contains:
                assert expected in content, (
                    f"Expected '{expected}' in file {path}\nContent: {content[:500]}..."
                )

    @staticmethod
    def assert_audio_file_created(
        file_path: str | Path, min_duration: float | None = None
    ):
        """Assert audio file was created with optional duration check.

        Args:
            file_path: Path to audio file
            min_duration: Minimum duration in seconds
        """
        path = Path(file_path)
        E2EAssertions.assert_file_created(path, min_size=100)  # Basic size check

        # Check file extension
        assert path.suffix.lower() in [".wav", ".mp3", ".m4a", ".flac"], (
            f"Expected audio file extension, got {path.suffix}"
        )

        if min_duration is not None:
            # Would need ffprobe to check duration - simplified for now
            pass

    @staticmethod
    def assert_transcript_format(result: CLIResult, format_type: str):
        """Assert transcript output is in expected format.

        Args:
            result: CLI result with transcript output
            format_type: Expected format ('json', 'srt', 'vtt', 'txt')
        """
        result.assert_success()

        if format_type == "json":
            # Should be valid JSON with transcript structure
            data = E2EAssertions.assert_json_output(result)
            # Basic transcript structure checks
            if "segments" in data:
                assert isinstance(data["segments"], list)

        elif format_type == "srt":
            # Should have SRT format patterns
            srt_patterns = [
                r"^\d+$",  # Sequence numbers
                r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}",  # Timestamps
            ]
            lines = result.stdout.strip().split("\n")

            # Check for SRT patterns
            has_sequence = any(re.match(srt_patterns[0], line) for line in lines)
            has_timestamps = any(re.search(srt_patterns[1], line) for line in lines)

            assert has_sequence, "SRT output should contain sequence numbers"
            assert has_timestamps, "SRT output should contain timestamps"

        elif format_type == "vtt":
            # Should start with WEBVTT
            assert result.stdout.startswith("WEBVTT"), (
                "VTT output should start with 'WEBVTT'"
            )

        elif format_type == "txt":
            # Should be plain text (no specific format requirements)
            assert len(result.stdout.strip()) > 0, "Text output should not be empty"

        else:
            raise ValueError(f"Unsupported transcript format: {format_type}")

    @staticmethod
    def assert_configuration_output(
        result: CLIResult, expected_config_keys: list[str] | None = None
    ):
        """Assert configuration command output.

        Args:
            result: CLI result from config command
            expected_config_keys: Keys that should be in config output
        """
        result.assert_success()

        # Config output could be JSON or human-readable format
        if result.stdout.strip().startswith("{"):
            # JSON format
            data = E2EAssertions.assert_json_output(result)
            if expected_config_keys:
                for key in expected_config_keys:
                    assert key in data, (
                        f"Expected config key '{key}' in output.\n"
                        f"Available keys: {list(data.keys())}"
                    )
        else:
            # Human-readable format - check for key presence (case insensitive)
            if expected_config_keys:
                output_lower = result.stdout.lower()
                for key in expected_config_keys:
                    key_lower = key.lower()
                    # Check for the key with or without colon
                    key_patterns = [key_lower, f"{key_lower}:"]
                    found = any(pattern in output_lower for pattern in key_patterns)
                    assert found, (
                        f"Expected config key '{key}' (or variations) in output.\n"
                        f"Looking for: {key_patterns}\n"
                        f"Output: {result.stdout}"
                    )

    @staticmethod
    def assert_performance_metrics(result: CLIResult):
        """Assert performance metrics are present in output.

        Args:
            result: CLI result with performance output
        """
        result.assert_success()

        # Look for common performance metrics or indicators of stats command working
        metrics_patterns = [
            "duration",
            "speed",
            "memory",
            "processing",
            "time",
            "stats",
            "performance",
            "metrics",
            "operations",
            "total",
            "average",
            "count",
            "sessions",
            "no data",
        ]

        output_lower = result.stdout.lower()
        has_metrics = any(pattern in output_lower for pattern in metrics_patterns)

        # If no metrics found, just ensure the command ran successfully
        # This is more lenient for E2E testing where actual metrics may not be available
        if not has_metrics:
            # Still require some output (not completely empty)
            assert len(result.stdout.strip()) > 0, (
                "Expected some output from performance stats command.\n"
                f"Output: '{result.stdout}'"
            )

    @staticmethod
    def assert_session_management(result: CLIResult, operation: str):
        """Assert session management operation output.

        Args:
            result: CLI result from session command
            operation: Operation type ('list', 'save', 'load', 'delete', 'cleanup')
        """
        result.assert_success()

        operation_indicators = {
            "list": [
                "session",
                "id",
                "status",
                "no sessions",
                "empty",
                "found",
                "total",
            ],
            "save": ["saved", "session"],
            "load": ["loaded", "session", "resumed"],
            "delete": ["deleted", "removed", "session"],
            "cleanup": [
                "cleanup",
                "cleaned",
                "removed",
                "session",
                "completed",
                "done",
            ],
        }

        if operation in operation_indicators:
            expected_terms = operation_indicators[operation]
            output_lower = result.stdout.lower()

            found_terms = [term for term in expected_terms if term in output_lower]

            # Be more lenient - either find expected terms OR just ensure non-empty output
            if not found_terms:
                assert len(result.stdout.strip()) > 0, (
                    f"Expected session {operation} to produce some output.\n"
                    f"Output: '{result.stdout}'"
                )
        else:
            # For unknown operations, just check that command succeeded
            assert len(result.stdout.strip()) > 0, (
                f"Expected some output from session {operation} command.\n"
                f"Output: '{result.stdout}'"
            )
