"""CLI runner utilities for E2E testing."""

import json
import os
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class CLIResult:
    """Result of a CLI command execution."""

    def __init__(
        self,
        returncode: int,
        stdout: str,
        stderr: str,
        command: list[str],
        execution_time: float,
    ):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.command = command
        self.execution_time = execution_time

    @property
    def success(self) -> bool:
        """Whether the command succeeded."""
        return self.returncode == 0

    @property
    def failed(self) -> bool:
        """Whether the command failed."""
        return self.returncode != 0

    def assert_success(self) -> "CLIResult":
        """Assert that the command succeeded."""
        if self.failed:
            raise AssertionError(
                f"Command failed with exit code {self.returncode}\n"
                f"Command: {' '.join(self.command)}\n"
                f"STDOUT: {self.stdout}\n"
                f"STDERR: {self.stderr}"
            )
        return self

    def assert_failure(self, expected_code: int | None = None) -> "CLIResult":
        """Assert that the command failed."""
        if self.success:
            raise AssertionError(
                f"Command unexpectedly succeeded\n"
                f"Command: {' '.join(self.command)}\n"
                f"STDOUT: {self.stdout}"
            )
        if expected_code is not None and self.returncode != expected_code:
            raise AssertionError(
                f"Command failed with exit code {self.returncode}, expected {expected_code}\n"
                f"Command: {' '.join(self.command)}\n"
                f"STDERR: {self.stderr}"
            )
        return self


class CLIRunner:
    """Runner for VoiceBridge CLI commands in E2E tests."""

    def __init__(self, test_dir: Path, timeout: float = 30.0):
        """Initialize CLI runner.

        Args:
            test_dir: Isolated test directory
            timeout: Default timeout for commands
        """
        self.test_dir = Path(test_dir)
        self.timeout = timeout
        self._setup_environment()

    def _setup_environment(self):
        """Set up test environment."""
        self.env = dict(os.environ)
        self.env.update(
            {
                "VOICEBRIDGE_TEST_MODE": "1",
                "VOICEBRIDGE_DISABLE_AUDIO": "1",
                "VOICEBRIDGE_NO_GUI": "1",
                "HOME": str(self.test_dir),
                "XDG_CONFIG_HOME": str(self.test_dir / ".config"),
                "XDG_DATA_HOME": str(self.test_dir / ".local" / "share"),
                # Disable colored output for predictable parsing
                "NO_COLOR": "1",
                "FORCE_COLOR": "0",
                "TERM": "dumb",  # Disable rich formatting
                "_TYPER_STANDARD_TRACEBACK": "1",  # Disable rich tracebacks
            }
        )

        # Create config directories
        (self.test_dir / ".config").mkdir(parents=True, exist_ok=True)
        (self.test_dir / ".local" / "share").mkdir(parents=True, exist_ok=True)

    def run(
        self,
        command: str | list[str],
        timeout: float | None = None,
        cwd: Path | None = None,
        input_data: str | None = None,
        expect_failure: bool = False,
    ) -> CLIResult:
        """Run a VoiceBridge CLI command.

        Args:
            command: Command to run (string or list)
            timeout: Command timeout (uses default if None)
            cwd: Working directory
            input_data: Input to pass to command
            expect_failure: Whether to expect command failure

        Returns:
            CLIResult with execution details
        """
        # Find project root and construct correct python path
        project_root = Path(__file__).parent.parent.parent.parent.parent
        python_path = project_root / ".venv" / "bin" / "python"

        if isinstance(command, str):
            # Parse command string to list, handling the CLI invocation
            if command.startswith("voicebridge "):
                command = command.replace("voicebridge ", "", 1)
            cmd_parts = [str(python_path), "-m", "voicebridge"] + command.split()
        else:
            cmd_parts = [str(python_path), "-m", "voicebridge"] + list(command)

        timeout = timeout or self.timeout
        cwd = cwd or project_root

        import time

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd_parts,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(cwd),
                env=self.env,
                input=input_data,
            )
            execution_time = time.time() - start_time

            cli_result = CLIResult(
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                command=cmd_parts,
                execution_time=execution_time,
            )

            if not expect_failure and cli_result.failed:
                # Log failure details for debugging
                print(f"CLI command failed: {' '.join(cmd_parts)}")
                print(f"Exit code: {result.returncode}")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")

            return cli_result

        except subprocess.TimeoutExpired as e:
            execution_time = time.time() - start_time
            raise TimeoutError(
                f"Command timed out after {timeout}s: {' '.join(cmd_parts)}"
            ) from e

    def run_stt(self, subcommand: str, *args, **kwargs) -> CLIResult:
        """Run an STT subcommand.

        Args:
            subcommand: STT subcommand (e.g., 'transcribe', 'batch-transcribe')
            *args: Additional arguments
            **kwargs: Passed to run()

        Returns:
            CLIResult
        """
        cmd = ["stt", subcommand] + list(args)
        return self.run(cmd, **kwargs)

    def run_tts(self, subcommand: str, *args, **kwargs) -> CLIResult:
        """Run a TTS subcommand.

        Args:
            subcommand: TTS subcommand
            *args: Additional arguments
            **kwargs: Passed to run()

        Returns:
            CLIResult
        """
        cmd = ["tts", subcommand] + list(args)
        return self.run(cmd, **kwargs)

    def run_audio(self, subcommand: str, *args, **kwargs) -> CLIResult:
        """Run an audio subcommand.

        Args:
            subcommand: Audio subcommand
            *args: Additional arguments
            **kwargs: Passed to run()

        Returns:
            CLIResult
        """
        cmd = ["audio", subcommand] + list(args)
        return self.run(cmd, **kwargs)

    @contextmanager
    def temp_file(self, content: str = "", suffix: str = ".txt"):
        """Create a temporary file for testing.

        Args:
            content: File content
            suffix: File extension

        Yields:
            Path to temporary file
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, dir=self.test_dir, delete=False
        ) as f:
            f.write(content)
            temp_path = Path(f.name)

        try:
            yield temp_path
        finally:
            temp_path.unlink(missing_ok=True)

    @contextmanager
    def temp_dir(self):
        """Create a temporary directory for testing.

        Yields:
            Path to temporary directory
        """
        temp_path = self.test_dir / f"temp_{os.getpid()}"
        temp_path.mkdir(exist_ok=True)

        try:
            yield temp_path
        finally:
            import shutil

            shutil.rmtree(temp_path, ignore_errors=True)

    def parse_json_output(self, result: CLIResult) -> dict[str, Any]:
        """Parse JSON output from CLI result.

        Args:
            result: CLI result with JSON output

        Returns:
            Parsed JSON data

        Raises:
            json.JSONDecodeError: If output is not valid JSON
        """
        return json.loads(result.stdout.strip())

    def get_config_path(self, config_name: str = "config.json") -> Path:
        """Get path to test config file.

        Args:
            config_name: Config file name

        Returns:
            Path to config file
        """
        config_dir = self.test_dir / ".config" / "voicebridge"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / config_name

    def create_test_config(self, config: dict[str, Any]) -> Path:
        """Create a test configuration file.

        Args:
            config: Configuration dictionary

        Returns:
            Path to created config file
        """
        config_path = self.get_config_path()
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        return config_path
