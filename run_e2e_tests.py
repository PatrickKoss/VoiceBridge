#!/usr/bin/env python3
"""
VoiceBridge End-to-End Test Runner

This script runs comprehensive E2E tests for all VoiceBridge CLI commands.
It provides different test suites and reporting options.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


class E2ETestRunner:
    """Orchestrate E2E test execution with different configurations."""

    def __init__(self, project_root: Path | None = None):
        """Initialize test runner."""
        self.project_root = project_root or Path(__file__).parent
        self.test_dir = self.project_root / "voicebridge" / "tests"

    def run_test_suite(
        self, suite: str, verbose: bool = False, markers: list[str] | None = None
    ) -> dict[str, any]:
        """
        Run a specific test suite.

        Args:
            suite: Test suite name
            verbose: Enable verbose output
            markers: Pytest markers to filter tests

        Returns:
            Test results dictionary
        """
        print(f"\n🚀 Running {suite} test suite...")

        cmd = [".venv/bin/python", "-m", "pytest", "--disable-warnings"]

        # Add test suite specific arguments
        suite_config = self._get_suite_config(suite)
        cmd.extend(suite_config["args"])

        # Add markers
        if markers:
            for marker in markers:
                cmd.extend(["-m", marker])

        # Add verbosity
        if verbose:
            cmd.append("-v")

        # Add test directory
        cmd.append(str(self.test_dir))

        # Run tests
        start_time = time.time()

        try:
            result = subprocess.run(
                cmd, cwd=self.project_root, capture_output=not verbose, text=True
            )

            execution_time = time.time() - start_time

            success = result.returncode == 0
            if not success and not verbose:
                print(f"Command failed with return code {result.returncode}")
                print(f"stdout: {result.stdout}")
                print(f"stderr: {result.stderr}")

            return {
                "suite": suite,
                "success": success,
                "returncode": result.returncode,
                "execution_time": execution_time,
                "stdout": result.stdout if not verbose else "",
                "stderr": result.stderr if not verbose else "",
            }

        except subprocess.TimeoutExpired:
            return {
                "suite": suite,
                "success": False,
                "returncode": 124,  # Timeout
                "execution_time": time.time() - start_time,
                "error": "Test suite timed out",
            }
        except Exception as e:
            return {
                "suite": suite,
                "success": False,
                "returncode": -1,
                "execution_time": time.time() - start_time,
                "error": str(e),
            }

    def _get_suite_config(self, suite: str) -> dict[str, any]:
        """Get configuration for a specific test suite."""
        configs = {
            "smoke": {
                "description": "Quick smoke tests for all commands",
                "args": [
                    "-k",
                    "test_help_command or test_imports_work or test_cli_help",
                ],
                "timeout": 300,  # 5 minutes
            },
            "core": {
                "description": "Core functionality tests",
                "args": [
                    "voicebridge/tests/test_e2e_simple.py::TestE2ESimple::test_imports_work"
                ],
                "timeout": 60,  # 1 minute
            },
            "stt": {
                "description": "Speech-to-Text specific tests",
                "args": ["test_e2e_stt_commands.py"],
                "timeout": 1200,  # 20 minutes
            },
            "tts": {
                "description": "Text-to-Speech specific tests",
                "args": ["test_e2e_tts_commands.py"],
                "timeout": 1200,  # 20 minutes
            },
            "audio": {
                "description": "Audio processing tests",
                "args": [
                    "test_e2e_audio_system.py",
                    "-k",
                    "TestE2EAudioProcessingCommands",
                ],
                "timeout": 900,  # 15 minutes
            },
            "system": {
                "description": "System and configuration tests",
                "args": [
                    "test_e2e_audio_system.py",
                    "-k",
                    "TestE2ESystem or TestE2EConfiguration",
                ],
                "timeout": 600,  # 10 minutes
            },
            "integration": {
                "description": "Full integration scenarios",
                "args": ["-m", "integration"],
                "timeout": 2400,  # 40 minutes
            },
            "performance": {
                "description": "Performance and scalability tests",
                "args": ["-m", "slow"],
                "timeout": 3600,  # 60 minutes
            },
            "full": {
                "description": "Complete test suite",
                "args": ["--tb=short"],
                "timeout": 4800,  # 80 minutes
            },
        }

        return configs.get(
            suite,
            {
                "description": f"Custom suite: {suite}",
                "args": ["-k", suite],
                "timeout": 1800,
            },
        )

    def run_parallel_suites(
        self, suites: list[str], max_workers: int = 2
    ) -> dict[str, dict]:
        """Run multiple test suites in parallel."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print(
            f"\n🔄 Running {len(suites)} test suites in parallel (max {max_workers} workers)..."
        )

        results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all test suites
            future_to_suite = {
                executor.submit(self.run_test_suite, suite): suite for suite in suites
            }

            # Collect results as they complete
            for future in as_completed(future_to_suite):
                suite = future_to_suite[future]
                try:
                    result = future.result()
                    results[suite] = result

                    status = "✅ PASSED" if result["success"] else "❌ FAILED"
                    print(f"{status} {suite} ({result['execution_time']:.1f}s)")

                except Exception as e:
                    results[suite] = {"suite": suite, "success": False, "error": str(e)}
                    print(f"❌ FAILED {suite} (Exception: {e})")

        return results

    def generate_report(
        self, results: dict[str, dict], output_file: Path | None = None
    ):
        """Generate a comprehensive test report."""
        report = self._create_test_report(results)

        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                f.write(report)
            print(f"\n📊 Report saved to: {output_file}")
        else:
            print("\n" + "=" * 80)
            print("📊 TEST EXECUTION REPORT")
            print("=" * 80)
            print(report)

    def _create_test_report(self, results: dict[str, dict]) -> str:
        """Create formatted test report."""
        total_suites = len(results)
        successful_suites = sum(1 for r in results.values() if r.get("success", False))
        total_time = sum(r.get("execution_time", 0) for r in results.values())

        report = f"""
SUMMARY:
  Total Suites: {total_suites}
  Successful:   {successful_suites}
  Failed:       {total_suites - successful_suites}
  Total Time:   {total_time:.1f}s ({total_time / 60:.1f}m)
  Success Rate: {successful_suites / total_suites * 100:.1f}%

DETAILED RESULTS:
"""

        for suite, result in results.items():
            status = "PASSED" if result.get("success", False) else "FAILED"
            time_str = f"{result.get('execution_time', 0):.1f}s"

            report += f"  {suite:15} {status:6} {time_str:>8}\n"

            if not result.get("success", False) and "error" in result:
                report += f"    Error: {result['error']}\n"

        return report

    def validate_environment(self) -> bool:
        """Validate test environment setup."""
        print("🔍 Validating test environment...")

        checks = [
            ("Python executable", self._check_python()),
            ("UV package manager", self._check_uv()),
            ("VoiceBridge installation", self._check_voicebridge()),
            ("Test dependencies", self._check_test_deps()),
            ("Audio support", self._check_audio_support()),
        ]

        all_passed = True
        for check_name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"  {status} {check_name}")
            if not passed:
                all_passed = False

        return all_passed

    def _check_python(self) -> bool:
        """Check Python version."""
        return sys.version_info >= (3, 10)

    def _check_uv(self) -> bool:
        """Check UV availability."""
        try:
            subprocess.run(["uv", "--version"], capture_output=True, timeout=5)
            return True
        except Exception:
            return False

    def _check_voicebridge(self) -> bool:
        """Check VoiceBridge CLI availability."""
        try:
            result = subprocess.run(
                ["uv", "run", "python", "-m", "voicebridge", "--help"],
                capture_output=True,
                timeout=10,
                cwd=self.project_root,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _check_test_deps(self) -> bool:
        """Check test dependencies."""
        try:
            import importlib.util

            return (
                importlib.util.find_spec("numpy") is not None
                and importlib.util.find_spec("pytest") is not None
            )
        except ImportError:
            return False

    def _check_audio_support(self) -> bool:
        """Check audio processing support."""
        try:
            import importlib.util

            return (
                importlib.util.find_spec("wave") is not None
                and importlib.util.find_spec("numpy") is not None
            )
        except ImportError:
            return False

    def list_available_suites(self):
        """List all available test suites."""
        print("\n📋 Available Test Suites:")
        print("-" * 50)

        for suite, config in {
            "smoke": self._get_suite_config("smoke"),
            "core": self._get_suite_config("core"),
            "stt": self._get_suite_config("stt"),
            "tts": self._get_suite_config("tts"),
            "audio": self._get_suite_config("audio"),
            "system": self._get_suite_config("system"),
            "integration": self._get_suite_config("integration"),
            "performance": self._get_suite_config("performance"),
            "full": self._get_suite_config("full"),
        }.items():
            timeout_min = config.get("timeout", 1800) / 60
            print(
                f"  {suite:12} - {config.get('description', 'No description')} (~{timeout_min:.0f}m)"
            )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run VoiceBridge End-to-End Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_e2e_tests.py smoke              # Quick smoke tests
  python run_e2e_tests.py stt tts            # STT and TTS tests
  python run_e2e_tests.py full --verbose     # Full suite with verbose output
  python run_e2e_tests.py --list             # List available suites
  python run_e2e_tests.py --validate         # Check environment setup
        """,
    )

    parser.add_argument(
        "suites",
        nargs="*",
        help="Test suites to run (smoke, core, stt, tts, audio, system, integration, performance, full)",
    )

    parser.add_argument(
        "--list", action="store_true", help="List available test suites"
    )

    parser.add_argument(
        "--validate", action="store_true", help="Validate test environment setup"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--parallel",
        "-p",
        type=int,
        default=1,
        help="Run suites in parallel (max workers)",
    )

    parser.add_argument(
        "--markers", "-m", nargs="+", help="Pytest markers to filter tests"
    )

    parser.add_argument("--report", type=Path, help="Save test report to file")

    parser.add_argument(
        "--timeout", type=int, help="Override default timeout for test suites (seconds)"
    )

    args = parser.parse_args()

    # Initialize test runner
    runner = E2ETestRunner()

    # Handle special commands
    if args.list:
        runner.list_available_suites()
        return 0

    if args.validate:
        if runner.validate_environment():
            print("\n✅ Environment validation passed!")
            return 0
        else:
            print("\n❌ Environment validation failed!")
            return 1

    # Validate environment before running tests
    if not runner.validate_environment():
        print("\n⚠️  Environment validation failed. Some tests may not work correctly.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != "y":
            return 1

    # Default to smoke tests if no suites specified
    if not args.suites:
        args.suites = ["smoke"]

    # Run test suites
    if len(args.suites) == 1 or args.parallel == 1:
        # Run sequentially
        results = {}
        for suite in args.suites:
            result = runner.run_test_suite(
                suite, verbose=args.verbose, markers=args.markers
            )
            results[suite] = result
    else:
        # Run in parallel
        results = runner.run_parallel_suites(args.suites, args.parallel)

    # Generate report
    runner.generate_report(results, args.report)

    # Return appropriate exit code
    all_passed = all(r.get("success", False) for r in results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
