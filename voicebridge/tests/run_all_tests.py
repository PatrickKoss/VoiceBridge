#!/usr/bin/env python3
"""Test runner for all modular architecture tests."""

import sys
import unittest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all test modules - specific imports instead of * to avoid F403


def run_tests():
    """Run all tests with detailed output."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test modules
    test_modules = [
        "tests.test_domain_models",
        "tests.test_config_adapters",
        "tests.test_audio_adapter",
        "tests.test_transcription_service",
        "tests.test_cli_commands",
        "tests.test_integration",
    ]

    for module in test_modules:
        try:
            tests = loader.loadTestsFromName(module)
            suite.addTest(tests)
            print(f"✓ Loaded tests from {module}")
        except Exception as e:
            print(f"✗ Failed to load tests from {module}: {e}")

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=True)

    print("\n" + "=" * 60)
    print("RUNNING MODULAR ARCHITECTURE TESTS")
    print("=" * 60)

    result = runner.run(suite)

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.failures:
        print("\nFAILURES:")
        for test, trace in result.failures:
            print(f"- {test}: {trace}")

    if result.errors:
        print("\nERRORS:")
        for test, trace in result.errors:
            print(f"- {test}: {trace}")

    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nResult: {'✅ SUCCESS' if success else '❌ FAILED'}")

    return success


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
