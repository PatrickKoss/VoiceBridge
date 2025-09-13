#!/usr/bin/env python3

import sys
import tempfile
from pathlib import Path

from voicebridge.adapters.config import FileProfileRepository
from voicebridge.adapters.logging import FileLogger
from voicebridge.domain.models import PerformanceMetrics, WhisperConfig


def test_profile_management():
    """Test profile save/load/delete functionality."""
    print("Testing profile management...")

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        profiles_dir = Path(temp_dir) / "profiles"
        profile_repo = FileProfileRepository(profiles_dir)

        # Create test config
        test_config = WhisperConfig(
            model_name="small",
            language="en",
            temperature=0.2,
        )

        # Save profile
        profile_repo.save_profile("test-profile", test_config)
        print("✓ Profile saved successfully")

        # Load profile
        loaded_config = profile_repo.load_profile("test-profile")

        # Check the loaded config
        assert loaded_config.model_name == "small", (
            f"Expected 'small', got '{loaded_config.model_name}'"
        )
        assert loaded_config.language == "en", (
            f"Expected 'en', got '{loaded_config.language}'"
        )
        assert loaded_config.temperature == 0.2, (
            f"Expected 0.2, got {loaded_config.temperature}"
        )
        print("✓ Profile loaded successfully")

        # List profiles
        profiles = profile_repo.list_profiles()
        assert "test-profile" in profiles, f"Profile not found in list: {profiles}"
        print("✓ Profile found in list")

        # Delete profile
        result = profile_repo.delete_profile("test-profile")
        assert result, "Failed to delete profile"

        # Verify deletion
        profiles = profile_repo.list_profiles()
        assert "test-profile" not in profiles, (
            f"Profile still exists after deletion: {profiles}"
        )
        print("✓ Profile deleted successfully")


def test_performance_logging():
    """Test performance logging functionality."""
    print("Testing performance logging...")

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = Path(temp_dir) / "test.log"
        performance_log = Path(temp_dir) / "performance.log"
        logger = FileLogger(log_file, performance_log)

        # Test logging function doesn't crash
        try:
            test_metrics = PerformanceMetrics(
                operation="test_operation",
                duration=1.234,
                model_load_time=0.5,
                details={"param": "value"},
            )
            logger.log_performance(test_metrics)
            print("✓ Performance logging works")
        except Exception as e:
            print(f"✗ Performance logging failed: {e}")
            raise


def main():
    """Run all UX feature tests."""
    print("Testing VoiceBridge User Experience Features")
    print("=" * 50)

    try:
        test_profile_management()
        test_performance_logging()

        print("\n" + "=" * 50)
        print("✅ All UX features tested successfully!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
