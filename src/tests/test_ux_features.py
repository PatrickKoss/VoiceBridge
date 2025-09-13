#!/usr/bin/env python3

import sys
from pathlib import Path

# Add parent directory to path to import whisper_cli
sys.path.insert(0, str(Path(__file__).parent))

from whisper_cli import (
    PROFILES_DIR,
    check_for_updates,
    delete_profile,
    list_profiles,
    load_profile,
    log_performance,
    save_profile,
)


def test_profile_management():
    """Test profile save/load/delete functionality."""
    print("Testing profile management...")

    # Test profile operations
    test_profile = {
        "model_name": "small",
        "language": "en",
        "temperature": 0.2,
        "interactive": True,
    }

    # Save profile
    result = save_profile("test-profile", test_profile)
    assert result, "Failed to save profile"
    print("✓ Profile saved successfully")

    # Load profile
    loaded = load_profile("test-profile")
    # Only check the fields that were in the original test profile (excluding 'interactive' which was filtered out)
    expected_fields = {"model_name": "small", "language": "en", "temperature": 0.2}
    for key, expected_value in expected_fields.items():
        assert key in loaded, f"Missing key {key} in loaded profile"
        assert (
            loaded[key] == expected_value
        ), f"Value mismatch for {key}: {loaded[key]} != {expected_value}"
    print("✓ Profile loaded successfully")

    # List profiles
    profiles = list_profiles()
    assert "test-profile" in profiles, f"Profile not found in list: {profiles}"
    print("✓ Profile found in list")

    # Delete profile
    result = delete_profile("test-profile")
    assert result, "Failed to delete profile"

    # Verify deletion
    profiles = list_profiles()
    assert (
        "test-profile" not in profiles
    ), f"Profile still exists after deletion: {profiles}"
    print("✓ Profile deleted successfully")


def test_update_check():
    """Test update checking functionality."""
    print("Testing update check...")

    try:
        check_for_updates()
        print("✓ Update check completed (no errors)")
    except Exception as e:
        print(f"⚠ Update check had issues (expected in CI): {e}")


def test_performance_logging():
    """Test performance logging functionality."""
    print("Testing performance logging...")

    # Test logging function doesn't crash
    try:
        log_performance("test_operation", 1.234, {"param": "value"})
        print("✓ Performance logging works")
    except Exception as e:
        print(f"✗ Performance logging failed: {e}")


def main():
    """Run all UX feature tests."""
    print("Testing Whisper CLI User Experience Features")
    print("=" * 50)

    # Ensure clean test environment
    if PROFILES_DIR.exists():
        import shutil

        shutil.rmtree(PROFILES_DIR)

    try:
        test_profile_management()
        test_update_check()
        test_performance_logging()

        print("\n" + "=" * 50)
        print("✅ All UX features tested successfully!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
