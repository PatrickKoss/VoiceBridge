#!/usr/bin/env python3
"""
Unit tests for VoiceBridge components

Tests cover:
- Configuration management
- Profile management
- Cross-platform functionality
- CLI command structure
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from voicebridge.adapters.config import FileConfigRepository, FileProfileRepository
from voicebridge.adapters.system import PlatformClipboardService
from voicebridge.domain.models import WhisperConfig


class TestConfigurationManagement(unittest.TestCase):
    """Test configuration loading, saving, and management."""

    def setUp(self):
        """Set up test environment."""
        self.test_config_dir = Path(tempfile.mkdtemp())
        self.config_repo = FileConfigRepository(self.test_config_dir)

    def test_load_config_defaults(self):
        """Test loading default configuration."""
        config = self.config_repo.load()

        # Check some expected defaults
        self.assertEqual(config.model_name, "medium")
        self.assertIsNone(config.language)
        self.assertEqual(config.temperature, 0.0)

    def test_load_config_from_file(self):
        """Test loading configuration from existing file."""
        test_config = WhisperConfig(model_name="large", temperature=0.5, language="en")

        # Save config first
        self.config_repo.save(test_config)

        # Load and verify
        loaded_config = self.config_repo.load()
        self.assertEqual(loaded_config.model_name, "large")
        self.assertEqual(loaded_config.temperature, 0.5)
        self.assertEqual(loaded_config.language, "en")

    def test_save_config(self):
        """Test saving configuration to file."""
        test_config = WhisperConfig(model_name="small", debug=True)

        self.config_repo.save(test_config)

        # Verify file exists and contains correct data
        config_file = self.test_config_dir / "config.json"
        self.assertTrue(config_file.exists())

        with open(config_file) as f:
            saved_data = json.load(f)

        self.assertEqual(saved_data["model_name"], "small")
        self.assertEqual(saved_data["debug"], True)


class TestProfileManagement(unittest.TestCase):
    """Test profile management functionality."""

    def setUp(self):
        """Set up test environment."""
        self.test_profiles_dir = Path(tempfile.mkdtemp())
        self.profile_repo = FileProfileRepository(self.test_profiles_dir)

    def test_save_and_load_profile(self):
        """Test saving and loading profiles."""
        test_config = WhisperConfig(model_name="medium", language="es", temperature=0.3)

        # Save profile
        self.profile_repo.save_profile("test-profile", test_config)

        # Load profile
        loaded_config = self.profile_repo.load_profile("test-profile")

        self.assertEqual(loaded_config.model_name, "medium")
        self.assertEqual(loaded_config.language, "es")
        self.assertEqual(loaded_config.temperature, 0.3)

    def test_list_profiles(self):
        """Test listing profiles."""
        # Initially empty
        profiles = self.profile_repo.list_profiles()
        self.assertEqual(len(profiles), 0)

        # Add some profiles
        config1 = WhisperConfig(model_name="small")
        config2 = WhisperConfig(model_name="base")

        self.profile_repo.save_profile("profile1", config1)
        self.profile_repo.save_profile("profile2", config2)

        profiles = self.profile_repo.list_profiles()
        self.assertEqual(len(profiles), 2)
        self.assertIn("profile1", profiles)
        self.assertIn("profile2", profiles)

    def test_delete_profile(self):
        """Test deleting profiles."""
        config = WhisperConfig(model_name="tiny")
        self.profile_repo.save_profile("to-delete", config)

        # Verify it exists
        profiles = self.profile_repo.list_profiles()
        self.assertIn("to-delete", profiles)

        # Delete it
        result = self.profile_repo.delete_profile("to-delete")
        self.assertTrue(result)

        # Verify it's gone
        profiles = self.profile_repo.list_profiles()
        self.assertNotIn("to-delete", profiles)

    def test_delete_nonexistent_profile(self):
        """Test deleting a non-existent profile."""
        result = self.profile_repo.delete_profile("nonexistent")
        self.assertFalse(result)


class TestClipboardFunctionality(unittest.TestCase):
    """Test clipboard functionality."""

    def test_copy_text_logic(self):
        """Test clipboard copy logic (platform-independent)."""
        clipboard_service = PlatformClipboardService()

        # Mock the platform-specific method based on current platform
        with patch.object(
            clipboard_service, "_copy_linux", return_value=True
        ) as mock_linux:
            with patch.object(
                clipboard_service, "_copy_windows", return_value=True
            ) as mock_windows:
                with patch.object(
                    clipboard_service, "_copy_macos", return_value=True
                ) as mock_macos:
                    result = clipboard_service.copy_text("test text")

                    # Verify that one of the platform methods was called
                    call_count = (
                        mock_linux.call_count
                        + mock_windows.call_count
                        + mock_macos.call_count
                    )
                    self.assertEqual(call_count, 1)
                    self.assertTrue(result)

    def test_copy_text_failure(self):
        """Test clipboard copy failure handling."""
        clipboard_service = PlatformClipboardService()

        # Mock all platform methods to raise exceptions
        with patch.object(
            clipboard_service, "_copy_linux", side_effect=Exception("Failed")
        ):
            with patch.object(
                clipboard_service, "_copy_windows", side_effect=Exception("Failed")
            ):
                with patch.object(
                    clipboard_service, "_copy_macos", side_effect=Exception("Failed")
                ):
                    result = clipboard_service.copy_text("test text")
                    self.assertFalse(result)

    def test_copy_empty_text(self):
        """Test copying empty text."""
        clipboard_service = PlatformClipboardService()

        with patch.object(clipboard_service, "_copy_linux", return_value=True):
            with patch.object(clipboard_service, "_copy_windows", return_value=True):
                with patch.object(clipboard_service, "_copy_macos", return_value=True):
                    result = clipboard_service.copy_text("")
                    # Should handle empty text gracefully
                    self.assertIsInstance(result, bool)


class TestModelConfiguration(unittest.TestCase):
    """Test WhisperConfig model behavior."""

    def test_default_config(self):
        """Test default configuration values."""
        config = WhisperConfig()

        self.assertEqual(config.model_name, "medium")
        self.assertIsNone(config.language)
        self.assertEqual(config.temperature, 0.0)
        self.assertFalse(config.debug)

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "model_name": "large",
            "language": "fr",
            "temperature": 0.5,
            "debug": True,
        }

        config = WhisperConfig.from_dict(data)

        self.assertEqual(config.model_name, "large")
        self.assertEqual(config.language, "fr")
        self.assertEqual(config.temperature, 0.5)
        self.assertTrue(config.debug)

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = WhisperConfig(model_name="medium", language="de", temperature=0.2)

        data = config.to_dict()

        self.assertEqual(data["model_name"], "medium")
        self.assertEqual(data["language"], "de")
        self.assertEqual(data["temperature"], 0.2)


class TestCrossPlatformBehavior(unittest.TestCase):
    """Test cross-platform specific behavior."""

    def test_clipboard_service_creation(self):
        """Test that clipboard service can be created."""
        service = PlatformClipboardService()
        self.assertIsNotNone(service)

    def test_config_directory_creation(self):
        """Test that config directories can be created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "test_config"
            config_repo = FileConfigRepository(config_dir)

            # Should create directory when saving
            config = WhisperConfig()
            config_repo.save(config)

            self.assertTrue(config_dir.exists())
            self.assertTrue(config_dir.is_dir())


if __name__ == "__main__":
    # Set up test environment
    unittest.main(verbosity=2, buffer=True)
