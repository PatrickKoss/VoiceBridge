#!/usr/bin/env python3
"""Unit tests for configuration adapters."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from voicebridge.adapters.config import FileConfigRepository, FileProfileRepository
from voicebridge.domain.models import OperationMode, WhisperConfig


class TestFileConfigRepository(unittest.TestCase):
    """Test FileConfigRepository adapter."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_repo = FileConfigRepository(self.temp_dir)

    def test_load_default_config(self):
        """Test loading default config when no file exists."""
        config = self.config_repo.load()

        self.assertEqual(config.model_name, "medium")
        self.assertIsNone(config.language)
        self.assertEqual(config.temperature, 0.0)
        self.assertFalse(config.debug)

    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        # Create custom config
        original_config = WhisperConfig(
            model_name="large",
            language="en",
            temperature=0.1,
            debug=True,
            paste_stream=True,
        )

        # Save config
        self.config_repo.save(original_config)

        # Load config
        loaded_config = self.config_repo.load()

        # Verify
        self.assertEqual(loaded_config.model_name, "large")
        self.assertEqual(loaded_config.language, "en")
        self.assertEqual(loaded_config.temperature, 0.1)
        self.assertTrue(loaded_config.debug)
        self.assertTrue(loaded_config.paste_stream)

    @patch("builtins.open", side_effect=OSError("Permission denied"))
    def test_save_config_error(self, mock_file):
        """Test error handling when saving config fails."""
        config = WhisperConfig()

        with self.assertRaises(RuntimeError) as context:
            self.config_repo.save(config)

        self.assertIn("Could not save config", str(context.exception))

    def test_load_corrupted_config(self):
        """Test loading corrupted config file returns defaults."""
        # Create corrupted config file
        config_file = self.temp_dir / "config.json"
        config_file.write_text("{ invalid json }")

        # Should return defaults
        config = self.config_repo.load()
        self.assertEqual(config.model_name, "medium")  # Default value

    def test_load_partial_config(self):
        """Test loading partial config merges with defaults."""
        # Create partial config
        partial_data = {"model_name": "small", "debug": True}
        config_file = self.temp_dir / "config.json"
        config_file.write_text(json.dumps(partial_data))

        config = self.config_repo.load()

        # Should have partial values
        self.assertEqual(config.model_name, "small")
        self.assertTrue(config.debug)

        # Should have default values for missing fields
        self.assertEqual(config.temperature, 0.0)
        self.assertEqual(config.key, "ctrl+f2")


class TestFileProfileRepository(unittest.TestCase):
    """Test FileProfileRepository adapter."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.profiles_dir = self.temp_dir / "profiles"
        self.profile_repo = FileProfileRepository(self.profiles_dir)

    def test_save_and_load_profile(self):
        """Test saving and loading a profile."""
        # Create test config
        config = WhisperConfig(
            model_name="tiny",
            language="fr",
            temperature=0.5,
            mode=OperationMode.PUSH_TO_TALK,
        )

        # Save profile
        self.profile_repo.save_profile("test_profile", config)

        # Load profile
        loaded_config = self.profile_repo.load_profile("test_profile")

        # Verify
        self.assertEqual(loaded_config.model_name, "tiny")
        self.assertEqual(loaded_config.language, "fr")
        self.assertEqual(loaded_config.temperature, 0.5)
        self.assertEqual(loaded_config.mode, OperationMode.PUSH_TO_TALK)

    def test_load_nonexistent_profile(self):
        """Test loading non-existent profile raises error."""
        with self.assertRaises(FileNotFoundError) as context:
            self.profile_repo.load_profile("nonexistent")

        self.assertIn("Profile 'nonexistent' not found", str(context.exception))

    def test_list_profiles_empty(self):
        """Test listing profiles when directory is empty."""
        profiles = self.profile_repo.list_profiles()
        self.assertEqual(profiles, [])

    def test_list_profiles_with_data(self):
        """Test listing profiles with existing data."""
        # Save some profiles
        config1 = WhisperConfig(model_name="small")
        config2 = WhisperConfig(model_name="medium")

        self.profile_repo.save_profile("profile1", config1)
        self.profile_repo.save_profile("profile2", config2)

        # List profiles
        profiles = self.profile_repo.list_profiles()

        self.assertEqual(set(profiles), {"profile1", "profile2"})

    def test_delete_existing_profile(self):
        """Test deleting existing profile."""
        # Save profile
        config = WhisperConfig(model_name="large")
        self.profile_repo.save_profile("to_delete", config)

        # Verify it exists
        self.assertIn("to_delete", self.profile_repo.list_profiles())

        # Delete it
        result = self.profile_repo.delete_profile("to_delete")

        # Verify deletion
        self.assertTrue(result)
        self.assertNotIn("to_delete", self.profile_repo.list_profiles())

    def test_delete_nonexistent_profile(self):
        """Test deleting non-existent profile."""
        result = self.profile_repo.delete_profile("nonexistent")
        self.assertFalse(result)

    @patch("builtins.open", side_effect=OSError("Permission denied"))
    def test_save_profile_error(self, mock_file):
        """Test error handling when saving profile fails."""
        config = WhisperConfig()

        with self.assertRaises(RuntimeError) as context:
            self.profile_repo.save_profile("test", config)

        self.assertIn("Could not save profile 'test'", str(context.exception))

    def test_load_corrupted_profile(self):
        """Test loading corrupted profile file."""
        # Create corrupted profile file
        profile_file = self.profiles_dir / "corrupted.json"
        profile_file.parent.mkdir(parents=True, exist_ok=True)
        profile_file.write_text("{ invalid json }")

        with self.assertRaises(RuntimeError) as context:
            self.profile_repo.load_profile("corrupted")

        self.assertIn("Could not load profile 'corrupted'", str(context.exception))

    def test_profile_file_path_generation(self):
        """Test profile file path generation."""
        # This tests the internal _profile_path method
        path = self.profile_repo._profile_path("test_profile")
        expected = self.profiles_dir / "test_profile.json"
        self.assertEqual(path, expected)


if __name__ == "__main__":
    unittest.main()
