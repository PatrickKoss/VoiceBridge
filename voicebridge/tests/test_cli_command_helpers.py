"""Tests for CLI command helpers."""

from voicebridge.cli.utils.command_helpers import build_whisper_config
from voicebridge.domain.models import WhisperConfig


class TestBuildWhisperConfig:
    """Tests for build_whisper_config function."""

    def test_build_whisper_config_with_model(self):
        """Test that model parameter maps to model_name in WhisperConfig."""
        config = build_whisper_config(model="small")

        assert config == {"model_name": "small"}

    def test_build_whisper_config_with_all_params(self):
        """Test build_whisper_config with all parameters."""
        config = build_whisper_config(
            model="base",
            language="en",
            initial_prompt="Hello",
            temperature=0.5,
            use_gpu=True,
        )

        expected = {
            "model_name": "base",
            "language": "en",
            "initial_prompt": "Hello",
            "temperature": 0.5,
            "use_gpu": True,
        }
        assert config == expected

    def test_build_whisper_config_with_none_values(self):
        """Test that None values are not included in config."""
        config = build_whisper_config(
            model=None,
            language=None,
            initial_prompt=None,
            temperature=0.0,
            use_gpu=None,
        )

        # Only temperature should be included since it's not 0.0
        assert config == {}

    def test_build_whisper_config_with_kwargs(self):
        """Test that additional kwargs are included."""
        config = build_whisper_config(
            model="tiny", custom_param="value", another_param=123
        )

        expected = {"model_name": "tiny", "custom_param": "value", "another_param": 123}
        assert config == expected

    def test_model_parameter_maps_to_model_name(self):
        """Specific test to ensure model parameter maps correctly to WhisperConfig.model_name."""
        # This test ensures the fix for the original issue
        config = build_whisper_config(model="large")

        # Verify the config can be used to create a WhisperConfig instance
        whisper_config = WhisperConfig(**config)
        assert whisper_config.model_name == "large"

    def test_temperature_zero_not_included(self):
        """Test that temperature=0.0 is not included in config."""
        config = build_whisper_config(temperature=0.0)
        assert "temperature" not in config

    def test_temperature_non_zero_included(self):
        """Test that non-zero temperature is included in config."""
        config = build_whisper_config(temperature=0.5)
        assert config["temperature"] == 0.5
