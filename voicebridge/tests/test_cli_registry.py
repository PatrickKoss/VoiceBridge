"""Tests for CLI command registry."""

from unittest.mock import Mock, patch

import pytest

from voicebridge.cli.registry import CommandRegistry, create_command_registry


class TestCommandRegistry:
    """Test cases for CommandRegistry."""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock service dependencies."""
        return {
            "config_repo": Mock(),
            "profile_repo": Mock(),
            "logger": Mock(),
        }

    @pytest.fixture
    def registry(self, mock_dependencies):
        """CommandRegistry instance."""
        return CommandRegistry(**mock_dependencies)

    def test_init(self, mock_dependencies):
        """Test registry initialization."""
        registry = CommandRegistry(**mock_dependencies)
        assert isinstance(registry, CommandRegistry)
        assert registry.dependencies == mock_dependencies
        assert registry._command_instances == {}

    def test_command_groups_defined(self, registry):
        """Test that command groups are properly defined."""
        expected_groups = [
            "speech",
            "transcription",
            "tts",
            "audio",
            "system",
            "config",
            "export",
            "advanced",
            "api",
        ]

        assert set(registry.COMMAND_GROUPS.keys()) == set(expected_groups)

        # All groups should map to classes
        for _group_name, command_class in registry.COMMAND_GROUPS.items():
            assert isinstance(command_class, type)

    def test_get_command_group_valid(self, registry):
        """Test getting a valid command group."""
        with patch.object(registry, "COMMAND_GROUPS", {"test_group": Mock}):
            mock_class = Mock()
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            registry.COMMAND_GROUPS["test_group"] = mock_class

            result = registry.get_command_group("test_group")

            assert result == mock_instance
            mock_class.assert_called_once_with(**registry.dependencies)

    def test_get_command_group_invalid(self, registry):
        """Test getting an invalid command group."""
        with pytest.raises(ValueError, match="Unknown command group 'invalid'"):
            registry.get_command_group("invalid")

    def test_get_command_group_caching(self, registry):
        """Test that command group instances are cached."""
        with patch.object(registry, "COMMAND_GROUPS", {"test_group": Mock}):
            mock_class = Mock()
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            registry.COMMAND_GROUPS["test_group"] = mock_class

            # Get twice
            result1 = registry.get_command_group("test_group")
            result2 = registry.get_command_group("test_group")

            # Should be same instance
            assert result1 == result2 == mock_instance
            # Class should only be called once (caching)
            mock_class.assert_called_once()

    def test_get_all_command_groups(self, registry):
        """Test getting all command groups."""
        # Mock a smaller set for testing
        mock_groups = {
            "group1": Mock(),
            "group2": Mock(),
        }

        with patch.object(registry, "COMMAND_GROUPS", mock_groups):
            with patch.object(registry, "get_command_group") as mock_get:
                mock_get.side_effect = lambda name: f"instance_{name}"

                result = registry.get_all_command_groups()

                assert result == {
                    "group1": "instance_group1",
                    "group2": "instance_group2",
                }
                assert mock_get.call_count == 2

    def test_list_command_groups(self, registry):
        """Test listing command group names."""
        result = registry.list_command_groups()

        expected = [
            "speech",
            "transcription",
            "tts",
            "audio",
            "system",
            "config",
            "export",
            "advanced",
            "api",
        ]
        assert set(result) == set(expected)

    def test_validate_dependencies_all_required_present(self, registry):
        """Test dependency validation with all required dependencies."""
        registry.dependencies = {
            "config_repo": Mock(),
            "profile_repo": Mock(),
            "logger": Mock(),
        }

        result = registry.validate_dependencies()

        assert result["config_repo"] is True
        assert result["profile_repo"] is True
        assert result["logger"] is True

    def test_validate_dependencies_missing_required(self, registry):
        """Test dependency validation with missing required dependencies."""
        registry.dependencies = {
            "config_repo": Mock(),
            # Missing profile_repo and logger
        }

        result = registry.validate_dependencies()

        assert result["config_repo"] is True
        assert result["profile_repo"] is False
        assert result["logger"] is False

    def test_validate_dependencies_optional_services(self, registry):
        """Test dependency validation with optional services."""
        registry.dependencies = {
            "config_repo": Mock(),
            "profile_repo": Mock(),
            "logger": Mock(),
            "transcription_orchestrator": Mock(),
            "tts_orchestrator": None,  # Present but None
        }

        result = registry.validate_dependencies()

        assert result["transcription_orchestrator"] is True
        assert result["tts_orchestrator"] is False  # None counts as False

    def test_get_dependency_summary(self, registry):
        """Test getting dependency summary."""
        registry.dependencies = {
            "config_repo": Mock(),
            "profile_repo": Mock(),
            "logger": Mock(),
            "transcription_orchestrator": Mock(),
        }

        result = registry.get_dependency_summary()

        # Should have entries for all command groups
        expected_groups = [
            "speech",
            "transcription",
            "tts",
            "audio",
            "system",
            "config",
            "export",
            "advanced",
            "api",
        ]
        assert set(result.keys()) == set(expected_groups)

        # Each entry should have required structure
        for _group_name, stats in result.items():
            assert "available" in stats
            assert "total" in stats
            assert "percentage" in stats
            assert isinstance(stats["available"], int)
            assert isinstance(stats["total"], int)
            assert isinstance(stats["percentage"], (int, float))

    def test_get_dependency_summary_config_group(self, registry):
        """Test dependency summary for config group specifically."""
        registry.dependencies = {
            "config_repo": Mock(),
            "profile_repo": Mock(),
            "logger": Mock(),
        }

        result = registry.get_dependency_summary()

        config_stats = result["config"]
        assert config_stats["available"] == 2  # config_repo and profile_repo
        assert config_stats["total"] == 2
        assert config_stats["percentage"] == 100.0


class TestCreateCommandRegistry:
    """Test cases for create_command_registry factory function."""

    def test_create_command_registry_basic(self):
        """Test basic registry creation."""
        deps = {"config_repo": Mock(), "logger": Mock()}

        registry = create_command_registry(**deps)

        assert isinstance(registry, CommandRegistry)
        assert registry.dependencies == deps

    def test_create_command_registry_with_services(self):
        """Test registry creation with all services."""
        deps = {
            "config_repo": Mock(),
            "profile_repo": Mock(),
            "logger": Mock(),
            "transcription_orchestrator": Mock(),
            "tts_orchestrator": Mock(),
            "system_service": Mock(),
        }

        registry = create_command_registry(**deps)

        assert isinstance(registry, CommandRegistry)
        assert registry.dependencies == deps

    def test_create_command_registry_empty(self):
        """Test registry creation with no dependencies."""
        registry = create_command_registry()

        assert isinstance(registry, CommandRegistry)
        assert registry.dependencies == {}

    def test_factory_function_returns_same_type(self):
        """Test that factory function returns same type as direct instantiation."""
        deps = {"test": Mock()}

        direct = CommandRegistry(**deps)
        factory = create_command_registry(**deps)

        assert type(direct) is type(factory)
        assert direct.dependencies == factory.dependencies
