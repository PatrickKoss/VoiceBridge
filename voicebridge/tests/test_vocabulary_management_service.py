"""Tests for vocabulary management service."""

from unittest.mock import Mock

import pytest

from voicebridge.adapters.vocabulary import VocabularyAdapter
from voicebridge.domain.models import VocabularyConfig
from voicebridge.services.vocabulary_management_service import (
    VocabularyManagementService,
)


class TestVocabularyManagementService:
    """Test cases for VocabularyManagementService."""

    @pytest.fixture
    def mock_vocabulary_adapter(self):
        """Mock VocabularyAdapter."""
        return Mock(spec=VocabularyAdapter)

    @pytest.fixture
    def mock_logger(self):
        """Mock Logger."""
        return Mock()

    @pytest.fixture
    def service(self, mock_vocabulary_adapter, mock_logger):
        """VocabularyManagementService instance."""
        return VocabularyManagementService(mock_vocabulary_adapter, mock_logger)

    @pytest.fixture
    def sample_config(self):
        """Sample vocabulary configuration."""
        return VocabularyConfig(
            custom_words=["hello", "world"],
            proper_nouns=["Alice", "Bob"],
            technical_jargon=["API", "SDK"],
            domain_terms={"medical": ["diagnosis", "treatment"]},
            boost_factor=1.5,
            enable_fuzzy_matching=True,
            phonetic_mappings={"colour": "color"},
        )

    def test_init(self, mock_vocabulary_adapter, mock_logger):
        """Test service initialization."""
        service = VocabularyManagementService(mock_vocabulary_adapter, mock_logger)
        assert service.vocabulary_adapter == mock_vocabulary_adapter
        assert service.logger == mock_logger

    def test_add_words_custom(self, service, mock_vocabulary_adapter, sample_config):
        """Test adding words to custom vocabulary."""
        mock_vocabulary_adapter.load_vocabulary_config.return_value = sample_config

        result = service.add_words(["new", "words"], "custom", "test_profile", 1.0)

        assert result is True
        mock_vocabulary_adapter.load_vocabulary_config.assert_called_once_with(
            "test_profile"
        )
        mock_vocabulary_adapter.save_vocabulary_config.assert_called_once()

        # Check that new words were added
        saved_config = mock_vocabulary_adapter.save_vocabulary_config.call_args[0][0]
        assert "new" in saved_config.custom_words
        assert "words" in saved_config.custom_words

    def test_add_words_proper_nouns(
        self, service, mock_vocabulary_adapter, sample_config
    ):
        """Test adding words to proper nouns vocabulary."""
        mock_vocabulary_adapter.load_vocabulary_config.return_value = sample_config

        result = service.add_words(["Charlie", "David"], "proper_nouns", "default")

        assert result is True
        saved_config = mock_vocabulary_adapter.save_vocabulary_config.call_args[0][0]
        assert "Charlie" in saved_config.proper_nouns
        assert "David" in saved_config.proper_nouns

    def test_add_words_technical(self, service, mock_vocabulary_adapter, sample_config):
        """Test adding words to technical vocabulary."""
        mock_vocabulary_adapter.load_vocabulary_config.return_value = sample_config

        result = service.add_words(["REST", "GraphQL"], "technical", "default")

        assert result is True
        saved_config = mock_vocabulary_adapter.save_vocabulary_config.call_args[0][0]
        assert "REST" in saved_config.technical_jargon
        assert "GraphQL" in saved_config.technical_jargon

    def test_add_words_domain_terms_existing(
        self, service, mock_vocabulary_adapter, sample_config
    ):
        """Test adding words to existing domain terms."""
        mock_vocabulary_adapter.load_vocabulary_config.return_value = sample_config

        result = service.add_words(["surgery", "procedure"], "medical", "default")

        assert result is True
        saved_config = mock_vocabulary_adapter.save_vocabulary_config.call_args[0][0]
        assert "surgery" in saved_config.domain_terms["medical"]
        assert "procedure" in saved_config.domain_terms["medical"]

    def test_add_words_domain_terms_new(
        self, service, mock_vocabulary_adapter, sample_config
    ):
        """Test adding words to new domain terms."""
        mock_vocabulary_adapter.load_vocabulary_config.return_value = sample_config

        result = service.add_words(["class", "method"], "programming", "default")

        assert result is True
        saved_config = mock_vocabulary_adapter.save_vocabulary_config.call_args[0][0]
        assert "programming" in saved_config.domain_terms
        assert "class" in saved_config.domain_terms["programming"]
        assert "method" in saved_config.domain_terms["programming"]

    def test_add_words_duplicate_prevention(
        self, service, mock_vocabulary_adapter, sample_config
    ):
        """Test that duplicate words are not added."""
        mock_vocabulary_adapter.load_vocabulary_config.return_value = sample_config

        # Try to add words that already exist
        result = service.add_words(["hello", "new_word"], "custom", "default")

        assert result is True
        saved_config = mock_vocabulary_adapter.save_vocabulary_config.call_args[0][0]
        # "hello" should not be duplicated, but "new_word" should be added
        assert saved_config.custom_words.count("hello") == 1
        assert "new_word" in saved_config.custom_words

    def test_add_words_exception(self, service, mock_vocabulary_adapter, mock_logger):
        """Test add_words with exception."""
        mock_vocabulary_adapter.load_vocabulary_config.side_effect = Exception(
            "Load failed"
        )

        result = service.add_words(["test"], "custom", "default")

        assert result is False
        mock_logger.error.assert_called_once()

    def test_remove_words_custom(self, service, mock_vocabulary_adapter, sample_config):
        """Test removing words from custom vocabulary."""
        mock_vocabulary_adapter.load_vocabulary_config.return_value = sample_config

        result = service.remove_words(["hello"], "custom", "default")

        assert result is True
        saved_config = mock_vocabulary_adapter.save_vocabulary_config.call_args[0][0]
        assert "hello" not in saved_config.custom_words
        assert "world" in saved_config.custom_words  # Should remain

    def test_remove_words_proper_nouns(
        self, service, mock_vocabulary_adapter, sample_config
    ):
        """Test removing words from proper nouns vocabulary."""
        mock_vocabulary_adapter.load_vocabulary_config.return_value = sample_config

        result = service.remove_words(["Alice"], "proper_nouns", "default")

        assert result is True
        saved_config = mock_vocabulary_adapter.save_vocabulary_config.call_args[0][0]
        assert "Alice" not in saved_config.proper_nouns
        assert "Bob" in saved_config.proper_nouns  # Should remain

    def test_remove_words_technical(
        self, service, mock_vocabulary_adapter, sample_config
    ):
        """Test removing words from technical vocabulary."""
        mock_vocabulary_adapter.load_vocabulary_config.return_value = sample_config

        result = service.remove_words(["API"], "technical", "default")

        assert result is True
        saved_config = mock_vocabulary_adapter.save_vocabulary_config.call_args[0][0]
        assert "API" not in saved_config.technical_jargon
        assert "SDK" in saved_config.technical_jargon  # Should remain

    def test_remove_words_domain_terms(
        self, service, mock_vocabulary_adapter, sample_config
    ):
        """Test removing words from domain terms."""
        mock_vocabulary_adapter.load_vocabulary_config.return_value = sample_config

        result = service.remove_words(["diagnosis"], "medical", "default")

        assert result is True
        saved_config = mock_vocabulary_adapter.save_vocabulary_config.call_args[0][0]
        assert "diagnosis" not in saved_config.domain_terms["medical"]
        assert "treatment" in saved_config.domain_terms["medical"]  # Should remain

    def test_remove_words_exception(
        self, service, mock_vocabulary_adapter, mock_logger
    ):
        """Test remove_words with exception."""
        mock_vocabulary_adapter.load_vocabulary_config.side_effect = Exception(
            "Load failed"
        )

        result = service.remove_words(["test"], "custom", "default")

        assert result is False
        mock_logger.error.assert_called_once()

    def test_list_vocabularies_all(
        self, service, mock_vocabulary_adapter, sample_config
    ):
        """Test listing all vocabularies."""
        mock_vocabulary_adapter.load_vocabulary_config.return_value = sample_config

        result = service.list_vocabularies(None, "default")

        expected = {
            "custom": ["hello", "world"],
            "proper_nouns": ["Alice", "Bob"],
            "technical": ["API", "SDK"],
            "medical": ["diagnosis", "treatment"],
        }
        assert result == expected

    def test_list_vocabularies_specific_type(
        self, service, mock_vocabulary_adapter, sample_config
    ):
        """Test listing specific vocabulary type."""
        mock_vocabulary_adapter.load_vocabulary_config.return_value = sample_config

        result = service.list_vocabularies("custom", "default")

        expected = {"custom": ["hello", "world"]}
        assert result == expected

    def test_list_vocabularies_nonexistent_type(
        self, service, mock_vocabulary_adapter, sample_config
    ):
        """Test listing nonexistent vocabulary type."""
        mock_vocabulary_adapter.load_vocabulary_config.return_value = sample_config

        result = service.list_vocabularies("nonexistent", "default")

        expected = {"nonexistent": []}
        assert result == expected

    def test_list_vocabularies_exception(
        self, service, mock_vocabulary_adapter, mock_logger
    ):
        """Test list_vocabularies with exception."""
        mock_vocabulary_adapter.load_vocabulary_config.side_effect = Exception(
            "Load failed"
        )

        result = service.list_vocabularies()

        assert result == {}
        mock_logger.error.assert_called_once()

    def test_import_vocabulary_success(
        self, service, mock_vocabulary_adapter, sample_config
    ):
        """Test successful vocabulary import."""
        mock_vocabulary_adapter.load_vocabulary_config.return_value = sample_config
        mock_vocabulary_adapter.import_vocabulary_from_file.return_value = [
            "imported1",
            "imported2",
        ]

        result = service.import_vocabulary(
            "/path/to/file.txt", "custom", "default", "txt"
        )

        assert result is True
        mock_vocabulary_adapter.import_vocabulary_from_file.assert_called_once_with(
            "/path/to/file.txt", "custom"
        )

    def test_import_vocabulary_no_words(
        self, service, mock_vocabulary_adapter, mock_logger
    ):
        """Test vocabulary import with no words found."""
        mock_vocabulary_adapter.import_vocabulary_from_file.return_value = []

        result = service.import_vocabulary("/path/to/file.txt", "custom", "default")

        assert result is False
        mock_logger.warning.assert_called_once()

    def test_import_vocabulary_exception(
        self, service, mock_vocabulary_adapter, mock_logger
    ):
        """Test vocabulary import with exception."""
        mock_vocabulary_adapter.import_vocabulary_from_file.side_effect = Exception(
            "Import failed"
        )

        result = service.import_vocabulary("/path/to/file.txt", "custom", "default")

        assert result is False
        mock_logger.error.assert_called_once()

    def test_export_vocabulary_success(
        self, service, mock_vocabulary_adapter, sample_config
    ):
        """Test successful vocabulary export."""
        mock_vocabulary_adapter.load_vocabulary_config.return_value = sample_config

        result = service.export_vocabulary("/path/to/output.txt", "default")

        assert result is True
        mock_vocabulary_adapter.load_vocabulary_config.assert_called_once_with(
            "default"
        )
        mock_vocabulary_adapter.export_vocabulary_to_file.assert_called_once_with(
            sample_config, "/path/to/output.txt"
        )

    def test_export_vocabulary_exception(
        self, service, mock_vocabulary_adapter, mock_logger
    ):
        """Test vocabulary export with exception."""
        mock_vocabulary_adapter.load_vocabulary_config.side_effect = Exception(
            "Load failed"
        )

        result = service.export_vocabulary("/path/to/output.txt", "default")

        assert result is False
        mock_logger.error.assert_called_once()

    def test_list_profiles(self, service, mock_vocabulary_adapter):
        """Test listing vocabulary profiles."""
        mock_vocabulary_adapter.list_vocabulary_profiles.return_value = [
            "default",
            "medical",
            "tech",
        ]

        result = service.list_profiles()

        assert result == ["default", "medical", "tech"]
        mock_vocabulary_adapter.list_vocabulary_profiles.assert_called_once()

    def test_delete_profile_success(
        self, service, mock_vocabulary_adapter, mock_logger
    ):
        """Test successful profile deletion."""
        mock_vocabulary_adapter.delete_vocabulary_profile.return_value = True

        result = service.delete_profile("test_profile")

        assert result is True
        mock_vocabulary_adapter.delete_vocabulary_profile.assert_called_once_with(
            "test_profile"
        )
        mock_logger.info.assert_called_once()

    def test_delete_profile_failure(
        self, service, mock_vocabulary_adapter, mock_logger
    ):
        """Test failed profile deletion."""
        mock_vocabulary_adapter.delete_vocabulary_profile.return_value = False

        result = service.delete_profile("test_profile")

        assert result is False
        mock_logger.info.assert_not_called()

    def test_delete_profile_exception(
        self, service, mock_vocabulary_adapter, mock_logger
    ):
        """Test profile deletion with exception."""
        mock_vocabulary_adapter.delete_vocabulary_profile.side_effect = Exception(
            "Delete failed"
        )

        result = service.delete_profile("test_profile")

        assert result is False
        mock_logger.error.assert_called_once()

    def test_get_config(self, service, mock_vocabulary_adapter, sample_config):
        """Test getting vocabulary configuration."""
        mock_vocabulary_adapter.load_vocabulary_config.return_value = sample_config

        result = service.get_config("test_profile")

        assert result == sample_config
        mock_vocabulary_adapter.load_vocabulary_config.assert_called_once_with(
            "test_profile"
        )

    def test_update_config_boost_factor(
        self, service, mock_vocabulary_adapter, sample_config
    ):
        """Test updating configuration boost factor."""
        mock_vocabulary_adapter.load_vocabulary_config.return_value = sample_config

        result = service.update_config("default", boost_factor=2.0)

        assert result is True
        saved_config = mock_vocabulary_adapter.save_vocabulary_config.call_args[0][0]
        assert saved_config.boost_factor == 2.0

    def test_update_config_fuzzy_matching(
        self, service, mock_vocabulary_adapter, sample_config
    ):
        """Test updating configuration fuzzy matching."""
        mock_vocabulary_adapter.load_vocabulary_config.return_value = sample_config

        result = service.update_config("default", enable_fuzzy_matching=False)

        assert result is True
        saved_config = mock_vocabulary_adapter.save_vocabulary_config.call_args[0][0]
        assert saved_config.enable_fuzzy_matching is False

    def test_update_config_phonetic_mappings(
        self, service, mock_vocabulary_adapter, sample_config
    ):
        """Test updating configuration phonetic mappings."""
        mock_vocabulary_adapter.load_vocabulary_config.return_value = sample_config
        new_mappings = {"favor": "favour", "center": "centre"}

        result = service.update_config("default", phonetic_mappings=new_mappings)

        assert result is True
        saved_config = mock_vocabulary_adapter.save_vocabulary_config.call_args[0][0]
        # Should merge with existing mappings
        assert saved_config.phonetic_mappings["colour"] == "color"  # Existing
        assert saved_config.phonetic_mappings["favor"] == "favour"  # New
        assert saved_config.phonetic_mappings["center"] == "centre"  # New

    def test_update_config_all_parameters(
        self, service, mock_vocabulary_adapter, sample_config
    ):
        """Test updating all configuration parameters."""
        mock_vocabulary_adapter.load_vocabulary_config.return_value = sample_config
        new_mappings = {"gray": "grey"}

        result = service.update_config(
            "default",
            boost_factor=3.0,
            enable_fuzzy_matching=False,
            phonetic_mappings=new_mappings,
        )

        assert result is True
        saved_config = mock_vocabulary_adapter.save_vocabulary_config.call_args[0][0]
        assert saved_config.boost_factor == 3.0
        assert saved_config.enable_fuzzy_matching is False
        assert saved_config.phonetic_mappings["gray"] == "grey"

    def test_update_config_exception(
        self, service, mock_vocabulary_adapter, mock_logger
    ):
        """Test updating configuration with exception."""
        mock_vocabulary_adapter.load_vocabulary_config.side_effect = Exception(
            "Load failed"
        )

        result = service.update_config("default", boost_factor=2.0)

        assert result is False
        mock_logger.error.assert_called_once()

    def test_add_words_logging(
        self, service, mock_vocabulary_adapter, sample_config, mock_logger
    ):
        """Test logging in add_words method."""
        mock_vocabulary_adapter.load_vocabulary_config.return_value = sample_config

        service.add_words(["test1", "test2"], "custom", "test_profile")

        mock_logger.info.assert_called_once_with(
            "Added 2 words to custom vocabulary for profile test_profile"
        )

    def test_remove_words_logging(
        self, service, mock_vocabulary_adapter, sample_config, mock_logger
    ):
        """Test logging in remove_words method."""
        mock_vocabulary_adapter.load_vocabulary_config.return_value = sample_config

        service.remove_words(["hello"], "custom", "test_profile")

        mock_logger.info.assert_called_once_with(
            "Removed 1 words from custom vocabulary for profile test_profile"
        )
