"""
Vocabulary management service for CLI operations.
"""

from voicebridge.adapters.vocabulary import VocabularyAdapter
from voicebridge.domain.models import VocabularyConfig
from voicebridge.ports.interfaces import Logger
from voicebridge.ports.interfaces import (
    VocabularyManagementService as VocabularyManagementServiceInterface,
)


class VocabularyManagementService(VocabularyManagementServiceInterface):
    """Service for managing vocabulary from CLI commands."""

    def __init__(self, vocabulary_adapter: VocabularyAdapter, logger: Logger):
        self.vocabulary_adapter = vocabulary_adapter
        self.logger = logger

    def add_words(
        self,
        words: list[str],
        vocabulary_type: str = "custom",
        profile: str = "default",
        weight: float = 1.0,
    ) -> bool:
        """Add words to vocabulary."""
        try:
            # Load existing configuration
            config = self.vocabulary_adapter.load_vocabulary_config(profile)

            # Add words to appropriate vocabulary type
            if vocabulary_type == "custom":
                for word in words:
                    if word not in config.custom_words:
                        config.custom_words.append(word)
            elif vocabulary_type == "proper_nouns":
                for word in words:
                    if word not in config.proper_nouns:
                        config.proper_nouns.append(word)
            elif vocabulary_type == "technical":
                for word in words:
                    if word not in config.technical_jargon:
                        config.technical_jargon.append(word)
            else:
                # Domain terms
                if vocabulary_type not in config.domain_terms:
                    config.domain_terms[vocabulary_type] = []
                for word in words:
                    if word not in config.domain_terms[vocabulary_type]:
                        config.domain_terms[vocabulary_type].append(word)

            # Save updated configuration
            self.vocabulary_adapter.save_vocabulary_config(config, profile)

            self.logger.info(
                f"Added {len(words)} words to {vocabulary_type} vocabulary for profile {profile}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to add vocabulary words: {e}")
            return False

    def remove_words(
        self,
        words: list[str],
        vocabulary_type: str = "custom",
        profile: str = "default",
    ) -> bool:
        """Remove words from vocabulary."""
        try:
            # Load existing configuration
            config = self.vocabulary_adapter.load_vocabulary_config(profile)

            # Remove words from appropriate vocabulary type
            if vocabulary_type == "custom":
                config.custom_words = [w for w in config.custom_words if w not in words]
            elif vocabulary_type == "proper_nouns":
                config.proper_nouns = [w for w in config.proper_nouns if w not in words]
            elif vocabulary_type == "technical":
                config.technical_jargon = [
                    w for w in config.technical_jargon if w not in words
                ]
            else:
                # Domain terms
                if vocabulary_type in config.domain_terms:
                    config.domain_terms[vocabulary_type] = [
                        w
                        for w in config.domain_terms[vocabulary_type]
                        if w not in words
                    ]

            # Save updated configuration
            self.vocabulary_adapter.save_vocabulary_config(config, profile)

            self.logger.info(
                f"Removed {len(words)} words from {vocabulary_type} vocabulary for profile {profile}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to remove vocabulary words: {e}")
            return False

    def list_vocabularies(
        self, vocabulary_type: str | None = None, profile: str = "default"
    ) -> dict[str, list[str]]:
        """List vocabulary words."""
        try:
            config = self.vocabulary_adapter.load_vocabulary_config(profile)

            vocabularies = {
                "custom": config.custom_words,
                "proper_nouns": config.proper_nouns,
                "technical": config.technical_jargon,
            }

            # Add domain terms
            for domain, terms in config.domain_terms.items():
                vocabularies[domain] = terms

            # Filter by vocabulary type if specified
            if vocabulary_type:
                return {vocabulary_type: vocabularies.get(vocabulary_type, [])}

            return vocabularies

        except Exception as e:
            self.logger.error(f"Failed to list vocabularies: {e}")
            return {}

    def import_vocabulary(
        self,
        file_path: str,
        vocabulary_type: str = "custom",
        profile: str = "default",
        format: str = "txt",
    ) -> bool:
        """Import vocabulary from file."""
        try:
            # Import words from file
            words = self.vocabulary_adapter.import_vocabulary_from_file(
                file_path, vocabulary_type
            )

            if not words:
                self.logger.warning(f"No words found in file: {file_path}")
                return False

            # Add words using existing add_words method
            return self.add_words(words, vocabulary_type, profile)

        except Exception as e:
            self.logger.error(f"Failed to import vocabulary: {e}")
            return False

    def export_vocabulary(self, file_path: str, profile: str = "default") -> bool:
        """Export vocabulary to file."""
        try:
            config = self.vocabulary_adapter.load_vocabulary_config(profile)
            self.vocabulary_adapter.export_vocabulary_to_file(config, file_path)

            self.logger.info(f"Exported vocabulary to: {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export vocabulary: {e}")
            return False

    def list_profiles(self) -> list[str]:
        """List available vocabulary profiles."""
        return self.vocabulary_adapter.list_vocabulary_profiles()

    def delete_profile(self, profile: str) -> bool:
        """Delete a vocabulary profile."""
        try:
            success = self.vocabulary_adapter.delete_vocabulary_profile(profile)
            if success:
                self.logger.info(f"Deleted vocabulary profile: {profile}")
            return success
        except Exception as e:
            self.logger.error(f"Failed to delete profile: {e}")
            return False

    def get_config(self, profile: str = "default") -> VocabularyConfig:
        """Get vocabulary configuration for a profile."""
        return self.vocabulary_adapter.load_vocabulary_config(profile)

    def update_config(
        self,
        profile: str = "default",
        boost_factor: float | None = None,
        enable_fuzzy_matching: bool | None = None,
        phonetic_mappings: dict[str, str] | None = None,
    ) -> bool:
        """Update vocabulary configuration settings."""
        try:
            config = self.vocabulary_adapter.load_vocabulary_config(profile)

            if boost_factor is not None:
                config.boost_factor = boost_factor

            if enable_fuzzy_matching is not None:
                config.enable_fuzzy_matching = enable_fuzzy_matching

            if phonetic_mappings is not None:
                config.phonetic_mappings.update(phonetic_mappings)

            self.vocabulary_adapter.save_vocabulary_config(config, profile)
            self.logger.info(f"Updated vocabulary configuration for profile: {profile}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to update vocabulary configuration: {e}")
            return False
