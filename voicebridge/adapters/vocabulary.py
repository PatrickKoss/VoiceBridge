import json
from pathlib import Path

from voicebridge.domain.models import VocabularyConfig


class VocabularyAdapter:
    def __init__(self, vocabulary_dir: str = "vocabulary"):
        self.vocabulary_dir = Path(vocabulary_dir)
        self.vocabulary_dir.mkdir(exist_ok=True)

    def load_vocabulary_config(self, profile_name: str = "default") -> VocabularyConfig:
        config_file = self.vocabulary_dir / f"{profile_name}.json"

        if not config_file.exists():
            return VocabularyConfig()

        try:
            with open(config_file, encoding="utf-8") as f:
                data = json.load(f)

            return VocabularyConfig(
                custom_words=data.get("custom_words", []),
                domain_terms=data.get("domain_terms", {}),
                proper_nouns=data.get("proper_nouns", []),
                technical_jargon=data.get("technical_jargon", []),
                phonetic_mappings=data.get("phonetic_mappings", {}),
                boost_factor=data.get("boost_factor", 2.0),
                enable_fuzzy_matching=data.get("enable_fuzzy_matching", True),
            )
        except (json.JSONDecodeError, FileNotFoundError):
            return VocabularyConfig()

    def save_vocabulary_config(
        self, config: VocabularyConfig, profile_name: str = "default"
    ) -> None:
        config_file = self.vocabulary_dir / f"{profile_name}.json"

        data = {
            "custom_words": config.custom_words,
            "domain_terms": config.domain_terms,
            "proper_nouns": config.proper_nouns,
            "technical_jargon": config.technical_jargon,
            "phonetic_mappings": config.phonetic_mappings,
            "boost_factor": config.boost_factor,
            "enable_fuzzy_matching": config.enable_fuzzy_matching,
        }

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_domain_vocabulary(self, domain: str) -> list[str]:
        domain_file = self.vocabulary_dir / "domains" / f"{domain}.txt"

        if not domain_file.exists():
            return []

        try:
            with open(domain_file, encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            return []

    def save_domain_vocabulary(self, domain: str, terms: list[str]) -> None:
        domain_dir = self.vocabulary_dir / "domains"
        domain_dir.mkdir(exist_ok=True)

        domain_file = domain_dir / f"{domain}.txt"

        with open(domain_file, "w", encoding="utf-8") as f:
            for term in sorted(set(terms)):  # Remove duplicates and sort
                f.write(f"{term}\n")

    def import_vocabulary_from_file(
        self, file_path: str, vocabulary_type: str = "custom"
    ) -> list[str]:
        try:
            with open(file_path, encoding="utf-8") as f:
                if file_path.endswith(".json"):
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict) and vocabulary_type in data:
                        return data[vocabulary_type]
                    else:
                        return list(data.values())[0] if data else []
                else:
                    # Treat as plain text file
                    return [line.strip() for line in f if line.strip()]
        except (json.JSONDecodeError, FileNotFoundError, UnicodeDecodeError):
            return []

    def export_vocabulary_to_file(
        self, config: VocabularyConfig, file_path: str
    ) -> None:
        data = {
            "custom_words": config.custom_words,
            "domain_terms": config.domain_terms,
            "proper_nouns": config.proper_nouns,
            "technical_jargon": config.technical_jargon,
            "phonetic_mappings": config.phonetic_mappings,
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def list_vocabulary_profiles(self) -> list[str]:
        profiles = []
        for file_path in self.vocabulary_dir.glob("*.json"):
            profiles.append(file_path.stem)
        return sorted(profiles)

    def delete_vocabulary_profile(self, profile_name: str) -> bool:
        config_file = self.vocabulary_dir / f"{profile_name}.json"
        try:
            if config_file.exists():
                config_file.unlink()
                return True
        except OSError:
            pass
        return False
