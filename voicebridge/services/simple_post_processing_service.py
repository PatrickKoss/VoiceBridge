"""
Simple post-processing service that matches the CLI expectations.
"""

import json
from pathlib import Path
from typing import Any


class SimplePostProcessingService:
    """Simple post-processing service for CLI use."""

    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.config_file = config_dir / "postprocessing.json"
        self._configs = self._load_configs()

    def _load_configs(self) -> dict[str, dict[str, Any]]:
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        # Default configuration
        return {
            "default": {
                "spell_check": False,
                "grammar_check": False,
                "punctuation": True,
                "capitalization": True,
                "custom_rules": [],
            }
        }

    def _save_configs(self):
        """Save configuration to file."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, "w") as f:
            json.dump(self._configs, f, indent=2)

    def get_config(self, profile: str = "default") -> dict[str, Any]:
        """Get configuration for a profile."""
        return self._configs.get(profile, self._configs["default"]).copy()

    def update_config(self, profile: str, config_updates: dict[str, Any]) -> bool:
        """Update configuration for a profile."""
        try:
            if profile not in self._configs:
                self._configs[profile] = self._configs["default"].copy()

            self._configs[profile].update(config_updates)
            self._save_configs()
            return True
        except Exception:
            return False

    def process_text(self, text: str, profile: str = "default") -> dict[str, Any]:
        """Process text with the given profile settings."""
        config = self.get_config(profile)
        processed_text = text
        changes = []

        # Apply punctuation fixes
        if config.get("punctuation", True):
            original = processed_text
            processed_text = self._fix_punctuation(processed_text)
            if processed_text != original:
                changes.append(
                    {
                        "type": "punctuation",
                        "original": original,
                        "corrected": processed_text,
                    }
                )

        # Apply capitalization fixes
        if config.get("capitalization", True):
            original = processed_text
            processed_text = self._fix_capitalization(processed_text)
            if processed_text != original:
                changes.append(
                    {
                        "type": "capitalization",
                        "original": original,
                        "corrected": processed_text,
                    }
                )

        # Apply custom rules
        custom_rules = config.get("custom_rules", [])
        if custom_rules:
            original = processed_text
            processed_text = self._apply_custom_rules(processed_text, custom_rules)
            if processed_text != original:
                changes.append(
                    {
                        "type": "custom_rules",
                        "original": original,
                        "corrected": processed_text,
                    }
                )

        # Note: spell_check and grammar_check would require external libraries
        # For now, just acknowledge they're enabled but don't process
        if config.get("spell_check", False):
            changes.append(
                {
                    "type": "info",
                    "original": "",
                    "corrected": "Spell check enabled (not implemented)",
                }
            )

        if config.get("grammar_check", False):
            changes.append(
                {
                    "type": "info",
                    "original": "",
                    "corrected": "Grammar check enabled (not implemented)",
                }
            )

        return {"text": processed_text, "changes": changes}

    def _fix_punctuation(self, text: str) -> str:
        """Basic punctuation fixes."""
        import re

        # Remove extra spaces before punctuation
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)

        # Add space after punctuation if missing
        text = re.sub(r"([,.!?;:])([a-zA-Z])", r"\1 \2", text)

        # Fix multiple spaces
        text = re.sub(r"\s+", " ", text)

        # Ensure text ends with punctuation
        text = text.strip()
        if text and text[-1] not in ".!?":
            text += "."

        return text

    def _fix_capitalization(self, text: str) -> str:
        """Basic capitalization fixes."""
        import re

        # Capitalize first letter of sentences
        sentences = re.split(r"([.!?]+\s*)", text)
        capitalized_sentences = []

        for i, sentence in enumerate(sentences):
            if i % 2 == 0 and sentence.strip():  # Actual sentence content
                sentence = sentence.strip()
                if sentence:
                    sentence = sentence[0].upper() + sentence[1:]
                    # Capitalize "I"
                    sentence = re.sub(r"\bi\b", "I", sentence)
                capitalized_sentences.append(sentence)
            else:
                capitalized_sentences.append(sentence)

        return "".join(capitalized_sentences)

    def _apply_custom_rules(self, text: str, rules: list[str]) -> str:
        """Apply custom replacement rules."""
        # Rules format: "old->new"
        for rule in rules:
            if "->" in rule:
                old, new = rule.split("->", 1)
                text = text.replace(old.strip(), new.strip())

        return text
