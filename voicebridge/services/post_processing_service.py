import re

from voicebridge.domain.models import PostProcessingConfig
from voicebridge.ports.interfaces import PostProcessingService


class WhisperPostProcessingService(PostProcessingService):
    def __init__(self):
        self._profanity_words = self._load_default_profanity_list()

    def process_text(self, text: str, config: PostProcessingConfig) -> str:
        if not text:
            return text

        processed_text = text

        # Apply processing steps in order
        if config.text_normalization:
            processed_text = self._normalize_text(processed_text)

        if config.remove_filler_words:
            processed_text = self._remove_filler_words(
                processed_text, config.filler_words
            )

        if config.enable_punctuation_cleanup:
            processed_text = self.clean_punctuation(processed_text)

        if config.enable_capitalization:
            processed_text = self.normalize_capitalization(processed_text)

        if config.sentence_segmentation:
            processed_text = self._segment_sentences(processed_text)

        if config.enable_profanity_filter:
            processed_text = self.filter_profanity(processed_text)

        if config.custom_replacements:
            processed_text = self._apply_custom_replacements(
                processed_text, config.custom_replacements
            )

        # Ensure text ends with proper punctuation
        processed_text = processed_text.strip()
        if processed_text and processed_text[-1] not in ".!?":
            processed_text += "."

        return processed_text

    def clean_punctuation(self, text: str) -> str:
        # Remove excessive punctuation and normalize spacing
        text = re.sub(r"[.]{2,}", "...", text)  # Multiple dots to ellipsis
        text = re.sub(r"[!]{2,}", "!", text)  # Multiple exclamation marks
        text = re.sub(r"[?]{2,}", "?", text)  # Multiple question marks

        # Fix spacing around punctuation
        text = re.sub(r"\s+([.!?,:;])", r"\1", text)  # Remove space before punctuation
        text = re.sub(
            r"([!?])([A-Za-z])", r"\1 \2", text
        )  # Space after ! or ? before letters
        text = re.sub(
            r"([^.])(\.)([A-Za-z])", r"\1\2 \3", text
        )  # Space after single period before letters (not ellipsis)
        text = re.sub(
            r"([,:;])\s*", r"\1 ", text
        )  # Space after commas, colons, semicolons

        return text

    def normalize_capitalization(self, text: str) -> str:
        sentences = re.split(r"([.!?]+\s*)", text)
        capitalized_sentences = []

        for i, sentence in enumerate(sentences):
            if i % 2 == 0 and sentence.strip():  # Actual sentence (not punctuation)
                # Capitalize first letter of sentence
                sentence = sentence.strip()
                if sentence:
                    sentence = sentence[0].upper() + sentence[1:].lower()
                    # Capitalize "I"
                    sentence = re.sub(r"\bi\b", "I", sentence)
                    # Capitalize proper nouns (basic heuristic)
                    sentence = self._capitalize_proper_nouns(sentence)
                capitalized_sentences.append(sentence)
            else:
                capitalized_sentences.append(sentence)

        return "".join(capitalized_sentences)

    def filter_profanity(self, text: str) -> str:
        words = text.split()
        filtered_words = []

        for word in words:
            clean_word = re.sub(r"[^\w]", "", word.lower())
            if clean_word in self._profanity_words:
                # Replace with asterisks, preserving original length
                replacement = "*" * len(clean_word)
                filtered_word = re.sub(
                    re.escape(clean_word), replacement, word, flags=re.IGNORECASE
                )
                filtered_words.append(filtered_word)
            else:
                filtered_words.append(word)

        return " ".join(filtered_words)

    def _normalize_text(self, text: str) -> str:
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Normalize quotes
        text = re.sub(r'[""' "`]", '"', text)

        # Normalize dashes
        text = re.sub(r"[–—]", "-", text)

        return text.strip()

    def _remove_filler_words(self, text: str, filler_words: list[str]) -> str:
        if not filler_words:
            return text

        # Create pattern for filler words (case-insensitive, word boundaries)
        pattern = r"\b(?:" + "|".join(re.escape(word) for word in filler_words) + r")\b"

        # Remove filler words but preserve sentence structure
        result = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Clean up extra spaces
        result = re.sub(r"\s+", " ", result)

        return result.strip()

    def _segment_sentences(self, text: str) -> str:
        # Add proper spacing after sentence endings
        text = re.sub(r"([.!?])([A-Z])", r"\1 \2", text)

        # Ensure text ends with proper punctuation
        text = text.strip()
        if text and text[-1] not in ".!?":
            text += "."

        return text

    def _apply_custom_replacements(
        self, text: str, replacements: dict[str, str]
    ) -> str:
        for old, new in replacements.items():
            text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)
        return text

    def _capitalize_proper_nouns(self, text: str) -> str:
        # Basic proper noun capitalization (common names, places, etc.)
        common_proper_nouns = [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
            "america",
            "europe",
            "asia",
            "africa",
            "australia",
            "antarctica",
            "google",
            "microsoft",
            "apple",
            "amazon",
            "facebook",
            "twitter",
            "python",
            "javascript",
            "java",
            "cpp",
            "rust",
            "golang",
        ]

        for noun in common_proper_nouns:
            pattern = r"\b" + re.escape(noun) + r"\b"
            text = re.sub(pattern, noun.title(), text, flags=re.IGNORECASE)

        return text

    def _load_default_profanity_list(self) -> set[str]:
        # Basic profanity list - in a real implementation, this might be loaded from a file
        return {
            "damn",
            "hell",
            "shit",
            "fuck",
            "bitch",
            "asshole",
            "bastard",
            "crap",
            "piss",
            "cock",
            "dick",
            "pussy",
            "slut",
            "whore",
        }
