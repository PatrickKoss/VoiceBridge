import re
from difflib import SequenceMatcher

from voicebridge.domain.models import VocabularyConfig
from voicebridge.ports.interfaces import VocabularyService


class WhisperVocabularyService(VocabularyService):
    def __init__(self):
        self._loaded_vocabulary: dict[str, list[str]] = {}
        self._phonetic_mappings: dict[str, str] = {}
        self._boost_factor: float = 2.0
        self._enable_fuzzy_matching: bool = True

    def load_vocabulary(self, config: VocabularyConfig) -> None:
        self._loaded_vocabulary.clear()
        self._phonetic_mappings = config.phonetic_mappings.copy()
        self._boost_factor = config.boost_factor
        self._enable_fuzzy_matching = config.enable_fuzzy_matching

        # Load all vocabulary types
        self._loaded_vocabulary["custom"] = config.custom_words
        self._loaded_vocabulary["proper_nouns"] = config.proper_nouns
        self._loaded_vocabulary["technical"] = config.technical_jargon

        # Flatten domain terms
        all_domain_terms = []
        for domain_terms in config.domain_terms.values():
            all_domain_terms.extend(domain_terms)
        self._loaded_vocabulary["domain"] = all_domain_terms

    def enhance_transcription(self, text: str, config: VocabularyConfig) -> str:
        if not text or not self._loaded_vocabulary:
            return text

        enhanced_text = text

        # Apply phonetic mappings first
        enhanced_text = self._apply_phonetic_corrections(enhanced_text)

        # Apply vocabulary corrections
        enhanced_text = self._apply_vocabulary_corrections(enhanced_text)

        return enhanced_text

    def add_domain_terms(self, domain: str, terms: list[str]) -> None:
        if "domain" not in self._loaded_vocabulary:
            self._loaded_vocabulary["domain"] = []

        for term in terms:
            if term not in self._loaded_vocabulary["domain"]:
                self._loaded_vocabulary["domain"].append(term)

    def _apply_phonetic_corrections(self, text: str) -> str:
        corrected_text = text
        for incorrect, correct in self._phonetic_mappings.items():
            # Case-insensitive replacement while preserving original case
            pattern = re.compile(re.escape(incorrect), re.IGNORECASE)
            corrected_text = pattern.sub(correct, corrected_text)
        return corrected_text

    def _apply_vocabulary_corrections(self, text: str) -> str:
        words = text.split()
        corrected_words = []

        for word in words:
            corrected_word = self._find_best_match(word)
            corrected_words.append(corrected_word)

        return " ".join(corrected_words)

    def _find_best_match(self, word: str) -> str:
        if not word:
            return word

        # Clean word for comparison (remove punctuation)
        clean_word = re.sub(r"[^\w]", "", word.lower())

        # Check for exact matches first
        for _vocab_type, vocab_list in self._loaded_vocabulary.items():
            for vocab_word in vocab_list:
                if clean_word == vocab_word.lower():
                    return self._preserve_case_and_punctuation(word, vocab_word)

        # If fuzzy matching is enabled, find close matches
        if self._enable_fuzzy_matching:
            best_match = self._find_fuzzy_match(clean_word)
            if best_match:
                return self._preserve_case_and_punctuation(word, best_match)

        return word

    def _find_fuzzy_match(self, word: str, threshold: float = 0.8) -> str | None:
        best_ratio = 0.0
        best_match = None

        for _vocab_type, vocab_list in self._loaded_vocabulary.items():
            for vocab_word in vocab_list:
                ratio = SequenceMatcher(None, word, vocab_word.lower()).ratio()
                if ratio > best_ratio and ratio >= threshold:
                    best_ratio = ratio
                    best_match = vocab_word

        return best_match

    def _preserve_case_and_punctuation(self, original: str, replacement: str) -> str:
        # Preserve leading/trailing punctuation and case pattern
        leading_punct = re.match(r"^[^\w]*", original).group()
        trailing_punct = re.search(r"[^\w]*$", original).group()

        # Apply case pattern from original to replacement
        if original.isupper():
            replacement = replacement.upper()
        elif original.istitle():
            replacement = replacement.title()
        elif original.islower():
            replacement = replacement.lower()

        return leading_punct + replacement + trailing_punct
