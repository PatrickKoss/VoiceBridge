import re


class ProfanityFilter:
    def __init__(self, custom_word_list: list[str] = None):
        self.profanity_words = self._load_profanity_words()
        self.severity_levels = self._load_severity_levels()
        self.leetspeak_map = self._load_leetspeak_mappings()

        if custom_word_list:
            self.profanity_words.update(word.lower() for word in custom_word_list)

    def filter_text(
        self, text: str, replacement_char: str = "*", severity_threshold: int = 1
    ) -> str:
        if not text:
            return text

        words = text.split()
        filtered_words = []

        for word in words:
            if self._is_profane(word, severity_threshold):
                filtered_word = self._replace_profanity(word, replacement_char)
                filtered_words.append(filtered_word)
            else:
                filtered_words.append(word)

        return " ".join(filtered_words)

    def detect_profanity(
        self, text: str, severity_threshold: int = 1
    ) -> list[dict[str, any]]:
        if not text:
            return []

        detections = []
        words = text.split()

        for i, word in enumerate(words):
            clean_word = self._clean_word(word)
            if self._is_profane_word(clean_word, severity_threshold):
                detections.append(
                    {
                        "word": word,
                        "position": i,
                        "severity": self.severity_levels.get(clean_word, 1),
                        "variants": self._find_variants(clean_word),
                    }
                )

        return detections

    def _is_profane(self, word: str, severity_threshold: int = 1) -> bool:
        clean_word = self._clean_word(word)
        return self._is_profane_word(clean_word, severity_threshold)

    def _is_profane_word(self, word: str, severity_threshold: int = 1) -> bool:
        word_lower = word.lower()

        # Direct match
        if word_lower in self.profanity_words:
            severity = self.severity_levels.get(word_lower, 1)
            return severity >= severity_threshold

        # Check for leetspeak variants
        normalized_word = self._normalize_leetspeak(word_lower)
        if normalized_word in self.profanity_words:
            severity = self.severity_levels.get(normalized_word, 1)
            return severity >= severity_threshold

        # Check for partial matches (substring)
        for profane_word in self.profanity_words:
            if len(profane_word) > 3 and profane_word in word_lower:
                severity = self.severity_levels.get(profane_word, 1)
                if severity >= severity_threshold:
                    return True

        return False

    def _clean_word(self, word: str) -> str:
        # Remove punctuation and special characters
        return re.sub(r"[^\w]", "", word)

    def _replace_profanity(self, word: str, replacement_char: str = "*") -> str:
        clean_word = self._clean_word(word)

        # Find profane parts and replace them
        word_lower = clean_word.lower()

        # Direct replacement
        if word_lower in self.profanity_words:
            replacement = replacement_char * len(clean_word)
            return self._preserve_punctuation(word, clean_word, replacement)

        # Replace leetspeak variants
        normalized = self._normalize_leetspeak(word_lower)
        if normalized in self.profanity_words:
            replacement = replacement_char * len(clean_word)
            return self._preserve_punctuation(word, clean_word, replacement)

        # Replace partial matches
        result = clean_word
        for profane_word in self.profanity_words:
            if len(profane_word) > 3 and profane_word in word_lower:
                replacement = replacement_char * len(profane_word)
                result = re.sub(
                    re.escape(profane_word), replacement, result, flags=re.IGNORECASE
                )

        return self._preserve_punctuation(word, clean_word, result)

    def _preserve_punctuation(self, original: str, clean: str, replacement: str) -> str:
        # Extract punctuation from original word
        leading_punct = re.match(r"^[^\w]*", original).group()
        trailing_punct = re.search(r"[^\w]*$", original).group()

        return leading_punct + replacement + trailing_punct

    def _normalize_leetspeak(self, word: str) -> str:
        normalized = word
        for leet, normal in self.leetspeak_map.items():
            normalized = normalized.replace(leet, normal)
        return normalized

    def _find_variants(self, word: str) -> list[str]:
        variants = []

        # Add leetspeak variant
        leet_variant = word
        for normal, leet in self.leetspeak_map.items():
            leet_variant = leet_variant.replace(normal, leet)
        if leet_variant != word:
            variants.append(leet_variant)

        return variants

    def _load_profanity_words(self) -> set[str]:
        # Basic profanity list - in production, this might be loaded from a file
        return {
            # Mild profanity (severity 1)
            "damn",
            "hell",
            "crap",
            "piss",
            # Moderate profanity (severity 2)
            "shit",
            "bitch",
            "bastard",
            "asshole",
            "ass",
            # Strong profanity (severity 3)
            "fuck",
            "fucking",
            "motherfucker",
            "fucker",
            "cock",
            "dick",
            "pussy",
            "cunt",
            "whore",
            "slut",
            # Offensive slurs (severity 4) - these should be handled with extreme care
            # Note: In a real implementation, you'd want a more comprehensive and
            # carefully curated list, possibly loaded from an external source
        }

    def _load_severity_levels(self) -> dict[str, int]:
        return {
            # Mild (1)
            "damn": 1,
            "hell": 1,
            "crap": 1,
            "piss": 1,
            # Moderate (2)
            "shit": 2,
            "bitch": 2,
            "bastard": 2,
            "asshole": 2,
            "ass": 2,
            # Strong (3)
            "fuck": 3,
            "fucking": 3,
            "motherfucker": 3,
            "fucker": 3,
            "cock": 3,
            "dick": 3,
            "pussy": 3,
            "cunt": 3,
            "whore": 3,
            "slut": 3,
        }

    def _load_leetspeak_mappings(self) -> dict[str, str]:
        return {
            "4": "a",
            "@": "a",
            "3": "e",
            "1": "i",
            "!": "i",
            "0": "o",
            "5": "s",
            "$": "s",
            "7": "t",
            "2": "z",
        }

    def add_custom_words(self, words: list[str], severity: int = 2) -> None:
        for word in words:
            word_lower = word.lower()
            self.profanity_words.add(word_lower)
            self.severity_levels[word_lower] = severity

    def remove_words(self, words: list[str]) -> None:
        for word in words:
            word_lower = word.lower()
            self.profanity_words.discard(word_lower)
            self.severity_levels.pop(word_lower, None)
