import re


class PunctuationProcessor:
    def __init__(self):
        self.punctuation_rules = {
            # Multiple punctuation normalization
            r"[.]{2,}": "...",
            r"[!]{2,}": "!",
            r"[?]{2,}": "?",
            r"[,]{2,}": ",",
            # Spacing fixes
            r"\s+([.!?,:;])": r"\1",  # Remove space before punctuation
            r"([.!?])\s*([A-Z])": r"\1 \2",  # Space after sentence-ending punctuation
            r"([,:;])\s*": r"\1 ",  # Space after commas, colons, semicolons
            # Quote normalization
            r'\s*"\s*': ' "',  # Proper quote spacing
            r'"\s*([.!?,])': r'"\1',  # No space between quote and punctuation
        }

    def process(self, text: str) -> str:
        if not text:
            return text

        processed_text = text

        # Apply all punctuation rules
        for pattern, replacement in self.punctuation_rules.items():
            processed_text = re.sub(pattern, replacement, processed_text)

        # Additional cleanup
        processed_text = self._fix_apostrophes(processed_text)
        processed_text = self._fix_contractions(processed_text)
        processed_text = self._normalize_dashes(processed_text)

        return processed_text.strip()

    def _fix_apostrophes(self, text: str) -> str:
        # Common apostrophe fixes
        contractions = {
            r"\bcan't\b": "can't",
            r"\bwon't\b": "won't",
            r"\bdont\b": "don't",
            r"\bisnt\b": "isn't",
            r"\barent\b": "aren't",
            r"\bwasnt\b": "wasn't",
            r"\bwerent\b": "weren't",
            r"\bhadnt\b": "hadn't",
            r"\bhasnt\b": "hasn't",
            r"\bhavent\b": "haven't",
            r"\bshouldnt\b": "shouldn't",
            r"\bwouldnt\b": "wouldn't",
            r"\bcouldnt\b": "couldn't",
            r"\bdidnt\b": "didn't",
            r"\bdoesnt\b": "doesn't",
            r"\bthats\b": "that's",
            r"\bits\b": "it's",  # Be careful with possessive "its"
            r"\byoure\b": "you're",
            r"\btheyre\b": "they're",
            r"\bwere\b": "we're",
            r"\bId\b": "I'd",
            r"\bIll\b": "I'll",
            r"\bIm\b": "I'm",
            r"\bIve\b": "I've",
        }

        for pattern, replacement in contractions.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def _fix_contractions(self, text: str) -> str:
        # Fix common transcription errors in contractions
        text = re.sub(r"\b(\w+)\s+n\s*'\s*t\b", r"\1n't", text)  # "do n't" -> "don't"
        text = re.sub(r"\b(\w+)\s+'\s*s\b", r"\1's", text)  # "it 's" -> "it's"
        text = re.sub(r"\b(\w+)\s+'\s*re\b", r"\1're", text)  # "you 're" -> "you're"
        text = re.sub(r"\b(\w+)\s+'\s*ve\b", r"\1've", text)  # "I 've" -> "I've"
        text = re.sub(r"\b(\w+)\s+'\s*ll\b", r"\1'll", text)  # "I 'll" -> "I'll"
        text = re.sub(r"\b(\w+)\s+'\s*d\b", r"\1'd", text)  # "I 'd" -> "I'd"

        return text

    def _normalize_dashes(self, text: str) -> str:
        # Normalize different types of dashes
        text = re.sub(r"[–—]", "-", text)  # Em/en dashes to hyphens
        text = re.sub(r"\s*-\s*", " - ", text)  # Proper spacing around dashes

        return text
