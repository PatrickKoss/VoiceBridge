import re


class CapitalizationProcessor:
    def __init__(self):
        self.proper_nouns = self._load_common_proper_nouns()
        self.always_lowercase = self._load_always_lowercase_words()

    def process(self, text: str) -> str:
        if not text:
            return text

        # Split into sentences
        sentences = self._split_into_sentences(text)
        processed_sentences = []

        for sentence in sentences:
            if sentence.strip():
                processed_sentence = self._process_sentence(sentence)
                processed_sentences.append(processed_sentence)
            else:
                processed_sentences.append(sentence)

        return "".join(processed_sentences)

    def _split_into_sentences(self, text: str) -> list[str]:
        # Split on sentence-ending punctuation while preserving the punctuation
        sentences = re.split(r"([.!?]+\s*)", text)
        return sentences

    def _process_sentence(self, sentence: str) -> str:
        # Lowercase the entire sentence first
        sentence = sentence.lower()

        # Capitalize first letter of sentence
        sentence = self._capitalize_first_letter(sentence)

        # Capitalize "I" everywhere
        sentence = re.sub(r"\bi\b", "I", sentence)

        # Capitalize proper nouns
        sentence = self._capitalize_proper_nouns(sentence)

        # Fix contractions with "I"
        sentence = self._fix_i_contractions(sentence)

        return sentence

    def _capitalize_first_letter(self, sentence: str) -> str:
        sentence = sentence.strip()
        if sentence and sentence[0].isalpha():
            return sentence[0].upper() + sentence[1:]
        return sentence

    def _capitalize_proper_nouns(self, text: str) -> str:
        words = text.split()
        capitalized_words = []

        for word in words:
            # Extract the core word (without punctuation)
            core_word = re.sub(r"[^\w]", "", word).lower()

            if core_word in self.proper_nouns:
                # Capitalize while preserving punctuation
                if core_word in word.lower():
                    capitalized_word = word.lower().replace(
                        core_word, self.proper_nouns[core_word]
                    )
                    capitalized_words.append(capitalized_word)
                else:
                    capitalized_words.append(word)
            elif core_word in self.always_lowercase:
                # Keep lowercase (articles, prepositions, etc.)
                capitalized_words.append(word.lower())
            else:
                capitalized_words.append(word)

        return " ".join(capitalized_words)

    def _fix_i_contractions(self, text: str) -> str:
        # Fix contractions that should start with capital I
        contractions = [
            (r"\bi'm\b", "I'm"),
            (r"\bi'll\b", "I'll"),
            (r"\bi've\b", "I've"),
            (r"\bi'd\b", "I'd"),
            (r"\bi'mma\b", "I'mma"),
        ]

        for pattern, replacement in contractions:
            text = re.sub(pattern, replacement, text)

        return text

    def _load_common_proper_nouns(self) -> dict[str, str]:
        # Dictionary mapping lowercase -> proper capitalization
        return {
            # Days of the week
            "monday": "Monday",
            "tuesday": "Tuesday",
            "wednesday": "Wednesday",
            "thursday": "Thursday",
            "friday": "Friday",
            "saturday": "Saturday",
            "sunday": "Sunday",
            # Months
            "january": "January",
            "february": "February",
            "march": "March",
            "april": "April",
            "may": "May",
            "june": "June",
            "july": "July",
            "august": "August",
            "september": "September",
            "october": "October",
            "november": "November",
            "december": "December",
            # Countries/Continents
            "america": "America",
            "europe": "Europe",
            "asia": "Asia",
            "africa": "Africa",
            "australia": "Australia",
            "antarctica": "Antarctica",
            "usa": "USA",
            "uk": "UK",
            "canada": "Canada",
            "mexico": "Mexico",
            "japan": "Japan",
            "china": "China",
            "india": "India",
            "russia": "Russia",
            "germany": "Germany",
            "france": "France",
            "italy": "Italy",
            "spain": "Spain",
            "brazil": "Brazil",
            # Tech companies
            "google": "Google",
            "microsoft": "Microsoft",
            "apple": "Apple",
            "amazon": "Amazon",
            "facebook": "Facebook",
            "twitter": "Twitter",
            "netflix": "Netflix",
            "tesla": "Tesla",
            "uber": "Uber",
            "airbnb": "Airbnb",
            # Programming languages
            "python": "Python",
            "javascript": "JavaScript",
            "java": "Java",
            "rust": "Rust",
            "golang": "Go",
            "typescript": "TypeScript",
            "php": "PHP",
            "ruby": "Ruby",
            "swift": "Swift",
            "kotlin": "Kotlin",
            # Common names
            "john": "John",
            "mary": "Mary",
            "david": "David",
            "sarah": "Sarah",
            "michael": "Michael",
            "jennifer": "Jennifer",
            "william": "William",
            "elizabeth": "Elizabeth",
            "james": "James",
            "linda": "Linda",
        }

    def _load_always_lowercase_words(self) -> set[str]:
        # Words that should typically remain lowercase (except at sentence start)
        return {
            "a",
            "an",
            "the",  # articles
            "and",
            "or",
            "but",
            "nor",
            "for",
            "so",
            "yet",  # coordinating conjunctions
            "in",
            "on",
            "at",
            "by",
            "to",
            "of",
            "from",
            "with",
            "without",  # prepositions
            "is",
            "am",
            "are",
            "was",
            "were",
            "be",
            "being",
            "been",  # common verbs
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
        }
