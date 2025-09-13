import json
import urllib.request
from urllib.error import HTTPError, URLError

from voicebridge.domain.models import TranscriptionSegment
from voicebridge.ports.interfaces import TranslationService


class MockTranslationService(TranslationService):
    """Mock translation service for testing - replace with real service in production"""

    def __init__(self):
        self.supported_languages = {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ja": "Japanese",
            "ko": "Korean",
            "zh": "Chinese",
            "ar": "Arabic",
            "hi": "Hindi",
        }

    def translate_text(self, text: str, target_language: str) -> str:
        """Mock translation - in production, use Google Translate API or similar"""
        if target_language not in self.supported_languages:
            raise ValueError(f"Unsupported target language: {target_language}")

        # Mock translations for demonstration
        mock_translations = {
            "es": f"[ES] {text}",  # Spanish
            "fr": f"[FR] {text}",  # French
            "de": f"[DE] {text}",  # German
            "it": f"[IT] {text}",  # Italian
            "pt": f"[PT] {text}",  # Portuguese
            "ru": f"[RU] {text}",  # Russian
            "ja": f"[JA] {text}",  # Japanese
            "ko": f"[KO] {text}",  # Korean
            "zh": f"[ZH] {text}",  # Chinese
            "ar": f"[AR] {text}",  # Arabic
            "hi": f"[HI] {text}",  # Hindi
        }

        if target_language == "en":
            return text

        return mock_translations.get(
            target_language, f"[{target_language.upper()}] {text}"
        )

    def translate_segments(
        self, segments: list[TranscriptionSegment], target_language: str
    ) -> list[TranscriptionSegment]:
        """Translate all segments while preserving timing and metadata"""
        translated_segments = []

        for segment in segments:
            translated_text = self.translate_text(segment.text, target_language)
            translated_segment = TranscriptionSegment(
                text=translated_text,
                start_time=segment.start_time,
                end_time=segment.end_time,
                confidence=segment.confidence,
                speaker_id=segment.speaker_id,
            )
            translated_segments.append(translated_segment)

        return translated_segments

    def detect_language(self, text: str) -> str:
        """Mock language detection - always returns 'en' for now"""
        # In production, use language detection API
        if not text.strip():
            return "en"

        # Simple heuristics for common languages (very basic)
        if any(char in text for char in "áéíóúñü¿¡"):
            return "es"
        elif any(char in text for char in "àâäéèêëïîôöùûüÿç"):
            return "fr"
        elif any(char in text for char in "äöüß"):
            return "de"
        elif any(ord(char) >= 0x4E00 and ord(char) <= 0x9FFF for char in text):
            return "zh"
        elif any(ord(char) >= 0x3040 and ord(char) <= 0x309F for char in text):
            return "ja"
        elif any(ord(char) >= 0xAC00 and ord(char) <= 0xD7AF for char in text):
            return "ko"
        elif any(ord(char) >= 0x0600 and ord(char) <= 0x06FF for char in text):
            return "ar"
        elif any(ord(char) >= 0x0900 and ord(char) <= 0x097F for char in text):
            return "hi"

        return "en"  # Default to English

    def get_supported_languages(self) -> dict[str, str]:
        """Return dictionary of supported language codes and names"""
        return self.supported_languages.copy()


class LibreTranslateService(TranslationService):
    """LibreTranslate API service - free and open source translation"""

    def __init__(
        self,
        api_url: str = "https://libretranslate.de/translate",
        api_key: str | None = None,
    ):
        self.api_url = api_url
        self.api_key = api_key
        self._supported_languages = None

    def translate_text(self, text: str, target_language: str) -> str:
        """Translate text using LibreTranslate API"""
        if not text.strip():
            return text

        detected_lang = self.detect_language(text)
        if detected_lang == target_language:
            return text

        data = {
            "q": text,
            "source": detected_lang,
            "target": target_language,
            "format": "text",
        }

        if self.api_key:
            data["api_key"] = self.api_key

        try:
            req_data = json.dumps(data).encode("utf-8")
            req = urllib.request.Request(
                self.api_url,
                data=req_data,
                headers={"Content-Type": "application/json"},
            )

            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result.get("translatedText", text)

        except (HTTPError, URLError, json.JSONDecodeError, KeyError) as e:
            # Fallback to original text if translation fails
            print(f"Translation failed: {e}")
            return text

    def translate_segments(
        self, segments: list[TranscriptionSegment], target_language: str
    ) -> list[TranscriptionSegment]:
        """Translate all segments in batch for efficiency"""
        if not segments:
            return segments

        translated_segments = []

        # Batch translate for efficiency

        for i, segment in enumerate(segments):
            try:
                translated_text = self.translate_text(segment.text, target_language)
                translated_segment = TranscriptionSegment(
                    text=translated_text,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    confidence=segment.confidence,
                    speaker_id=segment.speaker_id,
                )
                translated_segments.append(translated_segment)
            except Exception as e:
                print(f"Failed to translate segment {i}: {e}")
                # Keep original segment on failure
                translated_segments.append(segment)

        return translated_segments

    def detect_language(self, text: str) -> str:
        """Detect language using LibreTranslate API"""
        if not text.strip():
            return "en"

        detect_url = self.api_url.replace("/translate", "/detect")
        data = {"q": text[:500]}  # Use first 500 chars for detection

        if self.api_key:
            data["api_key"] = self.api_key

        try:
            req_data = json.dumps(data).encode("utf-8")
            req = urllib.request.Request(
                detect_url, data=req_data, headers={"Content-Type": "application/json"}
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode("utf-8"))
                if isinstance(result, list) and result:
                    return result[0].get("language", "en")
                return "en"

        except Exception as e:
            print(f"Language detection failed: {e}")
            return "en"  # Fallback to English

    def get_supported_languages(self) -> dict[str, str]:
        """Get supported languages from LibreTranslate API"""
        if self._supported_languages:
            return self._supported_languages

        languages_url = self.api_url.replace("/translate", "/languages")

        try:
            req = urllib.request.Request(languages_url)
            with urllib.request.urlopen(req, timeout=10) as response:
                languages = json.loads(response.read().decode("utf-8"))

            self._supported_languages = {
                lang["code"]: lang["name"] for lang in languages
            }
            return self._supported_languages

        except Exception as e:
            print(f"Failed to fetch supported languages: {e}")
            # Fallback to common languages
            return {
                "en": "English",
                "es": "Spanish",
                "fr": "French",
                "de": "German",
                "it": "Italian",
                "pt": "Portuguese",
                "ru": "Russian",
                "ja": "Japanese",
                "ko": "Korean",
                "zh": "Chinese",
                "ar": "Arabic",
            }


class GoogleTranslateService(TranslationService):
    """Google Translate API service - requires API key"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://translation.googleapis.com/language/translate/v2"

    def translate_text(self, text: str, target_language: str) -> str:
        """Translate text using Google Translate API"""
        if not text.strip():
            return text

        url = f"{self.base_url}?key={self.api_key}"
        data = {"q": text, "target": target_language, "format": "text"}

        try:
            req_data = json.dumps(data).encode("utf-8")
            req = urllib.request.Request(
                url, data=req_data, headers={"Content-Type": "application/json"}
            )

            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result["data"]["translations"][0]["translatedText"]

        except Exception as e:
            print(f"Google Translate failed: {e}")
            return text

    def translate_segments(
        self, segments: list[TranscriptionSegment], target_language: str
    ) -> list[TranscriptionSegment]:
        """Translate segments using Google Translate API"""
        translated_segments = []

        for segment in segments:
            translated_text = self.translate_text(segment.text, target_language)
            translated_segment = TranscriptionSegment(
                text=translated_text,
                start_time=segment.start_time,
                end_time=segment.end_time,
                confidence=segment.confidence,
                speaker_id=segment.speaker_id,
            )
            translated_segments.append(translated_segment)

        return translated_segments

    def detect_language(self, text: str) -> str:
        """Detect language using Google Translate API"""
        url = f"{self.base_url}/detect?key={self.api_key}"
        data = {"q": text[:500]}

        try:
            req_data = json.dumps(data).encode("utf-8")
            req = urllib.request.Request(
                url, data=req_data, headers={"Content-Type": "application/json"}
            )

            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result["data"]["detections"][0][0]["language"]

        except Exception:
            return "en"

    def get_supported_languages(self) -> dict[str, str]:
        """Get supported languages from Google Translate API"""
        url = f"{self.base_url}/languages?key={self.api_key}"

        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode("utf-8"))

            return {
                lang["language"]: lang.get("name", lang["language"])
                for lang in result["data"]["languages"]
            }
        except Exception:
            return {
                "en": "English",
                "es": "Spanish",
                "fr": "French",
                "de": "German",
            }
