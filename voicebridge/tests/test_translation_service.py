"""Tests for translation services."""

import json
import urllib.request
from unittest.mock import Mock, patch

import pytest

from voicebridge.domain.models import TranscriptionSegment
from voicebridge.services.translation_service import (
    GoogleTranslateService,
    LibreTranslateService,
    MockTranslationService,
)


class TestMockTranslationService:
    """Test cases for MockTranslationService."""

    @pytest.fixture
    def service(self):
        """MockTranslationService instance."""
        return MockTranslationService()

    def test_init(self, service):
        """Test service initialization."""
        assert "en" in service.supported_languages
        assert "es" in service.supported_languages
        assert service.supported_languages["en"] == "English"

    def test_translate_text_to_english(self, service):
        """Test translating text to English."""
        result = service.translate_text("Hello world", "en")
        assert result == "Hello world"

    def test_translate_text_to_spanish(self, service):
        """Test translating text to Spanish."""
        result = service.translate_text("Hello world", "es")
        assert result == "[ES] Hello world"

    def test_translate_text_to_unsupported_language(self, service):
        """Test translating text to unsupported language."""
        with pytest.raises(ValueError, match="Unsupported target language: xyz"):
            service.translate_text("Hello", "xyz")

    def test_translate_text_to_unmapped_language(self, service):
        """Test translating text to supported but unmapped language."""
        # Add a new supported language without mock translation
        service.supported_languages["new"] = "New Language"
        result = service.translate_text("Hello", "new")
        assert result == "[NEW] Hello"

    def test_translate_segments_empty_list(self, service):
        """Test translating empty segments list."""
        result = service.translate_segments([], "es")
        assert result == []

    def test_translate_segments_success(self, service):
        """Test successful segment translation."""
        segments = [
            TranscriptionSegment(
                text="Hello", start_time=0.0, end_time=1.0, confidence=0.9, speaker_id=1
            ),
            TranscriptionSegment(
                text="World", start_time=1.0, end_time=2.0, confidence=0.8, speaker_id=2
            ),
        ]

        result = service.translate_segments(segments, "es")

        assert len(result) == 2
        assert result[0].text == "[ES] Hello"
        assert result[0].start_time == 0.0
        assert result[0].end_time == 1.0
        assert result[0].confidence == 0.9
        assert result[0].speaker_id == 1

        assert result[1].text == "[ES] World"
        assert result[1].start_time == 1.0
        assert result[1].end_time == 2.0
        assert result[1].confidence == 0.8
        assert result[1].speaker_id == 2

    def test_detect_language_empty_text(self, service):
        """Test language detection with empty text."""
        assert service.detect_language("") == "en"
        assert service.detect_language("   ") == "en"

    def test_detect_language_spanish(self, service):
        """Test language detection for Spanish."""
        assert service.detect_language("Hola, ¿cómo estás?") == "es"
        assert service.detect_language("Niño con corazón") == "es"

    def test_detect_language_french(self, service):
        """Test language detection for French."""
        assert service.detect_language("Bonjour, ça va?") == "fr"
        assert service.detect_language("C'est très beau") == "fr"

    def test_detect_language_german(self, service):
        """Test language detection for German."""
        assert (
            service.detect_language("Guten Tag, wie geht's?") == "en"
        )  # No German chars
        assert service.detect_language("Weiß") == "de"  # Has ß (uniquely German)

    def test_detect_language_chinese(self, service):
        """Test language detection for Chinese."""
        assert service.detect_language("你好世界") == "zh"

    def test_detect_language_japanese(self, service):
        """Test language detection for Japanese."""
        assert service.detect_language("こんにちは") == "ja"

    def test_detect_language_korean(self, service):
        """Test language detection for Korean."""
        assert service.detect_language("안녕하세요") == "ko"

    def test_detect_language_arabic(self, service):
        """Test language detection for Arabic."""
        assert service.detect_language("مرحبا") == "ar"

    def test_detect_language_hindi(self, service):
        """Test language detection for Hindi."""
        assert service.detect_language("नमस्ते") == "hi"

    def test_detect_language_english_default(self, service):
        """Test language detection defaults to English."""
        assert service.detect_language("Hello world") == "en"
        assert service.detect_language("Regular English text") == "en"

    def test_get_supported_languages(self, service):
        """Test getting supported languages."""
        languages = service.get_supported_languages()

        assert isinstance(languages, dict)
        assert "en" in languages
        assert "es" in languages
        assert languages["en"] == "English"
        assert languages["es"] == "Spanish"

        # Should return a copy, not the original
        languages["test"] = "Test"
        assert "test" not in service.supported_languages


class TestLibreTranslateService:
    """Test cases for LibreTranslateService."""

    @pytest.fixture
    def service(self):
        """LibreTranslateService instance."""
        return LibreTranslateService(
            api_url="https://test.api/translate", api_key="test_key"
        )

    @pytest.fixture
    def service_no_key(self):
        """LibreTranslateService instance without API key."""
        return LibreTranslateService(api_url="https://test.api/translate")

    def test_init(self, service):
        """Test service initialization."""
        assert service.api_url == "https://test.api/translate"
        assert service.api_key == "test_key"
        assert service._supported_languages is None

    def test_init_no_key(self, service_no_key):
        """Test service initialization without API key."""
        assert service_no_key.api_key is None

    def test_translate_text_empty(self, service):
        """Test translating empty text."""
        assert service.translate_text("", "es") == ""
        assert service.translate_text("   ", "es") == "   "

    @patch.object(LibreTranslateService, "detect_language")
    def test_translate_text_same_language(self, mock_detect, service):
        """Test translating when source and target are same."""
        mock_detect.return_value = "es"

        result = service.translate_text("Hola", "es")
        assert result == "Hola"

    @patch.object(LibreTranslateService, "detect_language")
    @patch("urllib.request.urlopen")
    def test_translate_text_success(self, mock_urlopen, mock_detect, service):
        """Test successful text translation."""
        mock_detect.return_value = "en"

        # Mock response
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(
            {"translatedText": "Hola mundo"}
        ).encode()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=None)
        mock_urlopen.return_value = mock_response

        result = service.translate_text("Hello world", "es")
        assert result == "Hola mundo"

    @patch.object(LibreTranslateService, "detect_language")
    @patch("urllib.request.urlopen")
    def test_translate_text_with_api_key(self, mock_urlopen, mock_detect, service):
        """Test translation request includes API key."""
        mock_detect.return_value = "en"

        mock_response = Mock()
        mock_response.read.return_value = json.dumps(
            {"translatedText": "Hola"}
        ).encode()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=None)
        mock_urlopen.return_value = mock_response

        service.translate_text("Hello", "es")

        # Check that request was made with API key in data
        call_args = mock_urlopen.call_args[0][0]
        request_data = json.loads(call_args.data.decode())
        assert request_data["api_key"] == "test_key"

    @patch.object(LibreTranslateService, "detect_language")
    @patch("urllib.request.urlopen")
    def test_translate_text_without_api_key(
        self, mock_urlopen, mock_detect, service_no_key
    ):
        """Test translation request without API key."""
        mock_detect.return_value = "en"

        mock_response = Mock()
        mock_response.read.return_value = json.dumps(
            {"translatedText": "Hola"}
        ).encode()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=None)
        mock_urlopen.return_value = mock_response

        service_no_key.translate_text("Hello", "es")

        # Check that request was made without API key
        call_args = mock_urlopen.call_args[0][0]
        request_data = json.loads(call_args.data.decode())
        assert "api_key" not in request_data

    @patch.object(LibreTranslateService, "detect_language")
    @patch("urllib.request.urlopen")
    def test_translate_text_http_error(self, mock_urlopen, mock_detect, service):
        """Test translation with HTTP error."""
        mock_detect.return_value = "en"
        mock_urlopen.side_effect = urllib.request.HTTPError(
            None, 500, "Server Error", None, None
        )

        result = service.translate_text("Hello", "es")
        assert result == "Hello"  # Should return original text

    @patch.object(LibreTranslateService, "detect_language")
    @patch("urllib.request.urlopen")
    def test_translate_text_json_error(self, mock_urlopen, mock_detect, service):
        """Test translation with JSON decode error."""
        mock_detect.return_value = "en"

        mock_response = Mock()
        mock_response.read.return_value = b"invalid json"
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=None)
        mock_urlopen.return_value = mock_response

        result = service.translate_text("Hello", "es")
        assert result == "Hello"  # Should return original text

    def test_translate_segments_empty(self, service):
        """Test translating empty segments."""
        result = service.translate_segments([], "es")
        assert result == []

    @patch.object(LibreTranslateService, "translate_text")
    def test_translate_segments_success(self, mock_translate, service):
        """Test successful segment translation."""
        mock_translate.side_effect = ["Hola", "Mundo"]

        segments = [
            TranscriptionSegment(
                text="Hello", start_time=0.0, end_time=1.0, confidence=0.9
            ),
            TranscriptionSegment(
                text="World", start_time=1.0, end_time=2.0, confidence=0.8
            ),
        ]

        result = service.translate_segments(segments, "es")

        assert len(result) == 2
        assert result[0].text == "Hola"
        assert result[1].text == "Mundo"

    @patch.object(LibreTranslateService, "translate_text")
    def test_translate_segments_with_error(self, mock_translate, service):
        """Test segment translation with error."""
        mock_translate.side_effect = [Exception("Translation failed"), "Mundo"]

        segments = [
            TranscriptionSegment(
                text="Hello", start_time=0.0, end_time=1.0, confidence=0.9
            ),
            TranscriptionSegment(
                text="World", start_time=1.0, end_time=2.0, confidence=0.8
            ),
        ]

        result = service.translate_segments(segments, "es")

        assert len(result) == 2
        assert result[0].text == "Hello"  # Original segment kept
        assert result[1].text == "Mundo"  # Translated

    def test_detect_language_empty(self, service):
        """Test language detection with empty text."""
        assert service.detect_language("") == "en"

    @patch("urllib.request.urlopen")
    def test_detect_language_success(self, mock_urlopen, service):
        """Test successful language detection."""
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(
            [{"language": "es", "confidence": 0.9}]
        ).encode()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=None)
        mock_urlopen.return_value = mock_response

        result = service.detect_language("Hola mundo")
        assert result == "es"

    @patch("urllib.request.urlopen")
    def test_detect_language_with_api_key(self, mock_urlopen, service):
        """Test language detection includes API key."""
        mock_response = Mock()
        mock_response.read.return_value = json.dumps([{"language": "es"}]).encode()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=None)
        mock_urlopen.return_value = mock_response

        service.detect_language("Hola")

        # Check that request includes API key
        call_args = mock_urlopen.call_args[0][0]
        request_data = json.loads(call_args.data.decode())
        assert request_data["api_key"] == "test_key"

    @patch("urllib.request.urlopen")
    def test_detect_language_error(self, mock_urlopen, service):
        """Test language detection with error."""
        mock_urlopen.side_effect = Exception("Network error")

        result = service.detect_language("Hello")
        assert result == "en"  # Should fallback to English

    @patch("urllib.request.urlopen")
    def test_get_supported_languages_success(self, mock_urlopen, service):
        """Test successful supported languages retrieval."""
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(
            [
                {"code": "en", "name": "English"},
                {"code": "es", "name": "Spanish"},
            ]
        ).encode()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=None)
        mock_urlopen.return_value = mock_response

        result = service.get_supported_languages()

        assert result == {"en": "English", "es": "Spanish"}
        assert service._supported_languages == result  # Should cache result

    def test_get_supported_languages_cached(self, service):
        """Test that supported languages are cached."""
        service._supported_languages = {"cached": "result"}

        result = service.get_supported_languages()
        assert result == {"cached": "result"}

    @patch("urllib.request.urlopen")
    def test_get_supported_languages_error(self, mock_urlopen, service):
        """Test supported languages retrieval with error."""
        mock_urlopen.side_effect = Exception("Network error")

        result = service.get_supported_languages()

        # Should return fallback languages
        assert "en" in result
        assert "es" in result
        assert result["en"] == "English"


class TestGoogleTranslateService:
    """Test cases for GoogleTranslateService."""

    @pytest.fixture
    def service(self):
        """GoogleTranslateService instance."""
        return GoogleTranslateService(api_key="test_key")

    def test_init(self, service):
        """Test service initialization."""
        assert service.api_key == "test_key"
        assert "googleapis.com" in service.base_url

    def test_translate_text_empty(self, service):
        """Test translating empty text."""
        assert service.translate_text("", "es") == ""
        assert service.translate_text("   ", "es") == "   "

    @patch("urllib.request.urlopen")
    def test_translate_text_success(self, mock_urlopen, service):
        """Test successful text translation."""
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(
            {"data": {"translations": [{"translatedText": "Hola mundo"}]}}
        ).encode()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=None)
        mock_urlopen.return_value = mock_response

        result = service.translate_text("Hello world", "es")
        assert result == "Hola mundo"

    @patch("urllib.request.urlopen")
    def test_translate_text_includes_api_key(self, mock_urlopen, service):
        """Test that translation request includes API key in URL."""
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(
            {"data": {"translations": [{"translatedText": "Hola"}]}}
        ).encode()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=None)
        mock_urlopen.return_value = mock_response

        service.translate_text("Hello", "es")

        # Check that URL includes API key
        call_args = mock_urlopen.call_args[0][0]
        assert "key=test_key" in call_args.full_url

    @patch("urllib.request.urlopen")
    def test_translate_text_error(self, mock_urlopen, service):
        """Test translation with error."""
        mock_urlopen.side_effect = Exception("API error")

        result = service.translate_text("Hello", "es")
        assert result == "Hello"  # Should return original text

    @patch.object(GoogleTranslateService, "translate_text")
    def test_translate_segments(self, mock_translate, service):
        """Test segment translation."""
        mock_translate.side_effect = ["Hola", "Mundo"]

        segments = [
            TranscriptionSegment(
                text="Hello", start_time=0.0, end_time=1.0, confidence=0.9
            ),
            TranscriptionSegment(
                text="World", start_time=1.0, end_time=2.0, confidence=0.8
            ),
        ]

        result = service.translate_segments(segments, "es")

        assert len(result) == 2
        assert result[0].text == "Hola"
        assert result[1].text == "Mundo"

    @patch("urllib.request.urlopen")
    def test_detect_language_success(self, mock_urlopen, service):
        """Test successful language detection."""
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(
            {"data": {"detections": [[{"language": "es", "confidence": 0.9}]]}}
        ).encode()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=None)
        mock_urlopen.return_value = mock_response

        result = service.detect_language("Hola mundo")
        assert result == "es"

    @patch("urllib.request.urlopen")
    def test_detect_language_error(self, mock_urlopen, service):
        """Test language detection with error."""
        mock_urlopen.side_effect = Exception("API error")

        result = service.detect_language("Hello")
        assert result == "en"  # Should fallback to English

    @patch("urllib.request.urlopen")
    def test_get_supported_languages_success(self, mock_urlopen, service):
        """Test successful supported languages retrieval."""
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(
            {
                "data": {
                    "languages": [
                        {"language": "en", "name": "English"},
                        {"language": "es", "name": "Spanish"},
                    ]
                }
            }
        ).encode()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=None)
        mock_urlopen.return_value = mock_response

        result = service.get_supported_languages()
        assert result == {"en": "English", "es": "Spanish"}

    @patch("urllib.request.urlopen")
    def test_get_supported_languages_error(self, mock_urlopen, service):
        """Test supported languages retrieval with error."""
        mock_urlopen.side_effect = Exception("API error")

        result = service.get_supported_languages()

        # Should return fallback languages
        assert "en" in result
        assert "es" in result
