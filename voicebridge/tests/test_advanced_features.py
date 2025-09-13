import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from voicebridge.adapters.circuit_breaker import (
    CircuitBreakerError,
    WhisperCircuitBreakerService,
)
from voicebridge.adapters.vocabulary import VocabularyAdapter
from voicebridge.domain.models import (
    EventType,
    PostProcessingConfig,
    RetryConfig,
    RetryStrategy,
    VocabularyConfig,
    WebhookEvent,
)
from voicebridge.services.post_processing_service import WhisperPostProcessingService
from voicebridge.services.progress_service import WhisperProgressService
from voicebridge.services.retry_service import WhisperRetryService
from voicebridge.services.vocabulary_service import WhisperVocabularyService


class TestVocabularyService:
    @pytest.fixture
    def vocabulary_service(self):
        return WhisperVocabularyService()

    @pytest.fixture
    def vocabulary_config(self):
        return VocabularyConfig(
            custom_words=["datacenter", "kubernetes"],
            proper_nouns=["OpenAI", "Google"],
            technical_jargon=["API", "JSON", "HTTP"],
            phonetic_mappings={"koo-ber-net-ees": "kubernetes"},
            boost_factor=2.0,
            enable_fuzzy_matching=True,
        )

    def test_load_vocabulary(self, vocabulary_service, vocabulary_config):
        vocabulary_service.load_vocabulary(vocabulary_config)

        assert "datacenter" in vocabulary_service._loaded_vocabulary["custom"]
        assert "OpenAI" in vocabulary_service._loaded_vocabulary["proper_nouns"]
        assert vocabulary_service._boost_factor == 2.0

    def test_enhance_transcription(self, vocabulary_service, vocabulary_config):
        vocabulary_service.load_vocabulary(vocabulary_config)

        # Test phonetic correction
        text = "I use koo-ber-net-ees for container orchestration"
        enhanced = vocabulary_service.enhance_transcription(text, vocabulary_config)
        assert "kubernetes" in enhanced

    def test_fuzzy_matching(self, vocabulary_service, vocabulary_config):
        vocabulary_service.load_vocabulary(vocabulary_config)

        # Test fuzzy matching for close words
        text = "I work at googl"  # Close to "Google"
        enhanced = vocabulary_service.enhance_transcription(text, vocabulary_config)
        # Should find "Google" as close match
        assert enhanced != text


class TestPostProcessingService:
    @pytest.fixture
    def postprocessing_service(self):
        return WhisperPostProcessingService()

    @pytest.fixture
    def postprocessing_config(self):
        return PostProcessingConfig(
            enable_punctuation_cleanup=True,
            enable_capitalization=True,
            enable_profanity_filter=True,
            remove_filler_words=True,
            filler_words=["um", "uh", "like"],
        )

    def test_punctuation_cleanup(self, postprocessing_service):
        text = "hello...world!!!how are you???"
        cleaned = postprocessing_service.clean_punctuation(text)
        assert cleaned == "hello...world! how are you?"

    def test_capitalization(self, postprocessing_service):
        text = "hello world. i am fine. how are you?"
        capitalized = postprocessing_service.normalize_capitalization(text)
        assert capitalized.startswith("Hello world. I am fine.")

    def test_profanity_filter(self, postprocessing_service):
        text = "This is damn good work"
        filtered = postprocessing_service.filter_profanity(text)
        assert "****" in filtered
        assert "damn" not in filtered

    def test_filler_word_removal(self, postprocessing_service, postprocessing_config):
        text = "So um I think like this is uh really good"
        processed = postprocessing_service._remove_filler_words(
            text, ["um", "uh", "like"]
        )
        assert "um" not in processed
        assert "uh" not in processed
        assert "like" not in processed
        assert "really good" in processed

    def test_full_processing_pipeline(
        self, postprocessing_service, postprocessing_config
    ):
        text = "hello world...this is um really good work damn it"
        processed = postprocessing_service.process_text(text, postprocessing_config)

        # Should be capitalized, punctuation cleaned, filler words removed, profanity filtered
        assert processed.startswith("Hello")
        assert "um" not in processed
        assert "***" in processed  # damn filtered
        assert processed.endswith(".")


class TestRetryService:
    @pytest.fixture
    def retry_service(self):
        return WhisperRetryService()

    @pytest.fixture
    def retry_config(self):
        return RetryConfig(
            max_attempts=3,
            initial_delay=0.1,  # Short for tests
            max_delay=1.0,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            retryable_errors=["ConnectionError", "TimeoutError"],
        )

    def test_successful_operation(self, retry_service, retry_config):
        def successful_operation():
            return "success"

        result = retry_service.execute_with_retry(successful_operation, retry_config)
        assert result == "success"

    def test_retryable_error(self, retry_service, retry_config):
        attempt_count = 0

        def failing_then_succeeding_operation():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Network error")
            return "success"

        result = retry_service.execute_with_retry(
            failing_then_succeeding_operation, retry_config
        )
        assert result == "success"
        assert attempt_count == 3

    def test_non_retryable_error(self, retry_service, retry_config):
        def non_retryable_operation():
            raise ValueError("Not retryable")

        with pytest.raises(ValueError):
            retry_service.execute_with_retry(non_retryable_operation, retry_config)

    def test_max_attempts_exceeded(self, retry_service, retry_config):
        def always_failing_operation():
            raise ConnectionError("Always fails")

        with pytest.raises(ConnectionError):
            retry_service.execute_with_retry(always_failing_operation, retry_config)


class TestProgressService:
    @pytest.fixture
    def progress_service(self):
        return WhisperProgressService()

    def test_create_tracker(self, progress_service):
        tracker = progress_service.create_tracker("op1", "transcription", 5)

        assert tracker.operation_id == "op1"
        assert tracker.operation_type == "transcription"
        assert tracker.total_steps == 5
        assert tracker.current_progress == 0.0
        assert tracker.status == "running"

    def test_update_progress(self, progress_service):
        progress_service.create_tracker("op1", "transcription", 4)

        progress_service.update_progress("op1", 0.5, "Processing audio")

        updated_tracker = progress_service.get_tracker("op1")
        assert updated_tracker.current_progress == 0.5
        assert updated_tracker.current_step == "Processing audio"
        assert updated_tracker.steps_completed == 2

    def test_complete_operation(self, progress_service):
        progress_service.create_tracker("op1", "transcription")
        progress_service.complete_operation("op1")

        completed_tracker = progress_service.get_tracker("op1")
        assert completed_tracker.current_progress == 1.0
        assert completed_tracker.status == "completed"

    def test_eta_calculation(self, progress_service):
        import time

        progress_service.create_tracker("op1", "transcription")
        time.sleep(0.1)  # Small delay
        progress_service.update_progress("op1", 0.5)

        updated_tracker = progress_service.get_tracker("op1")
        eta = updated_tracker.calculate_eta()
        assert eta is not None
        assert eta > 0


class TestCircuitBreakerService:
    @pytest.fixture
    def circuit_breaker(self):
        return WhisperCircuitBreakerService(failure_threshold=3, timeout_duration=1.0)

    def test_closed_circuit_success(self, circuit_breaker):
        def successful_operation():
            return "success"

        result = circuit_breaker.call(successful_operation, "test_service")
        assert result == "success"

        state = circuit_breaker.get_state("test_service")
        assert state.state == "closed"
        assert state.failure_count == 0

    def test_circuit_opens_after_failures(self, circuit_breaker):
        def failing_operation():
            raise Exception("Service error")

        # Trigger failures to open circuit
        for _ in range(3):
            with pytest.raises(Exception, match="Service error"):
                circuit_breaker.call(failing_operation, "test_service")

        state = circuit_breaker.get_state("test_service")
        assert state.state == "open"
        assert state.failure_count == 3

    def test_open_circuit_blocks_calls(self, circuit_breaker):
        def failing_operation():
            raise Exception("Service error")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(Exception, match="Service error"):
                circuit_breaker.call(failing_operation, "test_service")

        # Now calls should be blocked
        def any_operation():
            return "should not execute"

        with pytest.raises(CircuitBreakerError):
            circuit_breaker.call(any_operation, "test_service")

    def test_circuit_reset(self, circuit_breaker):
        def failing_operation():
            raise Exception("Service error")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(Exception, match="Service error"):
                circuit_breaker.call(failing_operation, "test_service")

        # Reset the circuit
        circuit_breaker.reset("test_service")

        state = circuit_breaker.get_state("test_service")
        assert state.state == "closed"
        assert state.failure_count == 0


class TestVocabularyAdapter:
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def vocabulary_adapter(self, temp_dir):
        return VocabularyAdapter(temp_dir)

    @pytest.fixture
    def sample_config(self):
        return VocabularyConfig(
            custom_words=["test", "example"],
            proper_nouns=["Python", "OpenAI"],
            technical_jargon=["API", "JSON"],
        )

    def test_save_and_load_config(self, vocabulary_adapter, sample_config):
        # Save config
        vocabulary_adapter.save_vocabulary_config(sample_config, "test_profile")

        # Load config
        loaded_config = vocabulary_adapter.load_vocabulary_config("test_profile")

        assert loaded_config.custom_words == sample_config.custom_words
        assert loaded_config.proper_nouns == sample_config.proper_nouns
        assert loaded_config.technical_jargon == sample_config.technical_jargon

    def test_import_vocabulary_from_json(self, vocabulary_adapter, temp_dir):
        # Create test JSON file
        test_data = ["word1", "word2", "word3"]
        test_file = Path(temp_dir) / "test_vocab.json"

        with open(test_file, "w") as f:
            json.dump(test_data, f)

        # Import vocabulary
        imported_words = vocabulary_adapter.import_vocabulary_from_file(str(test_file))

        assert imported_words == test_data

    def test_import_vocabulary_from_text(self, vocabulary_adapter, temp_dir):
        # Create test text file
        test_words = ["apple", "banana", "cherry"]
        test_file = Path(temp_dir) / "test_vocab.txt"

        with open(test_file, "w") as f:
            for word in test_words:
                f.write(f"{word}\n")

        # Import vocabulary
        imported_words = vocabulary_adapter.import_vocabulary_from_file(str(test_file))

        assert imported_words == test_words


class TestWebhookEvent:
    def test_webhook_event_creation(self):
        event = WebhookEvent(
            event_type=EventType.TRANSCRIPTION_COMPLETE,
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            operation_id="test_op_123",
            data={"text": "Hello world", "confidence": 0.95},
        )

        assert event.event_type == EventType.TRANSCRIPTION_COMPLETE
        assert event.operation_id == "test_op_123"
        assert event.data["text"] == "Hello world"

    def test_webhook_event_to_dict(self):
        event = WebhookEvent(
            event_type=EventType.TRANSCRIPTION_ERROR,
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            operation_id="test_op_456",
            data={"error": "Model loading failed"},
            session_id="session_123",
        )

        event_dict = event.to_dict()

        assert event_dict["event_type"] == "transcription_error"
        assert event_dict["operation_id"] == "test_op_456"
        assert event_dict["data"]["error"] == "Model loading failed"
        assert event_dict["session_id"] == "session_123"
        assert "timestamp" in event_dict


if __name__ == "__main__":
    pytest.main([__file__])
