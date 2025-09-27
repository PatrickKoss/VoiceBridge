"""Tests for webhook service."""

import asyncio
import importlib.util
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from voicebridge.domain.models import EventType, IntegrationConfig, WebhookEvent

HAS_AIOHTTP = importlib.util.find_spec("aiohttp") is not None

# Only import webhook service if aiohttp is available
if HAS_AIOHTTP:
    from voicebridge.services.webhook_service import (
        WebhookEventBuilder,
        WhisperWebhookService,
    )


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
class TestWhisperWebhookService:
    """Test cases for WhisperWebhookService."""

    @pytest.fixture
    def config(self):
        """Sample IntegrationConfig."""
        return IntegrationConfig(
            webhook_urls=["https://example.com/webhook"],
            event_types=[EventType.TRANSCRIPTION_COMPLETE],
            timeout_seconds=10,
            retry_attempts=2,
            enable_async_delivery=True,
        )

    @pytest.fixture
    def service(self, config):
        """WhisperWebhookService instance."""
        return WhisperWebhookService(config)

    @pytest.fixture
    def service_no_config(self):
        """WhisperWebhookService instance without config."""
        return WhisperWebhookService()

    @pytest.fixture
    def sample_event(self):
        """Sample WebhookEvent."""
        return WebhookEvent(
            event_type=EventType.TRANSCRIPTION_COMPLETE,
            timestamp=datetime(2023, 1, 1, 12, 0, 0),
            operation_id="test-op",
            session_id="test-session",
            data={"text": "Hello world"},
        )

    def test_init_with_config(self, service, config):
        """Test service initialization with config."""
        assert service.config == config
        assert service._registered_webhooks == {}
        assert service._running is False
        assert service._worker_task is None

    def test_init_without_config(self, service_no_config):
        """Test service initialization without config."""
        assert service_no_config.config is not None
        assert isinstance(service_no_config.config, IntegrationConfig)

    @pytest.mark.asyncio
    async def test_start_service(self, service):
        """Test starting the webhook service."""
        await service.start_service()

        assert service._running is True
        assert service._worker_task is not None

        # Clean up
        await service.stop_service()

    @pytest.mark.asyncio
    async def test_start_service_idempotent(self, service):
        """Test that starting service multiple times is safe."""
        await service.start_service()
        first_task = service._worker_task

        await service.start_service()
        # Should not create new task if already running
        assert service._worker_task == first_task

        # Clean up
        await service.stop_service()

    @pytest.mark.asyncio
    async def test_stop_service(self, service):
        """Test stopping the webhook service."""
        await service.start_service()
        assert service._running is True

        await service.stop_service()
        assert service._running is False

    @pytest.mark.asyncio
    async def test_stop_service_not_running(self, service):
        """Test stopping service when not running."""
        # Should not raise error
        await service.stop_service()
        assert service._running is False

    def test_register_webhook(self, service):
        """Test registering a webhook."""
        url = "https://test.com/webhook"
        event_types = [EventType.TRANSCRIPTION_START, EventType.TRANSCRIPTION_COMPLETE]

        service.register_webhook(url, event_types)

        assert service._registered_webhooks[url] == event_types
        assert url in service.config.webhook_urls

    def test_register_webhook_existing_url(self, service):
        """Test registering webhook with existing URL in config."""
        url = "https://example.com/webhook"  # Already in config
        event_types = [EventType.TRANSCRIPTION_START]

        initial_count = len(service.config.webhook_urls)
        service.register_webhook(url, event_types)

        # Should not duplicate URL
        assert len(service.config.webhook_urls) == initial_count
        assert service._registered_webhooks[url] == event_types

    @patch("asyncio.create_task")
    def test_send_webhook_success(self, mock_create_task, service, sample_event):
        """Test successful webhook sending."""
        mock_create_task.return_value = Mock()

        result = service.send_webhook(sample_event, "https://test.com/webhook")

        assert result is True
        mock_create_task.assert_called_once()

    @patch("asyncio.create_task")
    def test_send_webhook_exception(self, mock_create_task, service, sample_event):
        """Test webhook sending with exception."""
        mock_create_task.side_effect = Exception("Task creation failed")

        result = service.send_webhook(sample_event, "https://test.com/webhook")

        assert result is False

    @pytest.mark.asyncio
    async def test_send_webhook_async_success(self, service, sample_event):
        """Test successful async webhook sending."""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)

            mock_post = Mock()
            mock_post.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.__aexit__ = AsyncMock(return_value=None)

            mock_session_instance = Mock()
            mock_session_instance.post.return_value = mock_post
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)

            mock_session.return_value = mock_session_instance

            result = await service._send_webhook_async(sample_event, "https://test.com")

            assert result is True

    @pytest.mark.asyncio
    async def test_send_webhook_async_error_status(self, service, sample_event):
        """Test async webhook sending with error status."""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = Mock()
            mock_response.status = 500
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)

            mock_post = Mock()
            mock_post.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.__aexit__ = AsyncMock(return_value=None)

            mock_session_instance = Mock()
            mock_session_instance.post.return_value = mock_post
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)

            mock_session.return_value = mock_session_instance

            result = await service._send_webhook_async(sample_event, "https://test.com")

            assert result is False

    @pytest.mark.asyncio
    async def test_send_webhook_async_exception(self, service, sample_event):
        """Test async webhook sending with exception."""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.side_effect = Exception("Network error")

            result = await service._send_webhook_async(sample_event, "https://test.com")

            assert result is False

    @pytest.mark.asyncio
    async def test_send_webhook_async_with_auth(self, service, sample_event):
        """Test async webhook sending with authentication."""
        service.config.authentication = {"Authorization": "Bearer token123"}

        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)

            mock_post = Mock()
            mock_post.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.__aexit__ = AsyncMock(return_value=None)

            mock_session_instance = Mock()
            mock_session_instance.post.return_value = mock_post
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)

            mock_session.return_value = mock_session_instance

            await service._send_webhook_async(sample_event, "https://test.com")

            # Check that authentication header was included
            call_args = mock_session_instance.post.call_args
            headers = call_args[1]["headers"]
            assert headers["Authorization"] == "Bearer token123"

    def test_trigger_event_async_enabled(self, service, sample_event):
        """Test triggering event with async delivery enabled."""
        service.config.enable_async_delivery = True

        service.trigger_event(sample_event)

        # Event should be in queue
        assert not service._event_queue.empty()

    def test_trigger_event_async_disabled(self, service, sample_event):
        """Test triggering event with async delivery disabled."""
        service.config.enable_async_delivery = False

        with patch.object(service, "_process_event_sync") as mock_process:
            service.trigger_event(sample_event)
            mock_process.assert_called_once_with(sample_event)

    def test_trigger_event_queue_full(self, service, sample_event):
        """Test triggering event when queue is full."""
        service.config.enable_async_delivery = True

        # Fill the queue to max capacity (simulate full queue)
        with patch.object(
            service._event_queue, "put_nowait", side_effect=asyncio.QueueFull
        ):
            # Should not raise exception
            service.trigger_event(sample_event)

    def test_process_event_sync(self, service_no_config, sample_event):
        """Test synchronous event processing."""
        # Register webhook for this event type
        service_no_config.register_webhook(
            "https://test.com", [EventType.TRANSCRIPTION_COMPLETE]
        )

        with patch.object(service_no_config, "send_webhook") as mock_send:
            service_no_config._process_event_sync(sample_event)
            mock_send.assert_called_once_with(sample_event, "https://test.com")

    def test_process_event_sync_wrong_event_type(self, service_no_config, sample_event):
        """Test sync processing with wrong event type."""
        # Register webhook for different event type
        service_no_config.register_webhook(
            "https://test.com", [EventType.TRANSCRIPTION_START]
        )

        with patch.object(service_no_config, "send_webhook") as mock_send:
            service_no_config._process_event_sync(sample_event)
            mock_send.assert_not_called()

    def test_process_event_sync_uses_config_event_types(self, service, sample_event):
        """Test sync processing uses config event types when no registration."""
        # Don't register webhook, should use config event types
        with patch.object(service, "send_webhook") as mock_send:
            service._process_event_sync(sample_event)
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_event_async(self, service_no_config, sample_event):
        """Test asynchronous event processing."""
        service_no_config.register_webhook(
            "https://test.com", [EventType.TRANSCRIPTION_COMPLETE]
        )

        with patch.object(service_no_config, "_send_webhook_with_retry") as mock_send:
            mock_send.return_value = True
            await service_no_config._process_event_async(sample_event)
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_webhook_with_retry_success_first_attempt(
        self, service, sample_event
    ):
        """Test webhook retry succeeds on first attempt."""
        with patch.object(service, "_send_webhook_async", return_value=True):
            result = await service._send_webhook_with_retry(
                sample_event, "https://test.com"
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_send_webhook_with_retry_success_after_retries(
        self, service, sample_event
    ):
        """Test webhook retry succeeds after some failures."""
        with patch.object(
            service, "_send_webhook_async", side_effect=[False, False, True]
        ):
            with patch("asyncio.sleep"):  # Mock sleep to speed up test
                result = await service._send_webhook_with_retry(
                    sample_event, "https://test.com"
                )
                assert result is True

    @pytest.mark.asyncio
    async def test_send_webhook_with_retry_all_attempts_fail(
        self, service, sample_event
    ):
        """Test webhook retry when all attempts fail."""
        with patch.object(service, "_send_webhook_async", return_value=False):
            with patch("asyncio.sleep"):  # Mock sleep to speed up test
                result = await service._send_webhook_with_retry(
                    sample_event, "https://test.com"
                )
                assert result is False

    @pytest.mark.asyncio
    async def test_webhook_worker_processes_events(self, service, sample_event):
        """Test webhook worker processes events from queue."""
        service._running = True
        service._event_queue.put_nowait(sample_event)

        with patch.object(service, "_process_event_async") as mock_process:
            # Run worker briefly
            worker_task = asyncio.create_task(service._webhook_worker())
            await asyncio.sleep(0.1)  # Give it time to process
            service._running = False

            try:
                await asyncio.wait_for(worker_task, timeout=1.0)
            except TimeoutError:
                worker_task.cancel()

            mock_process.assert_called_once_with(sample_event)

    @pytest.mark.asyncio
    async def test_webhook_worker_handles_timeout(self, service):
        """Test webhook worker handles timeout gracefully."""
        service._running = True

        # Worker should handle empty queue timeout gracefully
        worker_task = asyncio.create_task(service._webhook_worker())
        await asyncio.sleep(0.1)  # Brief run
        service._running = False

        try:
            await asyncio.wait_for(worker_task, timeout=1.0)
        except TimeoutError:
            worker_task.cancel()

    @pytest.mark.asyncio
    async def test_webhook_worker_handles_exception(self, service, sample_event):
        """Test webhook worker handles processing exceptions."""
        service._running = True
        service._event_queue.put_nowait(sample_event)

        with patch.object(
            service, "_process_event_async", side_effect=Exception("Processing error")
        ):
            worker_task = asyncio.create_task(service._webhook_worker())
            await asyncio.sleep(0.1)
            service._running = False

            try:
                await asyncio.wait_for(worker_task, timeout=1.0)
            except TimeoutError:
                worker_task.cancel()


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
class TestWebhookEventBuilder:
    """Test cases for WebhookEventBuilder."""

    def test_build_transcription_start(self):
        """Test building transcription start event."""
        event = WebhookEventBuilder.build_transcription_start(
            operation_id="test-op",
            session_id="test-session",
            model="base",
        )

        assert event.event_type == EventType.TRANSCRIPTION_START
        assert event.operation_id == "test-op"
        assert event.session_id == "test-session"
        assert event.data["model"] == "base"
        assert isinstance(event.timestamp, datetime)

    def test_build_transcription_start_minimal(self):
        """Test building transcription start event with minimal data."""
        event = WebhookEventBuilder.build_transcription_start(operation_id="test-op")

        assert event.event_type == EventType.TRANSCRIPTION_START
        assert event.operation_id == "test-op"
        assert event.session_id is None
        assert event.data == {}

    def test_build_transcription_complete(self):
        """Test building transcription complete event."""
        event = WebhookEventBuilder.build_transcription_complete(
            operation_id="test-op",
            text="Hello world",
            duration=5.5,
            session_id="test-session",
            confidence=0.95,
        )

        assert event.event_type == EventType.TRANSCRIPTION_COMPLETE
        assert event.operation_id == "test-op"
        assert event.session_id == "test-session"
        assert event.data["text"] == "Hello world"
        assert event.data["duration"] == 5.5
        assert event.data["confidence"] == 0.95

    def test_build_transcription_complete_minimal(self):
        """Test building transcription complete event with minimal data."""
        event = WebhookEventBuilder.build_transcription_complete(
            operation_id="test-op", text="Hello world"
        )

        assert event.event_type == EventType.TRANSCRIPTION_COMPLETE
        assert event.data["text"] == "Hello world"
        assert "duration" not in event.data

    def test_build_transcription_error(self):
        """Test building transcription error event."""
        event = WebhookEventBuilder.build_transcription_error(
            operation_id="test-op",
            error="Audio file not found",
            session_id="test-session",
            file_path="/path/to/file.wav",
        )

        assert event.event_type == EventType.TRANSCRIPTION_ERROR
        assert event.operation_id == "test-op"
        assert event.session_id == "test-session"
        assert event.data["error"] == "Audio file not found"
        assert event.data["file_path"] == "/path/to/file.wav"

    def test_build_progress_update(self):
        """Test building progress update event."""
        event = WebhookEventBuilder.build_progress_update(
            operation_id="test-op",
            progress=0.65,
            current_step="Processing audio",
            eta_seconds=120.0,
            session_id="test-session",
        )

        assert event.event_type == EventType.PROGRESS_UPDATE
        assert event.operation_id == "test-op"
        assert event.session_id == "test-session"
        assert event.data["progress"] == 0.65
        assert event.data["current_step"] == "Processing audio"
        assert event.data["eta_seconds"] == 120.0

    def test_build_progress_update_minimal(self):
        """Test building progress update event with minimal data."""
        event = WebhookEventBuilder.build_progress_update(
            operation_id="test-op", progress=0.5
        )

        assert event.data["progress"] == 0.5
        assert event.data["current_step"] == ""
        assert "eta_seconds" not in event.data

    def test_build_model_loaded(self):
        """Test building model loaded event."""
        event = WebhookEventBuilder.build_model_loaded(
            operation_id="test-op",
            model_name="whisper-base",
            load_time=2.5,
            session_id="test-session",
            device="cuda",
        )

        assert event.event_type == EventType.MODEL_LOADED
        assert event.operation_id == "test-op"
        assert event.session_id == "test-session"
        assert event.data["model_name"] == "whisper-base"
        assert event.data["load_time"] == 2.5
        assert event.data["device"] == "cuda"

    def test_build_model_loaded_minimal(self):
        """Test building model loaded event with minimal data."""
        event = WebhookEventBuilder.build_model_loaded(
            operation_id="test-op", model_name="whisper-base"
        )

        assert event.data["model_name"] == "whisper-base"
        assert "load_time" not in event.data
