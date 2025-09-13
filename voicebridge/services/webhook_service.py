import asyncio
from datetime import datetime

import aiohttp

from voicebridge.domain.models import EventType, IntegrationConfig, WebhookEvent
from voicebridge.ports.interfaces import WebhookService


class WhisperWebhookService(WebhookService):
    def __init__(self, config: IntegrationConfig = None):
        self.config = config or IntegrationConfig()
        self._registered_webhooks: dict[str, list[EventType]] = {}
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._worker_task = None

    async def start_service(self) -> None:
        if not self._running:
            self._running = True
            self._worker_task = asyncio.create_task(self._webhook_worker())

    async def stop_service(self) -> None:
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    def send_webhook(self, event: WebhookEvent, url: str) -> bool:
        try:
            # Add to async queue for processing
            asyncio.create_task(self._send_webhook_async(event, url))
            return True
        except Exception:
            return False

    async def _send_webhook_async(self, event: WebhookEvent, url: str) -> bool:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "WhisperCLI-Webhook/1.0",
        }

        # Add authentication headers if configured
        if self.config.authentication:
            headers.update(self.config.authentication)

        payload = {
            "event": event.to_dict(),
            "timestamp": event.timestamp.isoformat(),
            "webhook_version": "1.0",
        }

        try:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    return response.status < 400
        except Exception:
            return False

    def register_webhook(self, url: str, event_types: list[EventType]) -> None:
        self._registered_webhooks[url] = event_types
        # Also add to config for persistence
        if url not in self.config.webhook_urls:
            self.config.webhook_urls.append(url)

    def trigger_event(self, event: WebhookEvent) -> None:
        if self.config.enable_async_delivery:
            # Add to queue for async processing
            try:
                self._event_queue.put_nowait(event)
            except asyncio.QueueFull:
                pass  # Drop event if queue is full
        else:
            # Process synchronously
            self._process_event_sync(event)

    def _process_event_sync(self, event: WebhookEvent) -> None:
        for url in self.config.webhook_urls:
            event_types = self._registered_webhooks.get(url, self.config.event_types)
            if event.event_type in event_types:
                self.send_webhook(event, url)

    async def _webhook_worker(self) -> None:
        while self._running:
            try:
                # Get event from queue with timeout
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                await self._process_event_async(event)
                self._event_queue.task_done()
            except TimeoutError:
                continue
            except Exception:
                continue

    async def _process_event_async(self, event: WebhookEvent) -> None:
        tasks = []

        for url in self.config.webhook_urls:
            event_types = self._registered_webhooks.get(url, self.config.event_types)
            if event.event_type in event_types:
                task = self._send_webhook_with_retry(event, url)
                tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_webhook_with_retry(self, event: WebhookEvent, url: str) -> bool:
        for attempt in range(self.config.retry_attempts + 1):
            success = await self._send_webhook_async(event, url)
            if success:
                return True

            if attempt < self.config.retry_attempts:
                # Exponential backoff
                delay = min(2**attempt, 60)  # Cap at 60 seconds
                await asyncio.sleep(delay)

        return False


class WebhookEventBuilder:
    @staticmethod
    def build_transcription_start(
        operation_id: str, session_id: str = None, **kwargs
    ) -> WebhookEvent:
        return WebhookEvent(
            event_type=EventType.TRANSCRIPTION_START,
            timestamp=datetime.now(),
            operation_id=operation_id,
            session_id=session_id,
            data=kwargs,
        )

    @staticmethod
    def build_transcription_complete(
        operation_id: str,
        text: str,
        duration: float = None,
        session_id: str = None,
        **kwargs,
    ) -> WebhookEvent:
        data = {"text": text, **kwargs}
        if duration is not None:
            data["duration"] = duration

        return WebhookEvent(
            event_type=EventType.TRANSCRIPTION_COMPLETE,
            timestamp=datetime.now(),
            operation_id=operation_id,
            session_id=session_id,
            data=data,
        )

    @staticmethod
    def build_transcription_error(
        operation_id: str, error: str, session_id: str = None, **kwargs
    ) -> WebhookEvent:
        return WebhookEvent(
            event_type=EventType.TRANSCRIPTION_ERROR,
            timestamp=datetime.now(),
            operation_id=operation_id,
            session_id=session_id,
            data={"error": error, **kwargs},
        )

    @staticmethod
    def build_progress_update(
        operation_id: str,
        progress: float,
        current_step: str = "",
        eta_seconds: float = None,
        session_id: str = None,
        **kwargs,
    ) -> WebhookEvent:
        data = {"progress": progress, "current_step": current_step, **kwargs}
        if eta_seconds is not None:
            data["eta_seconds"] = eta_seconds

        return WebhookEvent(
            event_type=EventType.PROGRESS_UPDATE,
            timestamp=datetime.now(),
            operation_id=operation_id,
            session_id=session_id,
            data=data,
        )

    @staticmethod
    def build_model_loaded(
        operation_id: str,
        model_name: str,
        load_time: float = None,
        session_id: str = None,
        **kwargs,
    ) -> WebhookEvent:
        data = {"model_name": model_name, **kwargs}
        if load_time is not None:
            data["load_time"] = load_time

        return WebhookEvent(
            event_type=EventType.MODEL_LOADED,
            timestamp=datetime.now(),
            operation_id=operation_id,
            session_id=session_id,
            data=data,
        )
