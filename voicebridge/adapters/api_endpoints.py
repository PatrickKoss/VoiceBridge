import uuid
from datetime import datetime
from typing import Any

try:
    from fastapi import BackgroundTasks, FastAPI, HTTPException
    from pydantic import BaseModel

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from voicebridge.domain.models import (
    EventType,
    PostProcessingConfig,
    VocabularyConfig,
    WebhookEvent,
    WhisperConfig,
)


class TranscriptionRequest(BaseModel):
    audio_data: str  # Base64 encoded audio
    config: dict[str, Any] = {}
    vocabulary_config: dict[str, Any] = {}
    post_processing_config: dict[str, Any] = {}
    webhook_url: str | None = None


class TranscriptionResponse(BaseModel):
    operation_id: str
    status: str
    text: str | None = None
    confidence: float | None = None
    duration: float | None = None
    error: str | None = None


class ProgressResponse(BaseModel):
    operation_id: str
    progress: float
    status: str
    current_step: str
    eta_seconds: float | None = None
    start_time: str


class WhisperAPIService:
    def __init__(
        self,
        transcription_service=None,
        vocabulary_service=None,
        post_processing_service=None,
        webhook_service=None,
        progress_service=None,
    ):
        if not FASTAPI_AVAILABLE:
            raise ImportError(
                "FastAPI is required for API endpoints. Install with: pip install fastapi uvicorn"
            )

        self.app = FastAPI(title="Whisper CLI API", version="1.0.0")
        self.transcription_service = transcription_service
        self.vocabulary_service = vocabulary_service
        self.post_processing_service = post_processing_service
        self.webhook_service = webhook_service
        self.progress_service = progress_service

        self._active_operations: dict[str, dict[str, Any]] = {}

        self._setup_routes()

    def _setup_routes(self):
        @self.app.post("/transcribe", response_model=TranscriptionResponse)
        async def transcribe_audio(
            request: TranscriptionRequest, background_tasks: BackgroundTasks
        ):
            operation_id = str(uuid.uuid4())

            try:
                # Decode audio data
                import base64

                audio_data = base64.b64decode(request.audio_data)

                # Parse configurations
                whisper_config = (
                    WhisperConfig(**request.config)
                    if request.config
                    else WhisperConfig()
                )
                vocab_config = (
                    VocabularyConfig(**request.vocabulary_config)
                    if request.vocabulary_config
                    else None
                )
                post_proc_config = (
                    PostProcessingConfig(**request.post_processing_config)
                    if request.post_processing_config
                    else None
                )

                # Start transcription in background
                background_tasks.add_task(
                    self._process_transcription,
                    operation_id,
                    audio_data,
                    whisper_config,
                    vocab_config,
                    post_proc_config,
                    request.webhook_url,
                )

                # Create progress tracker
                if self.progress_service:
                    self.progress_service.create_tracker(
                        operation_id, "transcription", 4
                    )

                return TranscriptionResponse(
                    operation_id=operation_id, status="processing"
                )

            except Exception as e:
                return TranscriptionResponse(
                    operation_id=operation_id, status="error", error=str(e)
                )

        @self.app.get(
            "/transcribe/{operation_id}", response_model=TranscriptionResponse
        )
        async def get_transcription_status(operation_id: str):
            if operation_id not in self._active_operations:
                raise HTTPException(status_code=404, detail="Operation not found")

            operation = self._active_operations[operation_id]
            return TranscriptionResponse(**operation)

        @self.app.get("/progress/{operation_id}", response_model=ProgressResponse)
        async def get_progress(operation_id: str):
            if not self.progress_service:
                raise HTTPException(
                    status_code=501, detail="Progress tracking not available"
                )

            tracker = self.progress_service.get_tracker(operation_id)
            if not tracker:
                raise HTTPException(status_code=404, detail="Operation not found")

            return ProgressResponse(
                operation_id=operation_id,
                progress=tracker.current_progress,
                status=tracker.status,
                current_step=tracker.current_step,
                eta_seconds=tracker.calculate_eta(),
                start_time=tracker.start_time.isoformat(),
            )

        @self.app.post("/webhook/register")
        async def register_webhook(webhook_url: str, event_types: list[str]):
            if not self.webhook_service:
                raise HTTPException(
                    status_code=501, detail="Webhook service not available"
                )

            try:
                events = [EventType(event_type) for event_type in event_types]
                self.webhook_service.register_webhook(webhook_url, events)
                return {"message": "Webhook registered successfully"}
            except ValueError as e:
                raise HTTPException(
                    status_code=400, detail=f"Invalid event type: {e}"
                ) from e

        @self.app.get("/operations")
        async def list_operations():
            return {
                "active_operations": len(self._active_operations),
                "operations": [
                    {
                        "operation_id": op_id,
                        "status": op_data.get("status"),
                        "start_time": op_data.get("start_time"),
                    }
                    for op_id, op_data in self._active_operations.items()
                ],
            }

        @self.app.delete("/operations/{operation_id}")
        async def cancel_operation(operation_id: str):
            if operation_id in self._active_operations:
                self._active_operations[operation_id]["status"] = "cancelled"
                return {"message": "Operation cancelled"}
            else:
                raise HTTPException(status_code=404, detail="Operation not found")

        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "services": {
                    "transcription": self.transcription_service is not None,
                    "vocabulary": self.vocabulary_service is not None,
                    "post_processing": self.post_processing_service is not None,
                    "webhook": self.webhook_service is not None,
                    "progress": self.progress_service is not None,
                },
            }

    async def _process_transcription(
        self,
        operation_id: str,
        audio_data: bytes,
        whisper_config: WhisperConfig,
        vocab_config: VocabularyConfig = None,
        post_proc_config: PostProcessingConfig = None,
        webhook_url: str = None,
    ):
        # Initialize operation
        self._active_operations[operation_id] = {
            "operation_id": operation_id,
            "status": "processing",
            "start_time": datetime.now().isoformat(),
        }

        try:
            # Step 1: Transcription
            if self.progress_service:
                self.progress_service.update_progress(
                    operation_id, 0.1, "Starting transcription"
                )

            if self.webhook_service:
                event = self._build_webhook_event(
                    EventType.TRANSCRIPTION_START, operation_id
                )
                self.webhook_service.trigger_event(event)

            if not self.transcription_service:
                raise Exception("Transcription service not available")

            result = self.transcription_service.transcribe(audio_data, whisper_config)
            text = result.text

            # Step 2: Vocabulary enhancement
            if vocab_config and self.vocabulary_service:
                if self.progress_service:
                    self.progress_service.update_progress(
                        operation_id, 0.5, "Enhancing with vocabulary"
                    )

                text = self.vocabulary_service.enhance_transcription(text, vocab_config)

            # Step 3: Post-processing
            if post_proc_config and self.post_processing_service:
                if self.progress_service:
                    self.progress_service.update_progress(
                        operation_id, 0.8, "Post-processing text"
                    )

                text = self.post_processing_service.process_text(text, post_proc_config)

            # Step 4: Complete
            if self.progress_service:
                self.progress_service.update_progress(operation_id, 1.0, "Complete")
                self.progress_service.complete_operation(operation_id)

            # Update operation status
            self._active_operations[operation_id].update(
                {
                    "status": "completed",
                    "text": text,
                    "confidence": result.confidence,
                    "duration": result.duration,
                    "completed_time": datetime.now().isoformat(),
                }
            )

            # Send completion webhook
            if self.webhook_service:
                event = self._build_webhook_event(
                    EventType.TRANSCRIPTION_COMPLETE,
                    operation_id,
                    text=text,
                    confidence=result.confidence,
                    duration=result.duration,
                )
                self.webhook_service.trigger_event(event)

        except Exception as e:
            # Update operation with error
            self._active_operations[operation_id].update(
                {
                    "status": "error",
                    "error": str(e),
                    "error_time": datetime.now().isoformat(),
                }
            )

            # Send error webhook
            if self.webhook_service:
                event = self._build_webhook_event(
                    EventType.TRANSCRIPTION_ERROR, operation_id, error=str(e)
                )
                self.webhook_service.trigger_event(event)

    def _build_webhook_event(
        self, event_type: EventType, operation_id: str, **data
    ) -> WebhookEvent:
        return WebhookEvent(
            event_type=event_type,
            timestamp=datetime.now(),
            operation_id=operation_id,
            data=data,
        )

    def get_app(self) -> FastAPI:
        return self.app


def create_api_app(**services) -> FastAPI:
    api_service = WhisperAPIService(**services)
    return api_service.get_app()
