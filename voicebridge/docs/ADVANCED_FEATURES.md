# Whisper CLI Advanced Features Guide

This guide covers the advanced features added to the Whisper CLI, providing comprehensive transcription capabilities with enterprise-grade reliability and customization.

## 🎯 Quick Start

```bash
# Install dependencies for advanced features
venv/bin/pip install aiohttp fastapi uvicorn

# Try custom vocabulary
whisper-cli vocabulary add "Kubernetes" --type proper_nouns

# Configure post-processing  
whisper-cli postproc config --punctuation --capitalization

# Add a webhook
whisper-cli webhook add https://your-app.com/webhook --events transcription_complete

# Start API server
whisper-cli api start --port 8000
```

## 📚 Feature Overview

### 1. Custom Vocabulary System
- **Domain-specific terms**: Technical jargon, proper nouns, company names
- **Phonetic mappings**: Handle common mispronunciations
- **Fuzzy matching**: Intelligent word correction
- **Import/Export**: Bulk vocabulary management

### 2. Post-processing Pipeline
- **Punctuation cleanup**: Remove excessive punctuation, fix spacing
- **Capitalization**: Proper sentence and proper noun capitalization
- **Profanity filtering**: Configurable content filtering
- **Filler word removal**: Remove "um", "uh", "like", etc.
- **Text normalization**: Standardize quotes, dashes, and whitespace

### 3. Integration Hooks
- **Webhooks**: Real-time event notifications
- **REST API**: Programmatic access to transcription
- **Event system**: Comprehensive event tracking
- **Authentication**: Secure webhook delivery

### 4. Progress Tracking
- **Real-time progress bars**: Live transcription progress
- **ETA estimates**: Intelligent time remaining calculations
- **Operation management**: Track multiple concurrent operations
- **Progress callbacks**: Custom progress handling

### 5. Retry Logic & Circuit Breaker
- **Exponential backoff**: Intelligent retry strategies
- **Circuit breaker**: Prevent cascade failures
- **Error classification**: Automatic retry decision making
- **Service isolation**: Per-service failure tracking

## 🎛️ Vocabulary Management

### Adding Words

```bash
# Add custom words
whisper-cli vocabulary add "datacenter" --type custom

# Add proper nouns
whisper-cli vocabulary add "OpenAI" --type proper_nouns

# Add technical terms
whisper-cli vocabulary add "Kubernetes" --type technical

# Add domain-specific terms
whisper-cli vocabulary add "API" --type domain --domain "tech"
```

### Managing Vocabulary

```bash
# List all vocabulary
whisper-cli vocabulary list

# List specific type
whisper-cli vocabulary list --type proper_nouns

# Remove words
whisper-cli vocabulary remove "oldterm" --type custom

# Import from file
whisper-cli vocabulary import vocabulary.json --type custom

# Export vocabulary
whisper-cli vocabulary export my_vocabulary.json
```

### Vocabulary File Formats

**JSON Format:**
```json
{
  "custom_words": ["datacenter", "kubernetes"],
  "proper_nouns": ["OpenAI", "Google", "Microsoft"],
  "technical_jargon": ["API", "JSON", "HTTP", "REST"],
  "domain_terms": {
    "tech": ["microservice", "containerization"],
    "finance": ["cryptocurrency", "blockchain"]
  },
  "phonetic_mappings": {
    "koo-ber-net-ees": "kubernetes",
    "oh-pen-ay-eye": "OpenAI"
  }
}
```

**Text Format:**
```
datacenter
kubernetes
containerization
microservice
```

## 🔧 Post-processing Configuration

### Basic Configuration

```bash
# Show current settings
whisper-cli postproc config --show

# Enable punctuation cleanup
whisper-cli postproc config --punctuation

# Enable capitalization
whisper-cli postproc config --capitalization

# Enable profanity filter
whisper-cli postproc config --profanity-filter

# Remove filler words
whisper-cli postproc config --remove-filler
```

### Testing Post-processing

```bash
# Test processing on sample text
whisper-cli postproc test "hello world...this is um really good work"

# Output:
# Original text:
#   hello world...this is um really good work
# Processed text:
#   Hello world... This is really good work.
```

### Custom Profiles

```bash
# Configure different profiles
whisper-cli postproc config --profile meeting --punctuation --capitalization
whisper-cli postproc config --profile casual --remove-filler --no-capitalization

# Use profile in transcription
whisper-cli listen --postproc-profile meeting
```

## 🔗 Integration Hooks

### Webhook Setup

```bash
# Add webhook for transcription events
whisper-cli webhook add https://myapp.com/webhook \
  --events "transcription_complete,transcription_error" \
  --auth "Bearer your-secret-token"

# List configured webhooks
whisper-cli webhook list

# Test webhook
whisper-cli webhook test https://myapp.com/webhook --event transcription_complete

# Remove webhook
whisper-cli webhook remove https://myapp.com/webhook
```

### Webhook Payload Example

```json
{
  "event": {
    "event_type": "transcription_complete",
    "timestamp": "2024-01-01T12:00:00",
    "operation_id": "op_123456",
    "data": {
      "text": "Hello, this is the transcribed text.",
      "confidence": 0.95,
      "duration": 5.2,
      "language": "en"
    },
    "session_id": "session_789"
  },
  "timestamp": "2024-01-01T12:00:00",
  "webhook_version": "1.0"
}
```

### Available Events

- `transcription_start`: Transcription begins
- `transcription_complete`: Transcription finished successfully
- `transcription_error`: Transcription failed
- `progress_update`: Progress information updated
- `model_loaded`: Whisper model loaded successfully

## 🌐 REST API Server

### Starting the API Server

```bash
# Start API server
whisper-cli api start --port 8000 --host 0.0.0.0

# Check server status
whisper-cli api status

# Stop server
whisper-cli api stop
```

### API Endpoints

**POST /transcribe**
```bash
curl -X POST http://localhost:8000/transcribe \
  -H "Content-Type: application/json" \
  -d '{
    "audio_data": "base64_encoded_audio_data",
    "config": {
      "model_name": "medium",
      "language": "en"
    },
    "vocabulary_config": {
      "custom_words": ["kubernetes", "microservice"]
    },
    "post_processing_config": {
      "enable_punctuation_cleanup": true,
      "enable_capitalization": true
    },
    "webhook_url": "https://your-app.com/callback"
  }'
```

**GET /transcribe/{operation_id}**
```bash
curl http://localhost:8000/transcribe/op_123456
```

**GET /progress/{operation_id}**
```bash
curl http://localhost:8000/progress/op_123456
```

## 📊 Progress Tracking

### Monitoring Operations

```bash
# List active operations
whisper-cli operations list

# Check specific operation
whisper-cli operations status op_123456

# Cancel operation
whisper-cli operations cancel op_123456
```

### Progress Bars in CLI

When running transcriptions, you'll see real-time progress:

```
[████████████████████░░░░] 80.0% Processing audio chunks ETA: 2m 15s
```

### Integration with Progress Tracking

```python
from services.progress_service import WhisperProgressService, LiveProgressDisplay

progress_service = WhisperProgressService()
display = LiveProgressDisplay(progress_service)

# Start operation
tracker = progress_service.create_tracker("op1", "transcription", 5)
display.start_display("op1")

# Update progress
progress_service.update_progress("op1", 0.2, "Loading model")
progress_service.update_progress("op1", 0.8, "Processing audio")
progress_service.complete_operation("op1")
```

## 🛡️ Reliability Features

### Retry Configuration

The system automatically retries failed operations using configurable strategies:

```python
from domain.models import RetryConfig, RetryStrategy

config = RetryConfig(
    max_attempts=3,
    initial_delay=1.0,
    max_delay=60.0,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    retryable_errors=["ConnectionError", "TimeoutError", "HTTPError"]
)
```

### Circuit Breaker

```bash
# Check circuit breaker status
whisper-cli circuit status

# Reset circuit breaker
whisper-cli circuit reset --service transcription

# View statistics
whisper-cli circuit stats
```

## 🔌 Advanced Usage Examples

### Custom Transcription Pipeline

```bash
# Full-featured transcription with all advanced features
whisper-cli transcribe audio.wav \
  --model large \
  --language en \
  --vocabulary-profile tech \
  --postproc-profile professional \
  --webhook https://myapp.com/transcription-complete \
  --progress \
  --retry-attempts 3 \
  --session-name "important-meeting"
```

### Batch Processing with API

```python
import asyncio
import aiohttp

async def batch_transcribe(audio_files):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for file_path in audio_files:
            # Read and encode audio file
            with open(file_path, 'rb') as f:
                audio_data = base64.b64encode(f.read()).decode()
            
            # Submit transcription job
            task = session.post(
                'http://localhost:8000/transcribe',
                json={
                    'audio_data': audio_data,
                    'config': {'model_name': 'medium'},
                    'vocabulary_config': {
                        'custom_words': ['kubernetes', 'microservice']
                    }
                }
            )
            tasks.append(task)
        
        # Wait for all jobs to start
        responses = await asyncio.gather(*tasks)
        
        # Poll for completion
        operation_ids = [resp.json()['operation_id'] for resp in responses]
        return await poll_for_completion(session, operation_ids)

async def poll_for_completion(session, operation_ids):
    results = {}
    while operation_ids:
        for op_id in operation_ids[:]:
            resp = await session.get(f'http://localhost:8000/transcribe/{op_id}')
            data = resp.json()
            
            if data['status'] in ['completed', 'error']:
                results[op_id] = data
                operation_ids.remove(op_id)
        
        if operation_ids:
            await asyncio.sleep(1)
    
    return results
```

### Custom Post-processing

```python
from services.post_processing_service import WhisperPostProcessingService
from domain.models import PostProcessingConfig

service = WhisperPostProcessingService()
config = PostProcessingConfig(
    enable_punctuation_cleanup=True,
    enable_capitalization=True,
    custom_replacements={
        "AI": "Artificial Intelligence",
        "ML": "Machine Learning"
    },
    remove_filler_words=True,
    filler_words=["um", "uh", "like", "you know"]
)

# Process transcribed text
raw_text = "so um I think AI and ML are like really important you know"
processed = service.process_text(raw_text, config)
print(processed)
# Output: "So I think Artificial Intelligence and Machine Learning are really important."
```

## 🚀 Performance Optimization

### GPU Acceleration

All advanced features work seamlessly with GPU acceleration:

```bash
# Check GPU status
whisper-cli gpu status

# Benchmark with vocabulary
whisper-cli gpu benchmark --model large --vocabulary-profile tech
```

### Memory Management

Configure memory limits for large-scale processing:

```bash
# Set memory limits
whisper-cli transcribe audio.wav --max-memory 2048 --chunk-size 60
```

### Concurrent Processing

The system supports concurrent operations with progress tracking:

```bash
# Start multiple transcriptions
whisper-cli transcribe audio1.wav --session-name "job1" &
whisper-cli transcribe audio2.wav --session-name "job2" &
whisper-cli transcribe audio3.wav --session-name "job3" &

# Monitor all operations
whisper-cli operations list
```

## 🔍 Troubleshooting

### Common Issues

1. **Webhook delivery failures**
   ```bash
   # Check webhook configuration
   whisper-cli webhook list
   
   # Test webhook connectivity
   whisper-cli webhook test https://your-webhook-url.com
   ```

2. **Circuit breaker in open state**
   ```bash
   # Check status
   whisper-cli circuit status
   
   # Reset if needed
   whisper-cli circuit reset --service transcription
   ```

3. **High memory usage**
   ```bash
   # Reduce chunk size
   whisper-cli transcribe audio.wav --chunk-size 15 --max-memory 1024
   ```

4. **Vocabulary not taking effect**
   ```bash
   # Verify vocabulary loaded
   whisper-cli vocabulary list --profile your-profile
   
   # Test vocabulary enhancement
   whisper-cli postproc test "your test text with custom terms"
   ```

### Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
whisper-cli listen --debug
```

### Performance Monitoring

```bash
# Check performance statistics
whisper-cli performance stats

# Monitor retry statistics
whisper-cli circuit stats
```

## 📄 Configuration Files

Advanced features create configuration files in `~/.config/whisper-cli/`:

```
~/.config/whisper-cli/
├── config.json              # Main configuration
├── profiles/                # Configuration profiles
├── vocabulary/              # Vocabulary profiles
│   ├── default.json
│   └── tech.json
├── postprocessing/          # Post-processing profiles
│   ├── default.json
│   └── professional.json
├── webhooks.json           # Webhook configuration
└── sessions/               # Session data
```

This advanced feature set transforms Whisper CLI from a simple transcription tool into a powerful, enterprise-ready speech-to-text solution with comprehensive customization, reliability, and integration capabilities.