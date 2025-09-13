# Export & Analysis Features Guide

This guide covers the new export and analysis features added to whisper-cli, including multiple output formats, timestamp processing, and confidence analysis.

## 🎯 Overview

The whisper-cli now supports comprehensive export capabilities with:

- **5 Export Formats**: JSON, Plain Text, CSV, SRT subtitles, WebVTT
- **3 Timestamp Modes**: Word-level, sentence-level, paragraph-level  
- **Confidence Analysis**: Quality assessment and review recommendations
- **Speaker Support**: Basic speaker identification and metadata
- **Translation Ready**: Architecture supports multi-language output

## 📁 Export Formats

### JSON Format
Complete metadata export with full transcription data:
```json
{
  "text": "Full transcription text...",
  "language": "en",
  "duration": 120.5,
  "confidence": 0.89,
  "segments": [
    {
      "text": "Hello world",
      "start_time": 0.0,
      "end_time": 2.5,
      "confidence": 0.95,
      "speaker_id": 1
    }
  ]
}
```

### SRT Subtitles
Standard subtitle format for video:
```srt
1
00:00:00,000 --> 00:00:02,500
Speaker 1: Hello world

2
00:00:02,500 --> 00:00:05,000
Speaker 2: How are you?
```

### WebVTT Format
Web-compatible subtitles with speaker support:
```vtt
WEBVTT

00:00:00.000 --> 00:00:02.500
<v Speaker 1>Hello world

00:00:02.500 --> 00:00:05.000
<v Speaker 2>How are you?
```

### CSV Format
Spreadsheet-compatible data:
```csv
start_time,end_time,text,speaker_id,confidence
0.0,2.5,"Hello world",1,0.95
2.5,5.0,"How are you?",2,0.88
```

### Plain Text
Human-readable format with timestamps:
```
Language: en
Overall Confidence: 89.00%
Duration: 120.50s

[00:00:00] Speaker 1: Hello world (confidence: 95.00%)
[00:00:02] Speaker 2: How are you? (confidence: 88.00%)
```

## ⏱️ Timestamp Modes

### Word-Level
Preserves original word-by-word timing from Whisper:
- Most precise timing information
- Useful for detailed analysis
- Large number of segments

### Sentence-Level (Default)
Groups words into sentences based on punctuation and pauses:
- Balanced between precision and readability
- Natural conversation flow
- Recommended for most use cases

### Paragraph-Level  
Groups sentences into paragraphs based on longer pauses:
- Best readability for long content
- Ideal for presentations or lectures
- Fewer, longer segments

## 🎯 Confidence Analysis

The confidence analyzer provides quality assessment with:

### Confidence Levels
- **HIGH** (90%+): Excellent quality, minimal review needed
- **MEDIUM** (70-89%): Good quality, spot-check recommended  
- **LOW** (50-69%): Review flagged segments
- **VERY LOW** (<50%): Manual review required

### Quality Flags
The system automatically identifies:
- **Low Confidence Segments**: Below threshold scores
- **Short Segments**: Very brief segments (possible noise)
- **Long Segments**: Unusually long segments (possible multiple speakers)
- **Repeated Words**: Potential stuttering or processing issues
- **Incomplete Sentences**: Missing punctuation

### Review Recommendations
Automated suggestions based on analysis:
- Audio quality improvements
- Recording environment tips
- Manual review priorities
- Processing adjustments

## 💬 CLI Commands

### Export Commands

List supported formats:
```bash
whisper-cli export formats
```

Export a specific session:
```bash
whisper-cli export session SESSION_ID --format json --output result.json
```

Export with options:
```bash
whisper-cli export session SESSION_ID \
  --format srt \
  --timestamps sentence \
  --confidence \
  --speakers \
  --output subtitles.srt
```

Batch export all sessions:
```bash
whisper-cli export batch \
  --format csv \
  --output-dir ./exports \
  --timestamps paragraph
```

### Confidence Analysis Commands

Analyze a specific session:
```bash
whisper-cli confidence analyze SESSION_ID --detailed
```

Analyze all sessions:
```bash
whisper-cli confidence analyze-all --threshold 0.8
```

Configure confidence thresholds:
```bash
whisper-cli confidence configure \
  --high 0.9 \
  --medium 0.7 \
  --low 0.5 \
  --review 0.6
```

### Enhanced Transcription

Transcribe with export options:
```bash
whisper-cli transcribe audio.wav \
  --export-format srt \
  --timestamp-mode sentence \
  --confidence
```

## 🔧 Configuration

### Export Settings in Config
Add to your WhisperConfig:
```python
export_format = OutputFormat.SRT
timestamp_mode = TimestampMode.SENTENCE_LEVEL
include_confidence_scores = True
enable_speaker_detection = False  # Future feature
```

### Confidence Thresholds
Adjust thresholds based on your quality requirements:
- **Strict**: High=0.95, Medium=0.85, Low=0.70
- **Balanced** (default): High=0.90, Medium=0.70, Low=0.50  
- **Lenient**: High=0.85, Medium=0.60, Low=0.40

## 🎬 Example Workflows

### Subtitle Creation
```bash
# 1. Transcribe with good timing
whisper-cli transcribe video_audio.wav --timestamp-mode sentence

# 2. Export as SRT with speaker info
whisper-cli export session SESSION_ID --format srt --speakers

# 3. Review confidence if needed
whisper-cli confidence analyze SESSION_ID
```

### Data Analysis
```bash
# 1. Transcribe content
whisper-cli transcribe meeting.wav

# 2. Export as CSV for analysis
whisper-cli export session SESSION_ID --format csv --confidence

# 3. Batch analyze all meetings
whisper-cli confidence analyze-all --detailed
```

### Quality Control
```bash
# 1. Transcribe with confidence tracking
whisper-cli transcribe interview.wav --confidence

# 2. Get quality assessment
whisper-cli confidence analyze SESSION_ID --detailed

# 3. Export flagged segments for review
whisper-cli export session SESSION_ID --format json --confidence
```

## 🚀 Advanced Features

### Custom Export Pipeline
The modular architecture supports:
- Custom format implementations
- Translation services integration  
- Advanced speaker diarization
- Real-time export streaming

### Extensibility
Easy to add new features:
- **New Export Formats**: Implement `ExportService` interface
- **Custom Analysis**: Extend `ConfidenceAnalyzer`
- **Translation Services**: Implement `TranslationService`
- **Speaker Services**: Implement `SpeakerDiarizationService`

## 🐛 Troubleshooting

### Common Issues

**"Export service not available"**
- Ensure services are properly initialized in `main.py`
- Check that export commands have required dependencies

**Low confidence scores**
- Improve audio quality (reduce background noise)
- Use higher-quality microphone
- Ensure clear speech patterns

**Speaker detection not working**
- Currently uses mock implementation
- Real speaker diarization requires additional setup
- Check timing patterns and pause detection

### Performance Tips
- Use sentence-level timestamps for balanced performance
- JSON format provides complete data but larger files
- CSV format best for spreadsheet analysis
- SRT/VTT formats optimized for video use

## 🔮 Future Enhancements

Planned features:
- **Real Translation**: Google Translate / LibreTranslate integration
- **Advanced Speaker Diarization**: pyannote.audio integration
- **Custom Vocabulary**: Domain-specific term recognition
- **Batch Processing**: Multi-file transcription and export
- **API Integration**: REST endpoints for export services

---

For more information, run `whisper-cli --help` or check individual command help with `--help` flag.