# Memotag - Cognitive Decline Analysis System

A sophisticated web-based application for analyzing speech patterns to detect potential cognitive decline indicators through audio analysis, speech processing, and sentiment analysis.

## Features

### 1. Audio Processing
- **Multiple Input Methods**
  - Direct audio recording through browser
  - File upload support (WAV, MP3, etc.)
  - Real-time processing capabilities

- **Speech Recognition**
  - Accurate speech-to-text transcription
  - Support for clear audio processing
  - Error handling for unclear speech

### 2. Cognitive Analysis
- **Speech Metrics**
  - Speech rate calculation (words per minute)
  - Pause analysis and patterns
  - Voice quality assessment
  - Pitch variation analysis

- **Risk Assessment**
  - Cognitive decline risk scoring
  - Risk level categorization (Low/Medium/High)
  - Detailed interpretation of results

### 3. Sentiment Analysis
- **Comprehensive Analysis**
  - Overall sentiment scoring
  - Sentence-by-sentence breakdown
  - Confidence metrics
  - Emotional pattern detection

### 4. Visualizations
- **Audio Analysis**
  - Waveform display
  - Mel spectrogram analysis
  - Pitch contour visualization
  
- **Results Display**
  - Sentiment trend graphs
  - Risk assessment visualizations
  - Interactive data presentation

## Installation

1. **Clone the Repository**
   ```bash
   git clone [repository-url]
   cd memotag
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **FFmpeg Setup**
   - FFmpeg is included in the project directory
   - No additional installation required
   - Located in `/ffmpeg/ffmpeg-6.1.1-essentials_build/`

## Configuration

1. **Environment Setup**
   - Ensure Python 3.8+ is installed
   - Verify FFmpeg path in configuration
   - Check microphone permissions for recording

2. **API Keys (if needed)**
   - Configure Google Speech Recognition credentials
   - Set up any additional API keys

## Usage

1. **Start the Application**
   ```bash
   python api.py
   ```

2. **Access the Interface**
   - Open web browser
   - Navigate to `http://localhost:5000`
   - Interface will be ready for use

3. **Recording Audio**
   - Click "Start Recording" button
   - Speak clearly into microphone
   - Click "Stop Recording" when finished

4. **Uploading Audio**
   - Click "Choose File"
   - Select audio file (WAV, MP3)
   - Click "Analyze"

5. **Viewing Results**
   - Transcription display
   - Cognitive analysis metrics
   - Sentiment analysis results
   - Visual representations

## Analysis Components

### Speech Analysis
- Words per minute
- Pause patterns
- Voice quality metrics
- Pitch analysis

### Cognitive Assessment
- Risk score calculation
- Pattern recognition
- Decline indicators
- Detailed metrics

### Sentiment Evaluation
- Overall sentiment
- Sentence analysis
- Confidence scoring
- Trend visualization

## Technical Details

### Backend
- Flask REST API
- FFmpeg audio processing
- Librosa audio analysis
- BERT sentiment analysis
- NLTK text processing
- Matplotlib visualizations

### Frontend
- Responsive Bootstrap design
- Real-time audio recording
- Dynamic result display
- Interactive visualizations

## Directory Structure
```
memotag/
├── api.py                 # Main application
├── templates/
│   └── index.html        # Web interface
├── ffmpeg/               # Audio processing
├── static/              # Static assets
├── visualizations/      # Generated graphics
└── requirements.txt     # Dependencies
```

## Dependencies

Key packages and versions:
- Flask==2.3.3
- librosa==0.10.1
- numpy==1.24.3
- torch==2.0.1
- transformers==4.33.2
- matplotlib==3.7.1
- pydub==0.25.1
- SpeechRecognition==3.10.0

## Error Handling

The system includes comprehensive error handling for:
- Audio processing issues
- Speech recognition failures
- Analysis errors
- File format problems

## Future Enhancements

Planned improvements include:
- Additional cognitive markers
- Enhanced visualization options
- Historical data comparison
- Downloadable analysis reports
- Extended audio format support

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## Acknowledgments

- FFmpeg for audio processing
- Google Speech Recognition
- BERT model for sentiment analysis
- Various open-source contributors


 