import os
import tempfile
import logging
# Set matplotlib backend to non-interactive before other imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import speech_recognition as sr
from pydub import AudioSegment
import librosa
import librosa.display
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from sklearn.preprocessing import StandardScaler

# Disable any GUI backend attempts
plt.ioff()

# Set FFmpeg path
ffmpeg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ffmpeg', 'ffmpeg-6.1.1-essentials_build', 'bin', 'ffmpeg.exe')
AudioSegment.converter = ffmpeg_path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom sentence tokenizer as fallback
def custom_sent_tokenize(text):
    """Simple sentence tokenizer using regular expressions"""
    return [s.strip() for s in RegexpTokenizer('[.!?]+').tokenize(text)]

# Download required NLTK data
logger.info("Downloading required NLTK data...")
try:
    # Create NLTK data directory if it doesn't exist
    nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Download all necessary NLTK data
    required_packages = [
        'punkt_tab',
        'punkt',
        'averaged_perceptron_tagger',
        'maxent_ne_chunker',
        'words'
    ]
    
    for package in required_packages:
        try:
            nltk.download(package, quiet=True, raise_on_error=True)
            logger.info(f"✓ Downloaded NLTK package: {package}")
        except Exception as e:
            logger.error(f"❌ Error downloading {package}: {str(e)}")
            continue

    # Override the default tokenize functions with more robust versions
    def robust_sent_tokenize(text):
        try:
            return sent_tokenize(text)
        except LookupError:
            logger.warning("Falling back to custom sentence tokenizer")
            return custom_sent_tokenize(text)

    def robust_word_tokenize(text):
        try:
            return word_tokenize(text)
        except LookupError:
            logger.warning("Falling back to basic word tokenizer")
            return text.split()

    # Test tokenization
    try:
        test_text = "This is a test sentence. This is another sentence."
        sentences = robust_sent_tokenize(test_text)
        words = robust_word_tokenize(sentences[0])
        logger.info("✓ Tokenization test successful")
    except Exception as e:
        logger.error(f"❌ Error testing tokenization: {str(e)}")
        logger.warning("Using fallback tokenization methods")

    # Replace the global tokenize functions with our robust versions
    nltk.tokenize.sent_tokenize = robust_sent_tokenize
    nltk.tokenize.word_tokenize = robust_word_tokenize
    
    logger.info("✓ NLTK setup completed with fallback mechanisms in place")
except Exception as e:
    logger.error(f"❌ Error in NLTK setup: {str(e)}")
    logger.warning("Continuing with fallback tokenization methods")

app = Flask(__name__)
CORS(app)

# Create necessary directories
os.makedirs('visualizations', exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('temp', exist_ok=True)

# Initialize BERT model and tokenizer
logger.info("Loading BERT model and tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model.eval()
    logger.info("✓ BERT model and tokenizer loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading BERT model: {str(e)}")
    raise

def ensure_temp_dir():
    """Ensure temporary directory exists"""
    temp_dir = tempfile.gettempdir()
    return temp_dir

def convert_audio_to_wav(audio_bytes):
    """Convert audio bytes to WAV format."""
    try:
        # Create temporary files
        temp_input = os.path.join('temp', 'input_audio')
        temp_wav = os.path.join('temp', 'output.wav')
        
        # Save input bytes to temporary file
        with open(temp_input, 'wb') as f:
            f.write(audio_bytes)
        
        try:
            # Try to load the audio file with pydub
            audio = AudioSegment.from_file(temp_input)
            
            # Convert to WAV format with specific parameters
            audio = audio.set_channels(1)  # Convert to mono
            audio = audio.set_frame_rate(16000)  # Set sample rate to 16kHz
            audio = audio.set_sample_width(2)  # Set sample width to 16-bit
            
            # Export as WAV
            audio.export(temp_wav, format='wav', parameters=["-ac", "1", "-ar", "16000"])
            logger.info("Audio converted successfully using pydub")
            
        except Exception as e:
            logger.warning(f"Pydub conversion failed: {str(e)}, trying librosa...")
            try:
                # Try loading with librosa if pydub fails
                y, sr = librosa.load(temp_input, sr=16000, mono=True)
                import soundfile as sf
                sf.write(temp_wav, y, sr, format='WAV', subtype='PCM_16')
                logger.info("Audio converted successfully using librosa")
            except Exception as e:
                raise Exception(f"Both conversion methods failed: {str(e)}")
        
        # Read the converted WAV file
        with open(temp_wav, 'rb') as f:
            wav_data = f.read()
            
        # Clean up temporary files
        try:
            os.remove(temp_input)
            os.remove(temp_wav)
        except Exception as e:
            logger.warning(f"Error cleaning up temporary files: {str(e)}")
            
        return wav_data
        
    except Exception as e:
        raise Exception(f"Error converting audio: {str(e)}")

def analyze_sentiment(text):
    """Analyze sentiment of text using BERT"""
    try:
        logger.info(f"Starting sentiment analysis for text: {text[:100]}...")
        
        # Use NLTK's sent_tokenize for sentence splitting
        try:
            sentences = sent_tokenize(text)
            logger.info(f"Found {len(sentences)} sentences to analyze")
        except Exception as e:
            logger.error(f"❌ Error in sentence tokenization: {str(e)}")
            sentences = [text]
            logger.info("Using fallback sentence splitting")
        
        sentence_results = []
        overall_sentiment = 0
        
        for i, sentence in enumerate(sentences, 1):
            if not sentence.strip():
                continue
            
            logger.info(f"Analyzing sentence {i}/{len(sentences)}")
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            sentiment_score = torch.argmax(predictions).item() + 1  # Scale from 1-5
            confidence_score = torch.max(predictions).item()
            normalized_score = (sentiment_score - 3) / 2  # Convert to -1 to 1 scale
            
            logger.info(f"Sentence {i} sentiment: {sentiment_score}/5 (confidence: {confidence_score:.2f})")
            
            sentence_results.append({
                'sentence': sentence,
                'sentiment': sentiment_score,
                'confidence': float(confidence_score),
                'normalized_score': float(normalized_score)
            })
            
            overall_sentiment += normalized_score
        
        if sentence_results:
            overall_sentiment /= len(sentence_results)
        
        sentiment_label = 'Positive' if overall_sentiment > 0.2 else 'Negative' if overall_sentiment < -0.2 else 'Neutral'
        logger.info(f"Overall sentiment: {sentiment_label} (score: {overall_sentiment:.2f})")
        
        return {
            'overall_sentiment': float(overall_sentiment),
            'sentiment_label': sentiment_label,
            'sentence_analysis': sentence_results,
            'summary': {
                'positive_count': len([s for s in sentence_results if s['normalized_score'] > 0.2]),
                'neutral_count': len([s for s in sentence_results if -0.2 <= s['normalized_score'] <= 0.2]),
                'negative_count': len([s for s in sentence_results if s['normalized_score'] < -0.2]),
                'average_confidence': float(np.mean([s['confidence'] for s in sentence_results])) if sentence_results else 0
            }
        }
    except Exception as e:
        logger.error(f"❌ Error in sentiment analysis: {str(e)}")
        raise

def extract_audio_features(audio_path):
    """Extract audio features using librosa."""
    try:
        # Load audio file
        y, sr = librosa.load(audio_path)
        
        # Extract features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        
        # Calculate statistics
        features = {
            'mfcc_mean': np.mean(mfccs, axis=1).tolist(),
            'mfcc_std': np.std(mfccs, axis=1).tolist(),
            'spectral_centroid_mean': float(np.mean(spectral_centroid)),
            'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
            'zero_crossing_rate_mean': float(np.mean(zero_crossing_rate))
        }
        
        return features
    except Exception as e:
        raise Exception(f"Error extracting features: {str(e)}")

def transcribe_audio(audio_path):
    """Transcribe audio using Google Speech Recognition."""
    try:
        # Initialize recognizer
        recognizer = sr.Recognizer()
        
        # Read the audio file
        with sr.AudioFile(audio_path) as source:
            # Record the audio file
            audio_data = recognizer.record(source)
            
            # Adjust for ambient noise if needed
            recognizer.adjust_for_ambient_noise(source)
            
            # Try transcribing with increased timeout
            try:
                text = recognizer.recognize_google(audio_data, language='en-US')
                return text
            except sr.UnknownValueError:
                raise Exception("Could not understand the audio. Please speak clearly and try again.")
            except sr.RequestError as e:
                raise Exception(f"Could not request results from speech recognition service: {str(e)}")
            
    except Exception as e:
        raise Exception(f"Error transcribing audio: {str(e)}")

def calculate_risk_score(features):
    """Calculate risk score based on audio features."""
    try:
        # Normalize features
        scaler = StandardScaler()
        feature_values = np.array([
            features['spectral_centroid_mean'],
            features['spectral_rolloff_mean'],
            features['zero_crossing_rate_mean']
        ]).reshape(1, -1)
        
        normalized_features = scaler.fit_transform(feature_values)
        
        # Simple risk score calculation (example)
        risk_score = float(np.mean(normalized_features))
        return max(min(risk_score * 100, 100), 0)  # Scale between 0 and 100
    except Exception as e:
        raise Exception(f"Error calculating risk score: {str(e)}")

def generate_audio_visualizations(y, sr, features, sentiment_results=None):
    """Generate audio visualizations and return them as base64 encoded strings."""
    visualizations = {}
    
    # Use a valid matplotlib style
    plt.style.use('default')
    
    # Common figure settings
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.color': '#E0E0E0',
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8
    })
    
    try:
        # 1. Waveform
        fig = plt.figure(figsize=(10, 4))
        librosa.display.waveshow(y, sr=sr, alpha=0.8, color='#2196F3')
        plt.title('Audio Waveform', pad=10, fontsize=12, fontweight='bold')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        visualizations['waveform'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        # 2. Mel Spectrogram
        fig = plt.figure(figsize=(10, 4))
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
        librosa.display.specshow(mel_spect_db, y_axis='mel', x_axis='time', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram', pad=10, fontsize=12, fontweight='bold')
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        visualizations['mel_spectrogram'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        # 3. Pitch Contour
        fig = plt.figure(figsize=(10, 4))
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        plt.plot(np.mean(pitches, axis=0), color='#2196F3', linewidth=2)
        plt.title('Pitch Contour', pad=10, fontsize=12, fontweight='bold')
        plt.xlabel('Time')
        plt.ylabel('Frequency (Hz)')
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        visualizations['pitch_contour'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        # Add Sentiment Analysis Visualization if available
        if sentiment_results and sentiment_results.get('sentence_analysis'):
            fig = plt.figure(figsize=(12, 4))
            sentences = sentiment_results['sentence_analysis']
            
            # Extract sentiment scores and prepare data
            scores = [s['normalized_score'] for s in sentences]
            positions = np.arange(len(scores))
            
            # Create color map based on sentiment
            colors = ['#2196F3' if score > 0.2 else '#FFC107' if score > -0.2 else '#F44336' 
                     for score in scores]
            
            # Plot bars
            plt.bar(positions, scores, color=colors)
            
            # Customize the plot
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.1)
            plt.title('Sentence-level Sentiment Analysis', pad=10, fontsize=12, fontweight='bold')
            plt.xlabel('Sentence Number')
            plt.ylabel('Sentiment Score')
            
            # Add overall sentiment score
            overall_score = sentiment_results.get('overall_sentiment', 0)
            plt.axhline(y=overall_score, color='red', linestyle='--', alpha=0.5, 
                       label=f'Overall Sentiment: {overall_score:.2f}')
            plt.legend()
            
            # Save visualization
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            visualizations['sentiment_analysis'] = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
        # Create empty visualizations if there's an error
        for viz_type in ['waveform', 'mel_spectrogram', 'pitch_contour', 'sentiment_analysis']:
            if viz_type not in visualizations:
                visualizations[viz_type] = ''
    
    return visualizations

def calculate_cognitive_metrics(y, sr, features, transcription):
    """Calculate cognitive decline related metrics."""
    # Speech rate calculation
    words = len(transcription.split())
    duration = len(y) / sr
    speech_rate = words / (duration / 60)  # Words per minute
    
    # Pause analysis
    silence_threshold = 0.01
    intervals = librosa.effects.split(y, top_db=20)
    total_silence = (len(y) - sum(i[1] - i[0] for i in intervals)) / sr
    pause_ratio = total_silence / (len(y) / sr)
    
    # Pitch variability
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_mean = np.mean(pitches[pitches > 0])
    pitch_std = np.std(pitches[pitches > 0])
    
    # Voice quality metrics
    zero_crossings = librosa.zero_crossings(y)
    zcr = sum(zero_crossings) / len(zero_crossings)
    
    return {
        'speech_rate': {
            'value': round(speech_rate, 2),
            'unit': 'words/minute',
            'status': 'Normal' if 120 <= speech_rate <= 180 else 'Abnormal'
        },
        'pause_analysis': {
            'total_pauses': round(total_silence, 2),
            'pause_ratio': round(pause_ratio * 100, 2),
            'status': 'Normal' if pause_ratio < 0.4 else 'High'
        },
        'pitch_metrics': {
            'mean': round(float(pitch_mean), 2),
            'variability': round(float(pitch_std), 2),
            'status': 'Normal' if pitch_std > 20 else 'Reduced'
        },
        'voice_quality': {
            'zero_crossing_rate': round(float(zcr), 4),
            'status': 'Normal' if 0.01 <= zcr <= 0.05 else 'Abnormal'
        }
    }

@app.route('/')
def index():
    """Serve the main application page"""
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    """Analyze audio file and return results with visualizations."""
    try:
        if 'audio' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'Please select an audio file to analyze'
            }), 400

        audio_file = request.files['audio']
        if not audio_file.filename:
            return jsonify({
                'status': 'error',
                'message': 'No file selected'
            }), 400

        logger.info(f"Processing audio file: {audio_file.filename}")
        
        # Process audio file
        audio_bytes = audio_file.read()
        wav_data = convert_audio_to_wav(audio_bytes)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            temp_wav.write(wav_data)
            temp_wav_path = temp_wav.name
        
        try:
            # Load audio for analysis
            y, sr = librosa.load(temp_wav_path)
            
            # Extract features
            features = extract_audio_features(temp_wav_path)
            
            # Transcribe audio
            transcription = transcribe_audio(temp_wav_path)
            
            # Analyze sentiment
            sentiment_results = analyze_sentiment(transcription)
            
            # Generate visualizations with sentiment analysis
            visualizations = generate_audio_visualizations(y, sr, features, sentiment_results)
            
            # Calculate cognitive metrics
            cognitive_metrics = calculate_cognitive_metrics(y, sr, features, transcription)
            
            # Calculate risk score
            risk_score = calculate_risk_score(features)
            
            # Prepare response
            response = {
                'status': 'success',
                'message': 'Analysis completed successfully!',
                'results': {
                    'transcription': transcription,
                    'cognitive_analysis': {
                        'metrics': cognitive_metrics,
                        'visualizations': visualizations,
                        'risk_assessment': {
                            'score': round(risk_score, 2),
                            'level': 'Low' if risk_score < 30 else 'Medium' if risk_score < 70 else 'High',
                            'interpretation': 'Normal cognitive function' if risk_score < 30 
                                           else 'Mild cognitive changes' if risk_score < 70 
                                           else 'Significant cognitive changes'
                        }
                    },
                    'sentiment_analysis': {
                        'overall': {
                            'score': round(sentiment_results['overall_sentiment'] * 100, 2),
                            'label': sentiment_results['sentiment_label']
                        },
                        'sentences': sentiment_results['sentence_analysis'],
                        'summary': sentiment_results['summary'],
                        'visualization': visualizations.get('sentiment_analysis', '')
                    }
                }
            }
            
            logger.info("✓ Analysis completed successfully")
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"❌ Error in analysis: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
            
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(temp_wav_path):
                    os.unlink(temp_wav_path)
            except Exception as e:
                logger.warning(f"Warning: Error removing temporary file: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    # Ensure temp directory exists
    os.makedirs('temp', exist_ok=True)
    # Start Flask server
    app.run(host='0.0.0.0', port=5000, debug=True) 