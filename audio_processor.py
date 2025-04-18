import numpy as np
import librosa
import soundfile as sf
from python_speech_features import mfcc
import speech_recognition as sr
from pyAudioAnalysis import ShortTermFeatures
import spacy
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
import parselmouth
from parselmouth.praat import call
import os
from pydub import AudioSegment
import wave

class AudioProcessor:
    def __init__(self):
        # Load spaCy model for NLP processing
        self.nlp = spacy.load("en_core_web_sm")
        
        # Define hesitation markers and cognitive markers
        self.hesitation_markers = ['uh', 'um', 'er', 'ah', 'like', 'you know']
        self.cognitive_markers = {
            'repetition': ['again', 'repeat', 'same', 'similar'],
            'confusion': ['what', 'where', 'when', 'how', 'why'],
            'memory': ['remember', 'forget', 'recall', 'memory']
        }
        
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        
        # Adjust recognition settings for better sensitivity
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        
        # Set parameters for analysis
        self.frame_length = 2048
        self.hop_length = 512
    
    def process_audio(self, audio_path):
        """Process an audio file and extract relevant features."""
        try:
            print(f"\n=== Processing Audio File: {audio_path} ===")
            
            # First, try speech recognition
            text = self._speech_to_text(audio_path)
            if not text:
                print("Warning: No text was transcribed from the audio")
            else:
                print(f"Transcribed text length: {len(text)} characters")
            
            # Load and preprocess audio for feature extraction
            y, sr = librosa.load(audio_path, sr=None)
            duration = len(y) / sr
            print(f"Audio duration: {duration:.2f} seconds")
            
            y = self._preprocess_audio(y, sr)
            
            # Extract features
            acoustic_features = self._extract_acoustic_features(y, sr)
            linguistic_features = self._analyze_text(text, duration)
            
            # Combine features
            all_features = {**acoustic_features, **linguistic_features}
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(all_features)
            
            return {
                'speech_rate': linguistic_features['words_per_minute'],
                'pause_frequency': acoustic_features['pause_frequency'],
                'hesitation_score': linguistic_features['hesitation_score'],
                'features': all_features,
                'risk_score': risk_score,
                'transcribed_text': text,
                'duration': duration
            }
            
        except Exception as e:
            print(f"Error in process_audio: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise
    
    def _ensure_valid_wav(self, audio_path):
        """Ensure the audio file is a valid WAV file with proper format."""
        try:
            # If it's already a WAV file, verify it
            if audio_path.lower().endswith('.wav'):
                with wave.open(audio_path, 'rb') as wav_file:
                    # Check if it's a valid WAV file
                    if wav_file.getnchannels() == 1 and wav_file.getsampwidth() == 2:
                        return audio_path
            
            # Convert to proper WAV format
            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_channels(1)  # Convert to mono
            audio = audio.set_frame_rate(16000)  # Set sample rate to 16kHz
            audio = audio.set_sample_width(2)  # Set to 16-bit
            
            wav_path = os.path.splitext(audio_path)[0] + '_converted.wav'
            audio.export(wav_path, format='wav')
            print(f"Converted audio to: {wav_path}")
            return wav_path
            
        except Exception as e:
            print(f"Error in _ensure_valid_wav: {str(e)}")
            return audio_path
    
    def _preprocess_audio(self, y, sr):
        """Preprocess audio signal for better analysis."""
        try:
            # Remove DC offset
            y = librosa.util.normalize(y)
            
            # Apply pre-emphasis
            y = librosa.effects.preemphasis(y)
            
            # Remove silence and normalize
            y, _ = librosa.effects.trim(y, top_db=20)
            y = librosa.util.normalize(y)
            
            return y
            
        except Exception as e:
            print(f"Error in _preprocess_audio: {str(e)}")
            return y
    
    def _extract_acoustic_features(self, y, sr):
        """Extract detailed acoustic features using librosa and Praat."""
        # Extract MFCC features
        mfcc_features = mfcc(y, sr)
        
        # Extract pitch and pitch variability using Praat
        sound = parselmouth.Sound(y, sr)
        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values != 0]  # Remove unvoiced frames
        
        # Calculate jitter (pitch perturbation)
        jitter = self._calculate_jitter(pitch_values)
        
        # Calculate shimmer (amplitude perturbation)
        shimmer = self._calculate_shimmer(y, sr)
        
        # Extract energy and energy variability
        energy = librosa.feature.rms(y=y)[0]
        energy_mean = np.mean(energy)
        energy_std = np.std(energy)
        
        # Detect pauses and calculate pause statistics
        pause_features = self._analyze_pauses(y, sr)
        
        # Extract formants (vowel characteristics)
        formants = self._extract_formants(sound)
        
        return {
            'mfcc_mean': np.mean(mfcc_features, axis=0).tolist(),
            'mfcc_std': np.std(mfcc_features, axis=0).tolist(),
            'pitch_mean': float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0,
            'pitch_std': float(np.std(pitch_values)) if len(pitch_values) > 0 else 0,
            'jitter': float(jitter),
            'shimmer': float(shimmer),
            'energy_mean': float(energy_mean),
            'energy_std': float(energy_std),
            'pause_frequency': float(pause_features['frequency']),
            'pause_duration_mean': float(pause_features['duration_mean']),
            'formant1_mean': float(formants['f1_mean']),
            'formant2_mean': float(formants['f2_mean']),
            'formant3_mean': float(formants['f3_mean'])
        }
    
    def _calculate_jitter(self, pitch_values):
        """Calculate jitter (pitch perturbation)."""
        if len(pitch_values) < 2:
            return 0
        
        # Calculate absolute differences between consecutive pitch periods
        diffs = np.abs(np.diff(pitch_values))
        return np.mean(diffs) / np.mean(pitch_values)
    
    def _calculate_shimmer(self, y, sr):
        """Calculate shimmer (amplitude perturbation)."""
        # Extract amplitude envelope
        amplitude = np.abs(librosa.stft(y, n_fft=self.frame_length, hop_length=self.hop_length))
        amplitude = np.mean(amplitude, axis=0)
        
        if len(amplitude) < 2:
            return 0
        
        # Calculate absolute differences between consecutive amplitudes
        diffs = np.abs(np.diff(amplitude))
        return np.mean(diffs) / np.mean(amplitude)
    
    def _analyze_pauses(self, y, sr):
        """Analyze pause patterns in speech."""
        # Detect pauses using energy threshold
        energy = librosa.feature.rms(y=y)[0]
        threshold = np.mean(energy) * 0.1
        pauses = energy < threshold
        
        # Calculate pause statistics
        pause_durations = []
        current_pause = 0
        
        for is_pause in pauses:
            if is_pause:
                current_pause += 1
            elif current_pause > 0:
                pause_durations.append(current_pause)
                current_pause = 0
        
        if current_pause > 0:
            pause_durations.append(current_pause)
        
        return {
            'frequency': len(pause_durations) / (len(y) / sr) * 60,  # pauses per minute
            'duration_mean': np.mean(pause_durations) * self.hop_length / sr if pause_durations else 0
        }
    
    def _extract_formants(self, sound):
        """Extract formant frequencies using Praat."""
        formants = sound.to_formant_burg()
        f1 = formants.get_value_at_time(1, 0.5, 'HERTZ')
        f2 = formants.get_value_at_time(2, 0.5, 'HERTZ')
        f3 = formants.get_value_at_time(3, 0.5, 'HERTZ')
        
        return {
            'f1_mean': float(f1) if f1 else 0,
            'f2_mean': float(f2) if f2 else 0,
            'f3_mean': float(f3) if f3 else 0
        }
    
    def _speech_to_text(self, audio_path):
        """Convert speech to text using Google Speech Recognition with enhanced handling."""
        try:
            print("\n=== Starting Google Speech-to-Text Process ===")
            print(f"Processing file: {audio_path}")
            
            # Convert audio to the format required by Google Speech Recognition
            audio = AudioSegment.from_file(audio_path)
            
            # Convert to mono and set sample rate to 16kHz (preferred by Google Speech)
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(16000)
            print("Audio converted to: Mono, 16kHz sample rate (Google Speech-to-Text preferred format)")
            
            # Export as WAV for speech recognition
            temp_path = "temp_recognition.wav"
            audio.export(temp_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
            print("Audio exported in WAV format for Google Speech-to-Text processing")
            
            with sr.AudioFile(temp_path) as source:
                print("Reading audio file for Google Speech-to-Text...")
                
                # Adjust recognition settings
                self.recognizer.energy_threshold = 300
                self.recognizer.dynamic_energy_threshold = True
                self.recognizer.pause_threshold = 0.8
                print("Recognition settings configured for optimal results")
                
                # Adjust for ambient noise
                print("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Record audio with increased timeout
                print("Recording audio for processing...")
                audio_data = self.recognizer.record(source)
                
                print("Sending request to Google Speech-to-Text API...")
                try:
                    # Try with language model optimization
                    text = self.recognizer.recognize_google(
                        audio_data,
                        language='en-US',
                        show_all=False  # Set to True for debugging
                    )
                    print(f"Google Speech-to-Text transcription successful!")
                    print(f"Transcribed text: {text}")
                    
                    # Clean up temporary file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        print("Temporary audio file cleaned up")
                    
                    return text
                    
                except sr.UnknownValueError:
                    print("Google Speech-to-Text could not understand the audio")
                    print("Possible issues:")
                    print("- Audio quality too low")
                    print("- No speech detected")
                    print("- Background noise interference")
                    print("- Speech not clear enough for recognition")
                except sr.RequestError as e:
                    print(f"Error accessing Google Speech-to-Text service; {e}")
                    print("Check your internet connection and try again")
                
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                print("Temporary audio file cleaned up")
            
            return ""
            
        except Exception as e:
            print(f"Error in Google Speech-to-Text process: {str(e)}")
            print("Full error details:", e.__class__.__name__)
            import traceback
            print(traceback.format_exc())
            return ""
    
    def _analyze_text(self, text, duration):
        """Analyze text features using spaCy."""
        if not text:
            return {
                'words_per_minute': 0,
                'hesitation_score': 0,
                'word_diversity': 0,
                'sentence_complexity': 0,
                'cognitive_markers': {k: 0 for k in self.cognitive_markers.keys()}
            }
        
        doc = self.nlp(text)
        
        # Calculate words per minute using actual duration
        words_per_minute = (len([token for token in doc if not token.is_punct]) / duration) * 60 if duration > 0 else 0
        
        # Calculate hesitation score
        hesitation_count = sum(1 for token in doc if token.text.lower() in self.hesitation_markers)
        hesitation_score = hesitation_count / len(doc) if len(doc) > 0 else 0
        
        # Calculate word diversity
        unique_words = len(set(token.text.lower() for token in doc))
        word_diversity = unique_words / len(doc) if len(doc) > 0 else 0
        
        # Calculate sentence complexity
        sentences = list(doc.sents)
        sentence_complexity = np.mean([len(sent) for sent in sentences]) if sentences else 0
        
        # Analyze cognitive markers
        cognitive_scores = {}
        for marker_type, markers in self.cognitive_markers.items():
            count = sum(1 for token in doc if token.text.lower() in markers)
            cognitive_scores[marker_type] = count / len(doc) if len(doc) > 0 else 0
        
        return {
            'words_per_minute': words_per_minute,
            'hesitation_score': hesitation_score,
            'word_diversity': word_diversity,
            'sentence_complexity': sentence_complexity,
            'cognitive_markers': cognitive_scores
        }
    
    def _calculate_risk_score(self, features):
        """
        Calculate a risk score based on the extracted features.
        Higher scores indicate higher risk of cognitive decline.
        """
        # Weights for different feature categories
        weights = {
            'acoustic': {
                'pause_frequency': 0.15,
                'pause_duration_mean': 0.1,
                'jitter': 0.1,
                'shimmer': 0.1,
                'pitch_std': 0.05
            },
            'linguistic': {
                'hesitation_score': 0.15,
                'word_diversity': 0.1,
                'sentence_complexity': 0.1,
                'cognitive_markers': 0.15
            }
        }
        
        # Normalize features to 0-1 range
        normalized_features = {
            'pause_frequency': min(features['pause_frequency'] / 10, 1),
            'pause_duration_mean': min(features['pause_duration_mean'] / 0.5, 1),
            'jitter': min(features['jitter'] * 100, 1),
            'shimmer': min(features['shimmer'] * 100, 1),
            'pitch_std': min(features['pitch_std'] / 50, 1),
            'hesitation_score': features['hesitation_score'],
            'word_diversity': 1 - features['word_diversity'],
            'sentence_complexity': 1 - min(features['sentence_complexity'] / 20, 1),
            'cognitive_markers': sum(features['cognitive_markers'].values()) / len(features['cognitive_markers'])
        }
        
        # Calculate weighted sum
        risk_score = 0
        for category, category_weights in weights.items():
            for feature, weight in category_weights.items():
                if feature == 'cognitive_markers':
                    risk_score += normalized_features[feature] * weight
                else:
                    risk_score += normalized_features[feature] * weight
        
        return min(max(risk_score, 0), 1)  # Ensure score is between 0 and 1 