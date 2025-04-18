import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import librosa
import librosa.display
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import streamlit as st

def plot_waveform(y, sr, title="Audio Waveform"):
    """Plot audio waveform"""
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    return plt.gcf()

def plot_spectrogram(y, sr, title="Spectrogram"):
    """Plot spectrogram"""
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()

def plot_mel_spectrogram(y, sr, title="Mel Spectrogram"):
    """Plot mel spectrogram"""
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(mel_spect_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()

def plot_feature_correlation(y, sr, title="Feature Correlation Heatmap"):
    """Plot correlation heatmap of audio features"""
    plt.figure(figsize=(10, 8))
    try:
        # Extract features with error handling
        features = {}
        
        # Extract MFCC features
        try:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)  # Reduce to 5 MFCCs
            for i in range(mfcc.shape[0]):
                features[f'MFCC_{i+1}'] = np.mean(mfcc[i])
        except Exception as e:
            st.warning(f"Could not extract MFCC features: {str(e)}")
        
        # Extract spectral features
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['Spectral_Centroid'] = np.mean(spectral_centroids)
        except Exception as e:
            st.warning(f"Could not extract spectral centroid: {str(e)}")
            
        try:
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features['Spectral_Rolloff'] = np.mean(spectral_rolloff)
        except Exception as e:
            st.warning(f"Could not extract spectral rolloff: {str(e)}")
            
        try:
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            features['Zero_Crossing_Rate'] = np.mean(zero_crossing_rate)
        except Exception as e:
            st.warning(f"Could not extract zero crossing rate: {str(e)}")
        
        # Check if we have enough features to create a correlation matrix
        if len(features) < 2:
            plt.text(0.5, 0.5, "Not enough features extracted to compute correlation", 
                    ha='center', va='center')
            return plt.gcf()
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Calculate correlation
        correlation_matrix = df.corr()
        
        # Plot heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title(title)
        plt.tight_layout()
        
    except Exception as e:
        plt.text(0.5, 0.5, f"Error computing correlation: {str(e)}", 
                ha='center', va='center')
    
    return plt.gcf()

def plot_feature_boxplots(y, sr, title="Feature Distributions"):
    """Plot boxplots for audio features"""
    plt.figure(figsize=(12, 6))
    try:
        # Initialize features dictionary
        features_dict = {}
        
        # Extract features in smaller windows
        hop_length = 512
        frame_length = 2048
        n_frames = 50  # Limit the number of frames to analyze
        
        # Calculate features for each frame with error handling
        try:
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=3,  # Reduce to 3 MFCCs
                                        hop_length=hop_length,
                                        n_fft=frame_length)
            for i in range(min(3, mfccs.shape[0])):
                features_dict[f'MFCC_{i+1}'] = mfccs[i][:n_frames]
        except Exception as e:
            st.warning(f"Could not extract MFCC features: {str(e)}")
        
        try:
            # Extract spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(
                y=y, sr=sr, hop_length=hop_length, n_fft=frame_length
            )[0][:n_frames]
            features_dict['Spectral_Centroid'] = spectral_centroids
        except Exception as e:
            st.warning(f"Could not extract spectral centroid: {str(e)}")
        
        try:
            # Extract spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=y, sr=sr, hop_length=hop_length, n_fft=frame_length
            )[0][:n_frames]
            features_dict['Spectral_Rolloff'] = spectral_rolloff
        except Exception as e:
            st.warning(f"Could not extract spectral rolloff: {str(e)}")
        
        # Check if we have any features to plot
        if not features_dict:
            plt.text(0.5, 0.5, "No features could be extracted", 
                    ha='center', va='center')
            return plt.gcf()
        
        # Create DataFrame
        df = pd.DataFrame(features_dict)
        
        # Normalize the features to make them comparable
        for column in df.columns:
            if df[column].std() != 0:
                df[column] = (df[column] - df[column].mean()) / df[column].std()
        
        # Create boxplot
        df_melted = df.melt()
        sns.boxplot(data=df_melted, x='variable', y='value')
        plt.xticks(rotation=45)
        plt.title(title)
        plt.ylabel('Normalized Value')
        plt.tight_layout()
        
    except Exception as e:
        plt.text(0.5, 0.5, f"Error creating boxplots: {str(e)}", 
                ha='center', va='center')
    
    return plt.gcf()

def create_radar_chart(features, title="Speech Features Overview"):
    """Create radar chart using plotly"""
    # Filter out non-numeric values
    numeric_features = {k: v for k, v in features.items() if isinstance(v, (int, float))}
    
    if not numeric_features:
        # Return empty figure if no numeric features
        fig = go.Figure()
        fig.add_annotation(text="No numeric features available",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    categories = list(numeric_features.keys())
    values = list(numeric_features.values())
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Features'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.2]
            )
        ),
        showlegend=True,
        title=title
    )
    return fig

def plot_pitch_contour(y, sr, title="Pitch Contour"):
    """Plot pitch contour"""
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    plt.figure(figsize=(12, 4))
    plt.imshow(pitches, aspect='auto', origin='lower')
    plt.colorbar(label='Frequency (Hz)')
    plt.title(title)
    plt.ylabel('Frequency Bin')
    plt.xlabel('Time Frame')
    plt.tight_layout()
    return plt.gcf()

def plot_energy_curve(y, sr, title="Energy Curve"):
    """Plot energy curve"""
    energy = librosa.feature.rms(y=y)[0]
    frames = range(len(energy))
    t = librosa.frames_to_time(frames, sr=sr)
    
    plt.figure(figsize=(12, 4))
    plt.plot(t, energy)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Energy")
    plt.tight_layout()
    return plt.gcf()

def save_all_visualizations(y, sr, features, output_dir="visualizations"):
    """Generate and save all visualizations"""
    try:
        # Audio visualizations
        plot_waveform(y, sr).savefig(f"{output_dir}/waveform.png")
        plot_spectrogram(y, sr).savefig(f"{output_dir}/spectrogram.png")
        plot_mel_spectrogram(y, sr).savefig(f"{output_dir}/mel_spectrogram.png")
        plot_pitch_contour(y, sr).savefig(f"{output_dir}/pitch_contour.png")
        plot_energy_curve(y, sr).savefig(f"{output_dir}/energy_curve.png")
        
        # Feature visualizations
        features_df = pd.DataFrame([features])
        # Convert any non-numeric values to numeric where possible
        for col in features_df.columns:
            pd.to_numeric(features_df[col], errors='ignore')
        
        plot_feature_correlation(y, sr).savefig(f"{output_dir}/correlation_heatmap.png")
        plot_feature_boxplots(y, sr).savefig(f"{output_dir}/feature_boxplots.png")
        
        # Radar chart
        radar_fig = create_radar_chart(features)
        radar_fig.write_html(f"{output_dir}/radar_chart.html")
        
        plt.close('all')  # Close all figures to free memory
    except Exception as e:
        print(f"Error saving visualizations: {str(e)}")
        raise 