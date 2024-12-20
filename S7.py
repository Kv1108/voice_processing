import os
import librosa
import numpy as np
from pyAudioAnalysis import audioSegmentation as aS
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import soundfile as sf
import scipy

# Helper function to get the latest audio file
def get_latest_audio_file(folder_path):
    audio_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in folder: {folder_path}")
    
    audio_files.sort(key=lambda f: os.path.getmtime(os.path.join(folder_path, f)), reverse=True)
    return os.path.join(folder_path, audio_files[0])

# Helper function to save clustered audio segments
def save_cluster_audio(cluster_segments, audio, sampling_rate):
    os.makedirs("clusters", exist_ok=True)
    for cluster, segments in cluster_segments.items():
        cluster_audio = []
        for start, end in segments:
            start_sample = int(start * sampling_rate)
            end_sample = int(end * sampling_rate)
            cluster_audio.append(audio[start_sample:end_sample])
        
        cluster_audio = np.concatenate(cluster_audio) if cluster_audio else np.array([])        
        cluster_file = f"clusters/cluster_{cluster}.wav"
        sf.write(cluster_file, cluster_audio, sampling_rate)
        print(f"Audio for Cluster {cluster} saved to {cluster_file}")

# Feature extraction function to include additional features like Chroma and Spectral Contrast
def extract_features(audio, sr):
    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    
    # Extract Chroma Features
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    
    # Extract Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    spectral_contrast_mean = np.mean(spectral_contrast, axis=1)

    # Combine all features into a single vector
    return np.concatenate([mfcc_mean, chroma_mean, spectral_contrast_mean])

# Main function to process the audio file and apply VAD and clustering
def process_vad_and_clustering(audio_file):
    # Load audio file
    signal, sr = librosa.load(audio_file, sr=None)  # sr=None preserves the original sampling rate

    if len(signal) == 0:
        raise ValueError(f"The audio file {audio_file} contains no valid audio data.")

    try:
        segments = aS.silence_removal(
            signal=signal, 
            sampling_rate=sr, 
            st_win=0.02, 
            st_step=0.01, 
            smooth_window=0.01, 
            weight=0.05, 
            plot=False
        )
    except Exception as e:
        raise RuntimeError(f"Error during silence removal: {e}")

    # Extract features (MFCC, Chroma, Spectral Contrast) from audio segments
    feature_list = []
    for segment in segments:
        start, end = segment
        segment_audio = signal[int(start * sr):int(end * sr)]
        if len(segment_audio) == 0:  # Skip empty segments
            continue
        features = extract_features(segment_audio, sr)
        feature_list.append(features)

    if not feature_list:
        raise ValueError("No valid segments were found in the audio file.")

    feature_array = np.array(feature_list)
    scaler = StandardScaler()
    feature_array_scaled = scaler.fit_transform(feature_array)

    # Use Gaussian Mixture Model (GMM) for clustering
    gmm = GaussianMixture(n_components=2, random_state=0)  # Assuming 2 speakers
    gmm_labels = gmm.fit_predict(feature_array_scaled)
    print(f"Clusters based on GMM: {np.unique(gmm_labels)}")

    # Visualization with PCA
    if len(feature_array_scaled) > 1:
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(feature_array_scaled)

        plt.figure(figsize=(10, 6))
        plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=gmm_labels, cmap='viridis', s=10)
        plt.title(f"Clusters of Audio Segments (GMM Clustering)")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.colorbar(label="Cluster")
        plt.show()

    # Save clustered audio segments based on GMM
    clustered_segments = {i: [] for i in np.unique(gmm_labels)}
    for idx, label in enumerate(gmm_labels):
        start_time = segments[idx][0]  
        end_time = segments[idx][1]    
        clustered_segments[label].append((start_time, end_time))

    save_cluster_audio(clustered_segments, signal, sr)

# Main execution
audio_folder = "recorded_audio"
latest_audio_file = get_latest_audio_file(audio_folder)
print(f"Processing latest audio file: {latest_audio_file}")

process_vad_and_clustering(latest_audio_file)
