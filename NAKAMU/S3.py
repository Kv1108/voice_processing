import os
import librosa
import numpy as np
from pyAudioAnalysis import audioSegmentation as aS
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import soundfile as sf

def get_latest_audio_file(folder_path):
    audio_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in folder: {folder_path}")
    
    audio_files.sort(key=lambda f: os.path.getmtime(os.path.join(folder_path, f)), reverse=True)
    return os.path.join(folder_path, audio_files[0])

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

def estimate_num_speakers(features, max_speakers=10):
    if len(features) < 2:
        print("Not enough features to estimate number of speakers.")
        return 1

    silhouette_scores = []
    for n in range(2, min(max_speakers + 1, len(features))):
        n_neighbors = min(10, len(features) - 1)  # Use the smaller of 10 or (n_samples - 1)
        clustering = SpectralClustering(n_clusters=n, affinity='nearest_neighbors', n_neighbors=n_neighbors, random_state=0)

        labels = clustering.fit_predict(features)
        silhouette_scores.append(silhouette_score(features, labels))
    
    if not silhouette_scores:
        return 1  # Default to 1 speaker if no valid clustering is possible
    
    best_n = np.argmax(silhouette_scores) + 2
    return best_n

def process_vad_and_clustering(audio_file):
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

    if not segments:
        raise ValueError("No valid segments were found in the audio file.")
    
    mfcc_features = []
    for segment in segments:
        start, end = segment
        segment_audio = signal[int(start * sr):int(end * sr)]
        if len(segment_audio) == 0:  # Skip empty segments
            continue
        mfcc = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_features.append(mfcc_mean)

    if not mfcc_features:
        raise ValueError("No valid features were extracted from the audio file.")

    mfcc_features = np.array(mfcc_features)
    scaler = StandardScaler()
    mfcc_features_scaled = scaler.fit_transform(mfcc_features)

    n_clusters = estimate_num_speakers(mfcc_features_scaled)
    print(f"Estimated number of speakers: {n_clusters}")

    # Dynamic n_neighbors adjustment to avoid ValueError
    n_samples = len(mfcc_features_scaled)
    n_neighbors = min(10, n_samples - 1)  # Ensure n_neighbors is valid

    print(f"Number of samples: {n_samples}, n_neighbors set to: {n_neighbors}")
    
    try:
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=0, n_neighbors=n_neighbors)  
        labels = spectral.fit_predict(mfcc_features_scaled)
    except ValueError as e:
        print(f"Error with SpectralClustering: {e}")
        n_clusters = 1  # Fallback to 1 cluster
        labels = [0] * n_samples  # Assign all samples to one cluster

    if len(mfcc_features_scaled) > 1:
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(mfcc_features_scaled)

        plt.figure(figsize=(10, 6))
        plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', s=10)
        plt.title(f"Clusters of Audio Segments (Estimated Speakers: {n_clusters})")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.colorbar(label="Cluster")
        plt.show()

    clustered_segments = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(labels):
        start_time = segments[idx][0]  
        end_time = segments[idx][1]    
        clustered_segments[label].append((start_time, end_time))

    save_cluster_audio(clustered_segments, signal, sr)

audio_folder = "recorded_audio"
latest_audio_file = get_latest_audio_file(audio_folder)
print(f"Processing latest audio file: {latest_audio_file}")

process_vad_and_clustering(latest_audio_file)
