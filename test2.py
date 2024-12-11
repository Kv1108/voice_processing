import os
import librosa
import numpy as np
from pyAudioAnalysis import audioSegmentation as aS
from sklearn.cluster import KMeans
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

def estimate_num_speakers(features, max_speakers=5):
    if len(features) < 2:
        print("Not enough features to estimate number of speakers.")
        return 1

    silhouette_scores = []
    for n in range(2, min(max_speakers + 1, len(features))):
        kmeans = KMeans(n_clusters=n, random_state=0)
        labels = kmeans.fit_predict(features)
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


# It is basically using Fourier inversion theorem which is basically for reconstructing the wave(here audio waves) using frequency and phase information about the wave

# for more about the Fourier inversion theorem visit: https://en.wikipedia.org/wiki/Fourier_inversion_theorem

# the cepstrum is the result of computing the inverse Fourier transform (IFT) of the logarithm of the estimated signal spectrum. The applications of it is the analysis of human speech.

# We are using Mel-frequency cepstral coefficients (MFCCs) which are derived from a type of cepstral representation of the audio clip 


    # Extract features from audio segments
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
        raise ValueError("No valid segments were found in the audio file.")

    mfcc_features = np.array(mfcc_features)
    scaler = StandardScaler()
    mfcc_features_scaled = scaler.fit_transform(mfcc_features)

    n_clusters = estimate_num_speakers(mfcc_features_scaled)
    print(f"Estimated number of speakers: {n_clusters}")

    # Applying KMeans clustering

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(mfcc_features_scaled)

    # PCA stands for Principal Component Analysis 

    # The mfcc features we used exists in high dimentional space (13 dimentions here) so uding PCA we are reducing it to 2 dimentions for visualizaion
     
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

    # Save clustered audio segments

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
