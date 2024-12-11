try:
    import librosa
    import soundfile as sf
except ImportError as e:
    raise ImportError("Required library is not installed. Please install it using 'pip install librosa soundfile' before running this script.") from e

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA

def get_latest_audio_file(folder_path):
    audio_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in folder: {folder_path}")
    
    audio_files.sort(key=lambda f: os.path.getmtime(os.path.join(folder_path, f)), reverse=True)
    return os.path.join(folder_path, audio_files[0])

def save_cluster_audio(cluster_segments, audio, sr):
    os.makedirs("clusters", exist_ok=True)
    for cluster, segments in cluster_segments.items():
        cluster_audio = []
        for start, end in segments:
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            cluster_audio.append(audio[start_sample:end_sample])
        
        cluster_audio = np.concatenate(cluster_audio) if cluster_audio else np.array([])
        cluster_file = f"clusters/cluster_{cluster}.wav"
        sf.write(cluster_file, cluster_audio, sr)
        print(f"Audio for Cluster {cluster} saved to {cluster_file}")

audio_folder = "recorded_audio"  
latest_audio_file = get_latest_audio_file(audio_folder)
print(f"Processing latest audio file: {latest_audio_file}")

audio, sr = librosa.load(latest_audio_file, sr=None)

frame_size = 2048  # Number of samples per frame
hop_size = 1024    # Step size for moving the window
frames = librosa.util.frame(audio, frame_length=frame_size, hop_length=hop_size)

mfcc_features = []
for frame in frames.T:  # Iterate through frames
    mfcc = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)  # Take mean across coefficients
    mfcc_features.append(mfcc_mean)

mfcc_features = np.array(mfcc_features)

scaler = StandardScaler()
mfcc_features_scaled = scaler.fit_transform(mfcc_features)

n_clusters = 2  # Adjust this based on the expected number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
labels = kmeans.fit_predict(mfcc_features_scaled)

pca = PCA(n_components=2)
reduced_features = pca.fit_transform(mfcc_features_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', s=10)
plt.title("Clusters of Audio Segments")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster")
plt.show()

clustered_segments = {i: [] for i in range(n_clusters)}
for idx, label in enumerate(labels):
    start_time = idx * hop_size / sr  # Convert frame index to time
    end_time = (idx * hop_size + frame_size) / sr
    clustered_segments[label].append((start_time, end_time))

output_file = "cluster_segments.txt"
with open(output_file, "w") as file:
    for cluster, segments in clustered_segments.items():
        file.write(f"Cluster {cluster}:\n")
        for start, end in segments[:10]:  # Display first 10 segments
            file.write(f"  {start:.2f}s - {end:.2f}s\n")

print(f"Clustered segment information saved to {output_file}")

# Save audio for each cluster
save_cluster_audio(clustered_segments, audio, sr)
