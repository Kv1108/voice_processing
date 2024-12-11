import librosa
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os

# Step 1: Locate the most recent audio file
def get_latest_audio_file(folder_path):
    audio_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in folder: {folder_path}")
    audio_files.sort(key=lambda f: os.path.getmtime(os.path.join(folder_path, f)), reverse=True)
    return os.path.join(folder_path, audio_files[0])

audio_folder = "recorded_audio"
latest_audio_file = get_latest_audio_file(audio_folder)
print(f"Processing latest audio file: {latest_audio_file}")

# Step 2: Load the audio file
audio, sr = librosa.load(latest_audio_file, sr=None)

# Step 3: Split the audio into fixed-size frames
frame_size = 2048
hop_size = 1024
frames = librosa.util.frame(audio, frame_length=frame_size, hop_length=hop_size)

# Step 4: Extract MFCC features for each frame
mfcc_features = []
for frame in frames.T:
    mfcc = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_features.append(mfcc_mean)

mfcc_features = np.array(mfcc_features)

# Step 5: Normalize the features
scaler = StandardScaler()
mfcc_features_scaled = scaler.fit_transform(mfcc_features)

# Step 6: Automatically determine the number of clusters
def estimate_num_speakers(features, max_speakers=5):
    silhouette_scores = []
    for n in range(2, max_speakers + 1):  # Test between 2 and max_speakers clusters
        kmeans = KMeans(n_clusters=n, random_state=0)
        labels = kmeans.fit_predict(features)
        silhouette_scores.append(silhouette_score(features, labels))
    best_n = np.argmax(silhouette_scores) + 2  # +2 because range starts at 2
    return best_n

n_clusters = estimate_num_speakers(mfcc_features_scaled)
print(f"Estimated number of speakers: {n_clusters}")

# Step 7: Apply K-means clustering with the estimated number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
labels = kmeans.fit_predict(mfcc_features_scaled)

# Step 8: Visualize the clusters
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(mfcc_features_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', s=10)
plt.title(f"Clusters of Audio Segments (Estimated Speakers: {n_clusters})")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster")
plt.show()

# Optional: Save the clustered segments for further analysis
clustered_segments = {i: [] for i in range(n_clusters)}
for idx, label in enumerate(labels):
    start_time = idx * hop_size / sr
    end_time = (idx * hop_size + frame_size) / sr
    clustered_segments[label].append((start_time, end_time))

# Print clustered segments
for cluster, segments in clustered_segments.items():
    print(f"Cluster {cluster}:")
    for start, end in segments[:10]:
        print(f"  {start:.2f}s - {end:.2f}s")