import librosa
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 1: Load the audio file
file_path = "myrecording.wav"  # Replace with your file path
audio, sr = librosa.load(file_path, sr=None)

# Step 2: Split the audio into fixed-size frames
frame_size = 2048  # Number of samples per frame
hop_size = 1024    # Step size for moving the window
frames = librosa.util.frame(audio, frame_length=frame_size, hop_length=hop_size)

# Step 3: Extract MFCC features for each frame
mfcc_features = []
for frame in frames.T:  # Iterate through frames
    mfcc = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)  # Take mean across coefficients
    mfcc_features.append(mfcc_mean)

mfcc_features = np.array(mfcc_features)

# Step 4: Normalize the features
scaler = StandardScaler()
mfcc_features_scaled = scaler.fit_transform(mfcc_features)

# Step 5: Apply K-means clustering
n_clusters = 2  # Adjust this based on the expected number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
labels = kmeans.fit_predict(mfcc_features_scaled)

# Step 6: Visualize the clusters
# Use PCA to reduce dimensionality for visualization
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
reduced_features = pca.fit_transform(mfcc_features_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', s=10)
plt.title("Clusters of Audio Segments")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster")
plt.show()

# Optional: Save the clustered segments for further analysis
clustered_segments = {i: [] for i in range(n_clusters)}
for idx, label in enumerate(labels):
    start_time = idx * hop_size / sr  # Convert frame index to time
    end_time = (idx * hop_size + frame_size) / sr
    clustered_segments[label].append((start_time, end_time))

# Print clustered segments
for cluster, segments in clustered_segments.items():
    print(f"Cluster {cluster}:")
    for start, end in segments[:10]:  # Display first 10 segments
        print(f"  {start:.2f}s - {end:.2f}s")
