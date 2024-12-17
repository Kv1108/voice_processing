import librosa
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 1: Load audio file
file_path = "myrecording.wav"  # Replace with your file path
audio, sr = librosa.load(file_path, sr=None)

# Step 2: Divide the audio into fixed-size frames
frame_size = 2048  # Size of each frame
hop_size = 1024    # Step size for moving the window
frames = librosa.util.frame(audio, frame_length=frame_size, hop_length=hop_size)

# Step 3: Extract MFCC features for each frame
mfcc_features = []
for frame in frames.T:
    mfcc = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)  # Average across coefficients
    mfcc_features.append(mfcc_mean)

mfcc_features = np.array(mfcc_features)

# Step 4: Normalize the features
scaler = StandardScaler()
mfcc_features_scaled = scaler.fit_transform(mfcc_features)

# Step 5: Apply DBSCAN clustering
dbscan = DBSCAN(eps=1.5, min_samples=5)  # Tune `eps` for better results
labels = dbscan.fit_predict(mfcc_features_scaled)

# Step 6: Estimate the number of speakers
unique_labels = set(labels)
num_speakers = len(unique_labels) - (1 if -1 in labels else 0)  # Exclude noise label (-1)
print(f"Estimated number of speakers: {num_speakers}")

# Step 7: Visualize clusters
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
reduced_features = pca.fit_transform(mfcc_features_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', s=10)
plt.title("Speaker Clusters (DBSCAN)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster")
plt.show()
