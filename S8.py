import os
import librosa
import numpy as np
import soundfile as sf
from pyannote.audio.pipelines import SpeakerDiarization
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Helper function to get the latest audio file
def get_latest_audio_file(folder_path):
    audio_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in folder: {folder_path}")
    
    audio_files.sort(key=lambda f: os.path.getmtime(os.path.join(folder_path, f)), reverse=True)
    return os.path.join(folder_path, audio_files[0])

# Function to extract x-vectors (speaker embeddings)
def extract_x_vectors(audio_file):
    # Load audio file
    audio, sr = librosa.load(audio_file, sr=None)

    # Assuming pyannote.audio's pretrained model is used for speaker embedding
    from pyannote.audio import Inference
    inference = Inference("pyannote/speaker-diarization")
    
    # Get embeddings (x-vectors) from the audio file
    embeddings = inference(audio)
    
    return embeddings

# Function to save diarized segments based on clustering
def save_diarized_audio(embeddings, audio_file, sr, n_clusters):
    # Use Gaussian Mixture Model (GMM) to cluster the x-vectors into n clusters (speakers)
    gmm = GaussianMixture(n_components=n_clusters, random_state=0)
    labels = gmm.fit_predict(embeddings)

    # Plot PCA to visualize clusters
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis', s=10)
    plt.title(f"Speaker Diarization (GMM Clustering with {n_clusters} speakers)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(label="Cluster")
    plt.show()

    # Save audio segments based on the clusters
    audio, sr = librosa.load(audio_file, sr=None)
    os.makedirs("diarized_audio", exist_ok=True)
    
    for cluster_id in np.unique(labels):
        cluster_audio = []
        for idx, label in enumerate(labels):
            if label == cluster_id:
                start_sample = idx * 0.02 * sr  # Approximate segment size in samples
                end_sample = (idx + 1) * 0.02 * sr
                cluster_audio.append(audio[int(start_sample):int(end_sample)])

        cluster_audio = np.concatenate(cluster_audio) if cluster_audio else np.array([])
        cluster_filename = f"diarized_audio/cluster_{cluster_id}.wav"
        sf.write(cluster_filename, cluster_audio, sr)
        print(f"Cluster {cluster_id} audio saved to {cluster_filename}")

# Main function for speaker diarization
def process_speaker_diarization(audio_file):
    # Extract x-vectors (embeddings)
    embeddings = extract_x_vectors(audio_file)
    
    # Determine the number of speakers (clusters)
    n_clusters = 2  # You can use a model or heuristic to estimate this, e.g., GMM-based estimation
    
    # Save the diarized audio based on clustering
    save_diarized_audio(embeddings, audio_file, sr=16000, n_clusters=n_clusters)

# Main execution
audio_folder = "recorded_audio"
latest_audio_file = get_latest_audio_file(audio_folder)
print(f"Processing latest audio file: {latest_audio_file}")

process_speaker_diarization(latest_audio_file)
