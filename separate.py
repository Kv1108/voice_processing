import librosa
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import soundfile as sf
import speech_recognition as sr

# Function to get the latest audio file
def get_latest_audio_file(folder_path):
    audio_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in folder: {folder_path}")
    audio_files.sort(key=lambda f: os.path.getmtime(os.path.join(folder_path, f)), reverse=True)
    return os.path.join(folder_path, audio_files[0])

# Function to transcribe a specific audio segment
def transcribe_audio_segment(audio_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
            transcription = recognizer.recognize_google(audio)
            return transcription
    except sr.UnknownValueError:
        return "Error: Unable to understand the audio."
    except sr.RequestError as e:
        return f"Error: Could not request results from the Google Speech Recognition service; {e}"

# Main function
def main():
    # Folder containing the audio files
    audio_folder = "recorded_audio"
    latest_audio_file = get_latest_audio_file(audio_folder)
    print(f"Processing latest audio file: {latest_audio_file}")

    # Load audio
    audio, sr = librosa.load(latest_audio_file, sr=None)

    # Parameters for framing
    frame_size = 2048
    hop_size = 1024
    frames = librosa.util.frame(audio, frame_length=frame_size, hop_length=hop_size)

    # Extract MFCC features
    mfcc_features = []
    for frame in frames.T:
        mfcc = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_features.append(mfcc_mean)
    mfcc_features = np.array(mfcc_features)

    # Scale the features
    scaler = StandardScaler()
    mfcc_features_scaled = scaler.fit_transform(mfcc_features)

    # Perform clustering
    n_clusters = 2  # Adjust if known; otherwise, estimate automatically
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(mfcc_features_scaled)

    # PCA for visualization
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(mfcc_features_scaled)
    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', s=10)
    plt.title("Clusters of Audio Segments")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(label="Cluster")
    plt.show()

    # Group segments by cluster
    clustered_segments = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(labels):
        start_time = idx * hop_size / sr
        end_time = (idx * hop_size + frame_size) / sr
        clustered_segments[label].append((start_time, end_time))

    # Folder to save temporary audio files
    temp_folder = "temp_audio_segments"
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    # Transcriptions for each cluster
    transcriptions = {i: [] for i in range(n_clusters)}

    # Process each cluster
    for cluster, segments in clustered_segments.items():
        cluster_transcription = []
        for start, end in segments:
            segment_audio = audio[int(start * sr):int(end * sr)]
            segment_file = os.path.join(temp_folder, f"cluster_{cluster}_{int(start)}_{int(end)}.wav")
            
            # Save audio segment in 16-bit PCM format
            sf.write(segment_file, segment_audio, sr, subtype="PCM_16")

            # Transcribe the audio segment
            try:
                transcription = transcribe_audio_segment(segment_file)
                cluster_transcription.append(f"{start:.2f}s - {end:.2f}s: {transcription}")
            except Exception as e:
                print(f"Error transcribing segment {segment_file}: {e}")

        transcriptions[cluster] = cluster_transcription

    # Save transcriptions to a file
    transcription_folder = "transcriptions"
    if not os.path.exists(transcription_folder):
        os.makedirs(transcription_folder)

    for cluster, cluster_transcription in transcriptions.items():
        transcription_file = os.path.join(transcription_folder, f"cluster_{cluster}_transcription.txt")
        with open(transcription_file, "w") as file:
            for line in cluster_transcription:
                file.write(line + "\n")
        print(f"Cluster {cluster} transcription saved to {transcription_file}")

if __name__ == "__main__":
    main()
