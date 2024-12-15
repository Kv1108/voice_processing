import os
import librosa
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import soundfile as sf
import speech_recognition as sr

# Ensure required folders exist
def ensure_folder_exists(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")

# Get the latest audio file from a folder
def get_latest_audio_file(folder_path):
    audio_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in folder: {folder_path}")
    audio_files.sort(key=lambda f: os.path.getmtime(os.path.join(folder_path, f)), reverse=True)
    return os.path.join(folder_path, audio_files[0])

# Save clustered audio segments
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

# Estimate number of speakers based on clustering
def estimate_num_speakers(features, max_speakers=10):
    if len(features) < 2:
        print("Not enough features to estimate number of speakers.")
        return 1

    silhouette_scores = []
    for n in range(2, min(max_speakers + 1, len(features))):
        kmeans = KMeans(n_clusters=n, random_state=0)
        labels = kmeans.fit_predict(features)
        silhouette_scores.append(silhouette_score(features, labels))
    
    if not silhouette_scores:
        return 1

    best_n = np.argmax(silhouette_scores) + 2
    return best_n

# Transcribe audio using Google Speech Recognition
def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            print(f"Processing audio file: {audio_path}")
            audio = recognizer.record(source)
    except FileNotFoundError:
        return f"Error: File '{audio_path}' not found."
    except Exception as e:
        return f"Error processing the audio file: {e}"

    try:
        print("Transcribing audio...")
        transcription = recognizer.recognize_google(audio)
        return transcription
    except sr.UnknownValueError:
        return "Error: Unable to understand the audio."
    except sr.RequestError as e:
        return f"Error: Could not request results from the Google Speech Recognition service; {e}"

# Save transcription to a file
def save_transcription_to_file(audio_path, transcription, output_folder):
    base_name = os.path.basename(audio_path)
    transcription_file_path = os.path.join(output_folder, f"{os.path.splitext(base_name)[0]}_transcription.txt")
    try:
        with open(transcription_file_path, "w") as file:
            file.write(transcription)
        print(f"Transcription saved to '{transcription_file_path}'")
    except Exception as e:
        print(f"Error saving transcription: {e}")

# Main processing function
def process_vad_and_clustering(audio_file):
    signal, sr = librosa.load(audio_file, sr=None)

    if len(signal) == 0:
        raise ValueError(f"The audio file {audio_file} contains no valid audio data.")

    # Segment audio using energy-based silence removal
    intervals = librosa.effects.split(signal, top_db=30)

    mfcc_features = []
    segments = []
    for start, end in intervals:
        segment_audio = signal[start:end]
        if len(segment_audio) == 0:
            continue
        mfcc = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_features.append(mfcc_mean)
        segments.append((start / sr, end / sr))  # Convert samples to time

    if not mfcc_features:
        raise ValueError("No valid segments were found in the audio file.")

    mfcc_features = np.array(mfcc_features)
    scaler = StandardScaler()
    mfcc_features_scaled = scaler.fit_transform(mfcc_features)

    n_clusters = estimate_num_speakers(mfcc_features_scaled)
    print(f"Estimated number of speakers: {n_clusters}")

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(mfcc_features_scaled)

    clustered_segments = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(labels):
        start_time, end_time = segments[idx]
        clustered_segments[label].append((start_time, end_time))

    save_cluster_audio(clustered_segments, signal, sr)

    transcription_folder = "transcriptions"
    ensure_folder_exists(transcription_folder)

    # Transcribe and save transcriptions
    for cluster in clustered_segments:
        cluster_file = f"clusters/cluster_{cluster}.wav"
        transcription = transcribe_audio(cluster_file)
        save_transcription_to_file(cluster_file, transcription, transcription_folder)

# Main entry point
def main():
    audio_folder = "recorded_audio"
    ensure_folder_exists("clusters")
    ensure_folder_exists("transcriptions")
    
    latest_audio_file = get_latest_audio_file(audio_folder)
    print(f"Processing latest audio file: {latest_audio_file}")
    process_vad_and_clustering(latest_audio_file)

if __name__ == "__main__":
    main()
