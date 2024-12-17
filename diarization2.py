import os
import librosa
import numpy as np
import threading
import time
import soundfile as sf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from pyAudioAnalysis import audioSegmentation as aS
import speech_recognition as sr
from utils import ensure_folder_exists
import keyboard

# Global Variables
audio_folder = "recorded_audio"
processed_folder = "processed_audio"
clusters_folder = "clusters"
transcriptions_folder = "transcriptions"

# Ensure required folders exist
ensure_folder_exists(audio_folder)
ensure_folder_exists(processed_folder)
ensure_folder_exists(clusters_folder)
ensure_folder_exists(transcriptions_folder)

def apply_audio_filters(audio):
    """Apply filters to enhance audio quality."""
    audio = librosa.effects.preemphasis(audio)
    return audio

def extract_audio_features(audio_segment, sr):
    mfcc = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    chroma = librosa.feature.chroma_stft(y=audio_segment, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_segment, sr=sr)
    pitch = librosa.yin(audio_segment, fmin=50, fmax=300, sr=sr)
    rms = librosa.feature.rms(y=audio_segment)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_segment, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=audio_segment)

    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.mean(delta_mfcc, axis=1),
        np.mean(delta2_mfcc, axis=1),
        np.mean(chroma, axis=1),
        np.mean(spectral_contrast, axis=1),
        [np.mean(pitch)],
        np.mean(rms, axis=1),
        np.mean(spectral_centroid, axis=1),
        np.mean(spectral_bandwidth, axis=1),
        np.mean(zcr, axis=1)
    ])
    return features

def estimate_num_speakers(features, max_speakers=10):
    silhouette_scores = []
    for n in range(2, min(max_speakers + 1, len(features))):
        kmeans = KMeans(n_clusters=n, random_state=0)
        labels = kmeans.fit_predict(features)
        silhouette_scores.append(silhouette_score(features, labels))

    return np.argmax(silhouette_scores) + 2 if silhouette_scores else 1

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        transcription = recognizer.recognize_google(audio)
        return transcription
    except Exception as e:
        return f"Error during transcription: {e}"

def save_transcription(audio_path, transcription):
    base_name = os.path.basename(audio_path)
    transcription_file = os.path.join(transcriptions_folder, f"{os.path.splitext(base_name)[0]}_transcription.txt")
    with open(transcription_file, "w") as file:
        file.write(transcription)
    print(f"Transcription saved to {transcription_file}")

def save_cluster_audio(cluster_segments, audio, sr):
    for cluster, segments in cluster_segments.items():
        cluster_audio = []
        for start, end in segments:
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            cluster_audio.append(audio[start_sample:end_sample])

        cluster_audio = np.concatenate(cluster_audio) if cluster_audio else np.array([])
        cluster_file = os.path.join(clusters_folder, f"cluster_{cluster}.wav")
        sf.write(cluster_file, cluster_audio, sr)
        print(f"Cluster {cluster} saved to {cluster_file}")

def process_audio(audio_file):
    print(f"Processing: {audio_file}")
    signal, sr = librosa.load(audio_file, sr=None)
    signal = apply_audio_filters(signal)

    segments = aS.silence_removal(signal, sr, 0.05, 0.02, 0.02, 0.05)
    features = [extract_audio_features(signal[int(start * sr):int(end * sr)], sr) for start, end in segments]
    features = np.array(features)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    n_clusters = estimate_num_speakers(scaled_features)
    print(f"Estimated speakers: {n_clusters}")

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(scaled_features)

    clustered_segments = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(labels):
        clustered_segments[label].append(segments[idx])

    save_cluster_audio(clustered_segments, signal, sr)

    for cluster in clustered_segments:
        cluster_file = os.path.join(clusters_folder, f"cluster_{cluster}.wav")
        transcription = transcribe_audio(cluster_file)
        save_transcription(cluster_file, transcription)

def monitor_folder():
    processed_files = set(os.listdir(processed_folder))
    while True:
        if keyboard.is_pressed('q'):
            print("Stopping monitoring...")
            break

        current_files = set(os.listdir(audio_folder))
        new_files = current_files - processed_files

        for audio_file in new_files:
            full_path = os.path.join(audio_folder, audio_file)
            if os.path.isfile(full_path):
                try:
                    process_audio(full_path)
                    os.rename(full_path, os.path.join(processed_folder, audio_file))
                    processed_files.add(audio_file)
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")

        time.sleep(5)

if __name__ == "__main__":
    print("Starting real-time audio processing... Press 'q' to stop.")
    monitor_thread = threading.Thread(target=monitor_folder, daemon=True)
    monitor_thread.start()
    monitor_thread.join()
