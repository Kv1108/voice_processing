import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import speech_recognition as sr

from pyAudioAnalysis import audioSegmentation as aS
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def ensure_folder_exists(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")

def get_latest_audio_file(folder_path):
    audio_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in folder: {folder_path}")
    audio_files.sort(key=lambda f: os.path.getmtime(os.path.join(folder_path, f)), reverse=True)
    return os.path.join(folder_path, audio_files[0])

def extract_acoustic_features(signal, sr):
    # Extract Zero-Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=signal)[0]
    
    # Extract Energy (using corrected librosa feature call)
    energy = librosa.feature.rms(y=signal)[0]
    
    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)  # Take the mean across time axis
    
    # Extract Pitch (using librosa's estimate of pitch, optional)
    pitches, magnitudes = librosa.piptrack(y=signal, sr=sr)
    pitch = np.max(pitches, axis=0)  # Maximum pitch at each time frame
    pitch_mean = np.mean(pitch[pitch > 0])  # Exclude non-voiced segments
    
    # Combine all the features
    features = np.hstack([zero_crossing_rate.mean(), energy.mean(), mfcc_mean, pitch_mean])
    return features


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
        kmeans = KMeans(n_clusters=n, random_state=0)
        labels = kmeans.fit_predict(features)
        silhouette_scores.append(silhouette_score(features, labels))
    
    if not silhouette_scores:
        return 1  # Default to 1 speaker if no valid clustering is possible
    
    best_n = np.argmax(silhouette_scores) + 2
    return best_n

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
        return f"Error: Could not request results from Google Speech Recognition service; {e}"

def plot_clusters(features, labels):
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)
    
    plt.figure(figsize=(10, 6))
    for i in range(len(reduced_features)):
        plt.scatter(reduced_features[i, 0], reduced_features[i, 1], c=f'C{labels[i]}', label=f'Cluster {labels[i]}')
    plt.title("Clustering of Acoustic Features")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()

def process_vad_and_clustering(audio_file):
    signal, sr = librosa.load(audio_file, sr=None)

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

    feature_list = []
    for start, end in segments:
        segment_audio = signal[int(start * sr):int(end * sr)]
        if len(segment_audio) == 0:
            continue
        features = extract_acoustic_features(segment_audio, sr)
        feature_list.append(features)
    
    if not feature_list:
        raise ValueError("No valid segments were found in the audio file.")

    features = np.array(feature_list)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    n_clusters = estimate_num_speakers(features_scaled)
    print(f"Estimated number of speakers: {n_clusters}")

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(features_scaled)

    clustered_segments = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(labels):
        start_time = segments[idx][0]
        end_time = segments[idx][1]    
        clustered_segments[label].append((start_time, end_time))

    save_cluster_audio(clustered_segments, signal, sr)
    plot_clusters(features_scaled, labels)

def main():
    audio_folder = "recorded_audio"
    latest_audio_file = get_latest_audio_file(audio_folder)
    print(f"Processing latest audio file: {latest_audio_file}")
    process_vad_and_clustering(latest_audio_file)

if __name__ == "__main__":
    main()
