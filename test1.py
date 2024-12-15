import os
import librosa
import numpy as np
from pyAudioAnalysis import audioSegmentation as aS
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import soundfile as sf
import speech_recognition as sr

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

def extract_pitch(signal, sr):
    # Use librosa to estimate pitch (fundamental frequency)
    pitches, magnitudes = librosa.core.piptrack(y=signal, sr=sr)
    pitch_mean = np.mean(pitches, axis=1)
    return pitch_mean

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

def save_transcription_to_file(audio_path, transcription, output_folder):
    base_name = os.path.basename(audio_path)
    transcription_file_path = os.path.join(output_folder, f"{os.path.splitext(base_name)[0]}_transcription.txt")
    try:
        with open(transcription_file_path, "w") as file:
            file.write(transcription)
        print(f"Transcription saved to '{transcription_file_path}'")
    except Exception as e:
        print(f"Error saving transcription: {e}")

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

    pitch_features = []
    for segment in segments:
        start, end = segment
        segment_audio = signal[int(start * sr):int(end * sr)]
        if len(segment_audio) == 0:
            continue
        pitch = extract_pitch(segment_audio, sr)
        pitch_mean = np.mean(pitch)  # Average pitch for each segment
        pitch_features.append(pitch_mean)

    if not pitch_features:
        raise ValueError("No valid segments were found in the audio file.")

    pitch_features = np.array(pitch_features).reshape(-1, 1)
    scaler = StandardScaler()
    pitch_features_scaled = scaler.fit_transform(pitch_features)

    n_clusters = estimate_num_speakers(pitch_features_scaled)
    print(f"Estimated number of speakers: {n_clusters}")

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(pitch_features_scaled)

    clustered_segments = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(labels):
        start_time = segments[idx][0]
        end_time = segments[idx][1]    
        clustered_segments[label].append((start_time, end_time))

    save_cluster_audio(clustered_segments, signal, sr)

    transcription_folder = "transcriptions"
    ensure_folder_exists(transcription_folder)

    # Merge segments in each cluster to form longer, coherent chunks
    for cluster in clustered_segments:
        cluster_file = f"clusters/cluster_{cluster}.wav"
        if len(clustered_segments[cluster]) > 1:
            # Combine the audio segments into one longer segment
            cluster_audio = []
            for start, end in clustered_segments[cluster]:
                start_sample = int(start * sr)
                end_sample = int(end * sr)
                cluster_audio.append(signal[start_sample:end_sample])
            cluster_audio = np.concatenate(cluster_audio)
            sf.write(cluster_file, cluster_audio, sr)
        
        # Transcribe the longer merged segment
        transcription = transcribe_audio(cluster_file)
        save_transcription_to_file(cluster_file, transcription, transcription_folder)


    # Plot pitch values for clustering
    plt.figure(figsize=(12, 8))
    plt.scatter(range(len(pitch_features_scaled)), pitch_features_scaled, c=labels, cmap='viridis', s=50, alpha=0.7, edgecolors='k')
    plt.title(f"Clusters of Audio Segments based on Pitch (Estimated Speakers: {n_clusters})", fontsize=14)
    plt.xlabel("Segment Index")
    plt.ylabel("Normalized Pitch")
    plt.colorbar(label="Cluster")
    plt.grid(True)
    plt.show()


def main():
    audio_folder = "recorded_audio"
    latest_audio_file = get_latest_audio_file(audio_folder)
    print(f"Processing latest audio file: {latest_audio_file}")
    process_vad_and_clustering(latest_audio_file)

if __name__ == "__main__":
    main()
