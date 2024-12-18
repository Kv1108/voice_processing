import os
import librosa
import numpy as np
import soundfile as sf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.exceptions import NotFittedError

# Global Variables
SPEECH_SEGMENTS_FOLDER = "speech_segments"    # Input folder (from Stage 2)
SPEAKER_OUTPUT_FOLDER = "speaker_audio"       # Output folder for separated speakers

def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def extract_audio_features(audio_segment, sr):
    """
    Extract audio features for clustering:
    - MFCCs, Delta MFCCs, Spectral Contrast, Pitch, RMS.
    """
    # Ensure segment is long enough for processing
    if len(audio_segment) < 512:
        raise ValueError("Audio segment too short for feature extraction.")

    mfcc = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=13, n_fft=512, hop_length=256)
    delta_mfcc = librosa.feature.delta(mfcc)
    chroma = librosa.feature.chroma_stft(y=audio_segment, sr=sr, n_fft=512, hop_length=256)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_segment, sr=sr, n_fft=512)
    rms = librosa.feature.rms(y=audio_segment, frame_length=512, hop_length=256)
    pitch = librosa.yin(audio_segment, fmin=50, fmax=300, sr=sr, frame_length=512, hop_length=256)

    # Combine features
    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.mean(delta_mfcc, axis=1),
        np.mean(chroma, axis=1),
        np.mean(spectral_contrast, axis=1),
        [np.mean(rms)],
        [np.mean(pitch)],
    ])
    return features
    mfcc = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc)
    chroma = librosa.feature.chroma_stft(y=audio_segment, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_segment, sr=sr)
    rms = librosa.feature.rms(y=audio_segment)
    pitch = librosa.yin(audio_segment, fmin=50, fmax=300, sr=sr)

    # Combine features
    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.mean(delta_mfcc, axis=1),
        np.mean(chroma, axis=1),
        np.mean(spectral_contrast, axis=1),
        [np.mean(rms)],
        [np.mean(pitch)]
    ])
    return features

def estimate_num_speakers(features):
    """
    Estimate the optimal number of speakers using Silhouette Score.
    """
    min_clusters = 2
    max_clusters = min(10, len(features))  # Ensure max_clusters doesn't exceed the number of samples

    if len(features) < min_clusters:
        print("[Error] Not enough features to perform clustering. Skipping...")
        return min_clusters  # Default to at least 2 speakers if not enough data

    best_num_clusters = min_clusters
    best_silhouette = -1

    for num_clusters in range(min_clusters, max_clusters + 1):
        try:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(features)
            labels = kmeans.labels_

            # Only compute silhouette score if we have valid clusters
            if len(set(labels)) > 1 and len(features) > num_clusters:
                silhouette = silhouette_score(features, labels)
                if silhouette > best_silhouette:
                    best_num_clusters = num_clusters
                    best_silhouette = silhouette
        except ValueError as e:
            print(f"[Error] ValueError during clustering: {e}")
            break
        except NotFittedError as e:
            print(f"[Error] NotFittedError during clustering: {e}")
            break

    return best_num_clusters

def cluster_speakers(features, num_speakers):
    """
    Perform K-Means clustering to group segments by speaker.
    """
    kmeans = KMeans(n_clusters=num_speakers, random_state=42)
    labels = kmeans.fit_predict(features)
    return labels

def save_speaker_audio(cluster_labels, audio_segments, sr, file_name, output_folder):
    """
    Save clustered audio segments into speaker-specific audio files.
    Args:
        cluster_labels (list): Labels indicating speaker clusters.
        audio_segments (list): List of audio segments (numpy arrays).
        sr (int): Sampling rate.
        file_name (str): Original file name.
        output_folder (str): Folder to save speaker audio files.
    """
    ensure_folder_exists(output_folder)
    speaker_segments = {}

    for idx, label in enumerate(cluster_labels):
        if label not in speaker_segments:
            speaker_segments[label] = []
        speaker_segments[label].append(audio_segments[idx])

    # Combine and save audio for each speaker
    for speaker, segments in speaker_segments.items():
        combined_audio = np.concatenate(segments)
        speaker_file = os.path.join(output_folder, f"{file_name}_speaker_{speaker + 1}.wav")
        sf.write(speaker_file, combined_audio, sr)
        print(f"Saved Speaker {speaker + 1} audio to {speaker_file}")

def process_speaker_separation(input_folder):
    files = [f for f in os.listdir(input_folder) if f.endswith(".wav")]
    for file in files:
        try:
            print(f"Processing speaker separation for: {file}")
            file_path = os.path.join(input_folder, file)

            # Load audio
            audio, sr = librosa.load(file_path, sr=None)

            # Divide into fixed-length chunks (1-second chunks for simplicity)
            chunk_size = sr  # 1 second
            audio_segments = [
                segment for segment in (audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size))
                if len(segment) > 512  # Filter short segments
            ]

            if not audio_segments:
                print(f"[Warning] No valid audio segments for {file}. Skipping...")
                continue

            # Extract features for each segment
            features = np.array([extract_audio_features(segment, sr) for segment in audio_segments])

            # Skip processing if not enough features
            if len(features) < 2:
                print(f"[Warning] Not enough features for clustering. Skipping {file}...")
                continue

            # Estimate number of speakers and perform clustering
            num_speakers = estimate_num_speakers(features)
            print(f"Estimated number of speakers: {num_speakers}")

            # Perform KMeans clustering
            labels = cluster_speakers(features, num_speakers)

            # Save segmented audio for each speaker
            save_speaker_audio(labels, audio_segments, sr, file.split(".")[0], SPEAKER_OUTPUT_FOLDER)

        except Exception as e:
            print(f"Error processing {file}: {e}")

    # Corrected variable name from `folder_path` to `input_folder`
    files = [f for f in os.listdir(input_folder) if f.endswith(".wav")]
    for file in files:
        try:
            print(f"Processing speaker separation for: {file}")
            file_path = os.path.join(input_folder, file)

            # Load audio
            audio, sr = librosa.load(file_path, sr=None)

            # Divide into fixed-length chunks (1-second chunks for simplicity)
            chunk_size = sr  # 1 second
            audio_segments = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]

            # Extract features for each segment
            features = np.array([extract_audio_features(segment, sr) for segment in audio_segments])

            # Skip processing if not enough features
            if len(features) < 2:
                print(f"[Warning] Not enough features for clustering. Skipping {file}...")
                continue

            # Estimate number of speakers and perform clustering
            num_speakers = estimate_num_speakers(features)
            print(f"Estimated number of speakers: {num_speakers}")

            # Perform KMeans clustering
            labels = cluster_speakers(features, num_speakers)

            # Save segmented audio for each speaker
            save_speaker_audio(labels, audio_segments, sr, file.split(".")[0], SPEAKER_OUTPUT_FOLDER)

        except Exception as e:
            print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    print("Starting Speaker Separation...")
    ensure_folder_exists(SPEAKER_OUTPUT_FOLDER)
    process_speaker_separation(SPEECH_SEGMENTS_FOLDER)
