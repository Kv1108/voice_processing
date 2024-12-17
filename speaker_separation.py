import os
import librosa
import numpy as np
import soundfile as sf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Global Variables
SPEECH_SEGMENTS_FOLDER = "speech_segments"    # Input folder (from Stage 2)
SPEAKER_OUTPUT_FOLDER = "speaker_audio"       # Output folder for separated speakers

def ensure_folder_exists(folder_path):
    """Ensure the specified folder exists."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def extract_audio_features(audio_segment, sr):
    """
    Extract audio features for clustering:
    - MFCCs, Delta MFCCs, Spectral Contrast, Pitch, RMS.
    """
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

def estimate_num_speakers(features, max_speakers=5):
    """
    Estimate the optimal number of speakers using Silhouette Score.
    Args:
        features (numpy array): Audio features for clustering.
        max_speakers (int): Maximum expected number of speakers.
    Returns:
        int: Estimated number of speakers.
    """
    silhouette_scores = []
    for n in range(2, max_speakers + 1):
        kmeans = KMeans(n_clusters=n, random_state=42)
        labels = kmeans.fit_predict(features)
        silhouette = silhouette_score(features, labels)
        silhouette_scores.append(silhouette)

    best_num_speakers = 2 + np.argmax(silhouette_scores)  # Start from 2 speakers
    print(f"Estimated number of speakers: {best_num_speakers}")
    return best_num_speakers

def cluster_speakers(features, num_speakers):
    """
    Perform K-Means clustering to group segments by speaker.
    Args:
        features (numpy array): Extracted features.
        num_speakers (int): Number of speakers to cluster.
    Returns:
        list: Cluster labels for each segment.
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
    """
    Process all speech segments for speaker separation.
    Args:
        input_folder (str): Path to speech segments folder.
    """
    ensure_folder_exists(SPEAKER_OUTPUT_FOLDER)
    
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if file_path.endswith(".wav"):
            print(f"Processing speaker separation for: {file_name}")

            # Load the audio file
            audio, sr = librosa.load(file_path, sr=None)
            segment_length = 3  # Split into 3-second chunks
            audio_segments = [
                audio[i:i + sr * segment_length] 
                for i in range(0, len(audio), sr * segment_length)
            ]
            
            # Extract features for each segment
            features = []
            for segment in audio_segments:
                if len(segment) >= sr * 0.5:  # Skip very short segments
                    features.append(extract_audio_features(segment, sr))
            features = np.array(features)

            if len(features) < 2:
                print("Not enough segments for clustering. Skipping...")
                continue

            # Estimate number of speakers and cluster
            num_speakers = estimate_num_speakers(features)
            cluster_labels = cluster_speakers(features, num_speakers)

            # Save speaker-specific audio
            save_speaker_audio(cluster_labels, audio_segments, sr, os.path.splitext(file_name)[0], SPEAKER_OUTPUT_FOLDER)

if __name__ == "__main__":
    print("Starting Speaker Separation...")
    process_speaker_separation(SPEECH_SEGMENTS_FOLDER)
