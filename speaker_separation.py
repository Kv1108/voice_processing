import os
import numpy as np
from segmentation import extract_segments_and_embeddings
from clustering import cluster_embeddings
from assign_ids import assign_consistent_ids

def calculate_distances(embeddings):
    if len(embeddings) == 0:
        raise ValueError("Embeddings list is empty. Cannot compute distances.")
    embeddings = np.array(embeddings)
    if len(embeddings.shape) != 2:
        raise ValueError("Embeddings must be a 2-dimensional array.")
    distances = cdist(embeddings, embeddings, metric="cosine")
    return distances
    
def separate_speakers(audio_path, n_speakers=None):
    """
    Separate speakers and assign consistent IDs.
    
    Parameters:
        audio_path (str): Path to the input audio file.
        n_speakers (int, optional): Expected number of speakers. Defaults to None.
    
    Returns:
        speaker_segments (list): List of (speaker_id, start, end) tuples.
    """
    # Step 1: Extract embeddings and segments
    embeddings, segments = extract_segments_and_embeddings(audio_path)
    if embeddings is None or len(segments) == 0:
        print(f"No valid segments detected in {audio_path}.")
        return []

    # Step 2: Cluster embeddings to group similar speakers
    cluster_labels = cluster_embeddings(embeddings, n_speakers=n_speakers)

    # Step 3: Assign consistent speaker IDs
    consistent_ids = assign_consistent_ids(cluster_labels)

    # Step 4: Combine speaker labels with timestamps
    speaker_segments = []
    for idx, (start, end) in enumerate(segments):
        speaker_segments.append((consistent_ids[idx], start, end))
        print(f"Speaker {consistent_ids[idx]}: {start:.2f}s - {end:.2f}s")

    return speaker_segments

def save_transcription(speaker_segments, audio_path, output_folder):
    """
    Save the separated speaker transcription to a file.
    
    Parameters:
        speaker_segments (list): List of (speaker_id, start, end) tuples.
        audio_path (str): Path to the input audio file.
        output_folder (str): Folder to save the transcription.
    """
    # Prepare transcription text
    transcription_text = []
    for speaker_id, start, end in speaker_segments:
        transcription_text.append(f"Speaker {speaker_id}: {start:.2f}-{end:.2f}")

    # Save to file
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, os.path.basename(audio_path).replace(".wav", "_transcription.txt"))
    with open(output_file, "w") as f:
        f.write("\n".join(transcription_text))
    print(f"Transcription saved to {output_file}")

# Example usage
if __name__ == "__main__":
    audio_file = "recorded_audio/example_preprocessed.wav"
    output_dir = "transcriptions"

    speaker_segments = separate_speakers(audio_file, n_speakers=2)  # Adjust n_speakers as needed
    save_transcription(speaker_segments, audio_file, output_dir)
import numpy as np
from scipy.spatial.distance import cdist

def calculate_distances(embeddings):
    # Check the shape of embeddings
    print(f"Shape of embeddings: {np.shape(embeddings)}")
    
    if len(embeddings.shape) != 2:
        raise ValueError("Embeddings must be a 2-dimensional array.")
    
    distances = cdist(embeddings, embeddings, metric="cosine")
    return distances

# ...existing code...