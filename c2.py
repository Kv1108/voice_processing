import librosa
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import soundfile as sf

# Function to reconstruct audio from clustered segments
def reconstruct_audio_from_clusters(audio, labels, hop_size, frame_size, sr, n_clusters):
    clustered_audio = {i: [] for i in range(n_clusters)}
    
    # Group frames by cluster label
    for idx, label in enumerate(labels):
        start_time = idx * hop_size / sr
        end_time = (idx * hop_size + frame_size) / sr
        frame_audio = audio[int(start_time * sr):int(end_time * sr)]
        clustered_audio[label].append(frame_audio)
    
    # Concatenate frames for each speaker (cluster)
    speaker_audio = {}
    for cluster, frames in clustered_audio.items():
        speaker_audio[cluster] = np.concatenate(frames)
    
    return speaker_audio

# Assuming you have already run clustering and obtained 'labels' and other variables
speaker_audio = reconstruct_audio_from_clusters(audio, labels, hop_size, frame_size, sr, n_clusters)

# Save audio for each speaker
for speaker_id, audio_data in speaker_audio.items():
    output_filename = f"speaker_{speaker_id}.wav"
    sf.write(output_filename, audio_data, sr)
    print(f"Saved speaker {speaker_id} audio to {output_filename}")
