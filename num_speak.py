import os
import numpy as np
import librosa
from pyannote.audio import Pipeline

def get_latest_audio_file(folder_path):
    audio_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in folder: {folder_path}")
    
    audio_files.sort(key=lambda f: os.path.getmtime(os.path.join(folder_path, f)), reverse=True)
    return os.path.join(folder_path, audio_files[0])

audio_folder = "recorded_audio"
audio_file = get_latest_audio_file(audio_folder)
audio, sr = librosa.load(audio_file, sr=None)

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="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")

diarization = pipeline({"uri": "audio", "audio": audio_file})

def extract_speaker_embeddings(diarization):
    embeddings = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        embeddings.append((turn.start, turn.end, speaker))
    return embeddings

embeddings = extract_speaker_embeddings(diarization)

num_speakers = len(set([embedding[2] for embedding in embeddings]))

print(f"Estimated number of speakers: {num_speakers}")
