# diarize.py
from pyannote.audio import Pipeline
import torch
import os

# Load the pre-trained speaker diarization pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

def diarize_audio(audio_path):
    """
    This function takes an audio file path, performs speaker diarization using pyannote-audio,
    and returns the speaker segments with their respective timestamps.

    Parameters:
        audio_path (str): Path to the audio file to be processed for diarization.

    Returns:
        diarization: Diarization object containing speaker segments and timestamps.
    """
    try:
        # Check if the audio file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"The audio file at {audio_path} was not found.")

        # Perform speaker diarization
        diarization = pipeline(audio_path)
        
        # Display speaker segments and their timestamps
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            print(f"Speaker {speaker} spoke from {turn.start:.1f}s to {turn.end:.1f}s")
        
        return diarization

    except Exception as e:
        print(f"An error occurred during speaker diarization: {e}")
        return None
