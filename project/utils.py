import os
import time
import numpy as np
import sounddevice as sd
import wave
import torch
import whisper
from datetime import datetime

# Helper function to handle file uploads
def save_uploaded_file(uploaded_file, save_path="uploads"):
    """Save the uploaded file to the specified directory."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_path = os.path.join(save_path, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Function to record audio (for live input)
def record_audio(duration=10, sample_rate=16000):
    """Record audio for a specified duration and sample rate."""
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # Wait for the recording to finish
    return audio

# Save audio to WAV file
def save_audio_to_wav(audio_data, file_name="output.wav", sample_rate=16000):
    """Save recorded audio data to a WAV file."""
    with wave.open(file_name, "wb") as wf:
        wf.setnchannels(1)  # Mono audio
        wf.setsampwidth(2)  # 16-bit samples
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)

# Get the current timestamp
def get_timestamp():
    """Get the current timestamp."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Function to convert seconds to a timestamp (for speaker diarization)
def seconds_to_timestamp(seconds):
    """Convert seconds into a timestamp format."""
    return str(datetime.timedelta(seconds=seconds))

# Initialize Whisper model (for transcription)
def load_whisper_model(model_size="base"):
    """Load the Whisper model for transcription."""
    model = whisper.load_model(model_size)
    return model

# Function to transcribe audio
def transcribe_audio(audio_path, model):
    """Transcribe audio using the Whisper model."""
    result = model.transcribe(audio_path)
    return result['text']

# Function to apply VAD (Voice Activity Detection)
def apply_vad(audio, sample_rate=16000, vad_model=None):
    """Apply VAD to segment speech from non-speech."""
    if vad_model is None:
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
        (get_speech_timestamps, _, _, _, _) = utils
    speech_timestamps = get_speech_timestamps(audio, model, sampling_rate=sample_rate)
    return speech_timestamps

# Helper function to format speaker diarization results
def format_diarization_results(diarization):
    """Format speaker diarization results into readable format."""
    formatted_results = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        formatted_results.append(f"Speaker {speaker} spoke from {turn.start:.1f}s to {turn.end:.1f}s")
    return formatted_results

# Function to store transcription logs in a text file
def store_transcription_log(transcription, speaker_labels, timestamps, log_file="transcription_log.txt"):
    """Store transcription, speaker labels, and timestamps in a text file."""
    with open(log_file, "a") as log:
        log.write(f"{get_timestamp()} - Transcription:\n{transcription}\n")
        log.write(f"Speaker Labels: {speaker_labels}\n")
        log.write(f"Timestamps: {timestamps}\n\n")
    print("Transcription log saved.")

# Function to clean and preprocess audio file
def preprocess_audio(audio_file, target_sample_rate=16000):
    """Preprocess audio (resample if necessary)."""
    # Load audio and resample if needed (for simplicity, skipping here)
    # Preprocessing steps such as noise reduction or normalization can be added.
    print(f"Preprocessing audio file: {audio_file}")
    return audio_file  # In real use case, you may resample or clean the file here.
