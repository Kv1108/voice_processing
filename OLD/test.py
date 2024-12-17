import os
import numpy as np
from pyannote.audio import Pipeline
from pyannote.core import Segment
import whisper
import soundfile as sf
from pydub import AudioSegment
import subprocess


# HUGGING_FACE_TOKEN = "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"
os.environ['HUGGING_FACE_TOKEN'] = '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'
def ensure_folder_exists(folder_name):
    """Ensure the folder exists or create it."""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")

def preprocess_audio(input_file, output_file):
    """Preprocess audio to ensure compatibility (16kHz, mono)."""
    try:
        result = subprocess.run([
            "ffmpeg", "-i", input_file, "-ac", "1", "-ar", "16000", output_file, "-y"
        ], capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode())
        print(f"Preprocessed audio saved to {output_file}")
    except FileNotFoundError:
        print("FFmpeg not found. Please install FFmpeg and add it to your PATH.")
        raise

def diarize_audio(audio_file):
    
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HUGGING_FACE_TOKEN)
    except Exception as e:
        raise RuntimeError("Failed to load pyannote pipeline. Check your Hugging Face token.") from e

    diarization_result = pipeline(audio_file)

    clustered_segments = {}
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        if speaker not in clustered_segments:
            clustered_segments[speaker] = []
        clustered_segments[speaker].append((turn.start, turn.end))

    return clustered_segments

def save_diarized_audio(cluster_segments, audio_file, output_folder):
    """Save each speaker's segments into separate audio files."""
    signal, sampling_rate = sf.read(audio_file)
    ensure_folder_exists(output_folder)

    for speaker, segments in cluster_segments.items():
        speaker_audio = []
        for start, end in segments:
            start_sample = int(start * sampling_rate)
            end_sample = int(end * sampling_rate)
            speaker_audio.append(signal[start_sample:end_sample])

        if speaker_audio:
            speaker_audio = np.concatenate(speaker_audio)
            speaker_file = os.path.join(output_folder, f"speaker_{speaker}.wav")
            sf.write(speaker_file, speaker_audio, sampling_rate)
            print(f"Saved audio for {speaker} to {speaker_file}")

def transcribe_audio(audio_file, model):
    """Transcribe audio using Whisper model."""
    result = model.transcribe(audio_file)
    return result["text"]

def save_transcriptions(transcriptions, output_folder):
    """Save transcriptions to text files."""
    ensure_folder_exists(output_folder)
    for speaker, transcription in transcriptions.items():
        file_path = os.path.join(output_folder, f"{speaker}_transcription.txt")
        with open(file_path, "w") as f:
            f.write(transcription)
        print(f"Transcription for {speaker} saved to {file_path}")

def main():
    audio_folder = "recorded_audio"
    processed_audio_folder = "processed_audio"
    diarized_audio_folder = "diarized_audio"
    transcription_folder = "transcriptions"

    ensure_folder_exists(processed_audio_folder)

    # Get the latest audio file
    try:
        latest_audio_file = max(
            [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith(".wav")],
            key=os.path.getmtime
        )
    except ValueError:
        raise RuntimeError(f"No audio files found in {audio_folder}. Please add a .wav file.")

    print(f"Processing latest audio file: {latest_audio_file}")
    processed_audio_file = os.path.join(processed_audio_folder, "processed_audio.wav")

    # Preprocess audio
    preprocess_audio(latest_audio_file, processed_audio_file)

    # Perform diarization
    clustered_segments = diarize_audio(processed_audio_file)

    # Save diarized audio
    save_diarized_audio(clustered_segments, processed_audio_file, diarized_audio_folder)

    # Load Whisper model
    print("Loading Whisper model...")
    whisper_model = whisper.load_model("medium")

    # Transcribe each speaker's audio
    transcriptions = {}
    for speaker_file in os.listdir(diarized_audio_folder):
        if speaker_file.endswith(".wav"):
            speaker_path = os.path.join(diarized_audio_folder, speaker_file)
            transcriptions[speaker_file] = transcribe_audio(speaker_path, whisper_model)

    # Save transcriptions
    save_transcriptions(transcriptions, transcription_folder)

if __name__ == "__main__":
    main()
