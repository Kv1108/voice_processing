import os
import librosa
import numpy as np
import soundfile as sf
from pyannote.audio import Pipeline
import speech_recognition as sr

audio_folder = "recorded_audio"
output_folder = "separated_voices"
transcription_folder = "transcriptions"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(transcription_folder, exist_ok=True)

def get_latest_audio_file(folder_path):
    audio_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in folder: {folder_path}")
    
    audio_files.sort(key=lambda f: os.path.getmtime(os.path.join(folder_path, f)), reverse=True)
    return os.path.join(folder_path, audio_files[0])

audio_file = get_latest_audio_file(audio_folder)
y, sr = librosa.load(audio_file, sr=None)

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="YOUR_AUTH_TOKEN")

diarization = pipeline({"uri": "audio", "audio": audio_file})

def extract_speaker_segments(diarization, audio, sr):
    speaker_segments = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speaker_segments:
            speaker_segments[speaker] = []
        start_sample = int(turn.start * sr)
        end_sample = int(turn.end * sr)
        speaker_segments[speaker].append(audio[start_sample:end_sample])
    return speaker_segments

speaker_segments = extract_speaker_segments(diarization, y, sr)

for speaker, segments in speaker_segments.items():
    combined_audio = np.concatenate(segments)
    output_path = os.path.join(output_folder, f"{speaker}.wav")
    sf.write(output_path, combined_audio, sr)
    print(f"Saved: {output_path}")

recognizer = sr.Recognizer()
transcription_file_path = os.path.join(transcriptions, "transcription.txt")

with open(transcription_file_path, "w") as transcriptions:
    for speaker, segments in speaker_segments.items():
        combined_audio = np.concatenate(segments)
        output_path = os.path.join(output_folder, f"{speaker}.wav")
        with sr.AudioFile(output_path) as source_audio:
            audio_data = recognizer.record(source_audio)
            transcription = recognizer.recognize_google(audio_data)
            timestamp = librosa.get_duration(y=combined_audio, sr=sr)
            transcriptions.write(f"[{timestamp}] [{speaker}] {transcription}\n")
            print(f"[{timestamp}] [{speaker}] {transcription}")

print(f"Transcription saved to {transcription_file_path}")
