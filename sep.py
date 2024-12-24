import os
import wave
import datetime
import librosa
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import pyaudio
import speech_recognition as sr
import keyboard
from utils import ensure_folder_exists

# Constants
audio_folder = "recorded_audio"
transcription_folder = "transcriptions"
processed_audio_folder = "processed_audio"
ensure_folder_exists(audio_folder)
ensure_folder_exists(transcription_folder)
ensure_folder_exists(processed_audio_folder)

# Audio setup
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

frames = []
full_audio_frames = []
recognizer = sr.Recognizer()
recording = False
start_time = None

# Function to save full audio data
def save_full_audio(audio_data, filename):
    audio_path = os.path.join(audio_folder, f"{filename}.wav")
    with wave.open(audio_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(audio_data)
    print(f"Full audio saved as: {audio_path}")
    return audio_path

# Function to save transcription
def save_full_transcription(transcription_text, filename):
    transcription_file_path = os.path.join(transcription_folder, f"{filename}_transcription.txt")
    with open(transcription_file_path, "w") as file:
        file.write(transcription_text)
    print(f"Full transcription saved as: {transcription_file_path}")
    return transcription_file_path

# Function to extract MFCC features
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)

# Function to estimate the number of speakers and identify them
def estimate_and_identify_speakers(audio_path):
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    # Split audio into smaller chunks
    chunk_duration = 2  # 2 seconds
    n_chunks = int(duration // chunk_duration)
    features = []
    timestamps = []

    for i in range(n_chunks):
        start = i * chunk_duration
        end = start + chunk_duration
        chunk = y[int(start * sr):int(end * sr)]
        if len(chunk) > 0:
            mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
            features.append(np.mean(mfcc, axis=1))
            timestamps.append(f"{datetime.timedelta(seconds=int(start))} - {datetime.timedelta(seconds=int(end))}")

    if not features:
        raise ValueError("No features extracted. Check the audio file for valid chunks.")

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Determine the optimal number of speakers using the Bayesian Information Criterion (BIC)
    bic_scores = []
    n_components_range = range(1, 10)
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(features_scaled)
        bic_scores.append(gmm.bic(features_scaled))

    optimal_n_speakers = np.argmin(bic_scores) + 1

    # Perform clustering with the optimal number of speakers
    gmm = GaussianMixture(n_components=optimal_n_speakers, random_state=42)
    labels = gmm.fit_predict(features_scaled)

    return labels, timestamps, optimal_n_speakers

# Function to transcribe and display speakers in live format
def process_audio_with_speakers(audio_path, transcription_path):
    labels, timestamps, n_speakers = estimate_and_identify_speakers(audio_path)

    with open(transcription_path, "r") as f:
        transcriptions = f.readlines()

    print(f"\n[Output with {n_speakers} Speakers]\n")
    for idx, (label, timestamp) in enumerate(zip(labels, timestamps)):
        speaker = f"Speaker {label + 1}"
        transcription = transcriptions[idx].strip() if idx < len(transcriptions) else ""
        print(f"[{timestamp}] [{speaker}] {transcription}")

# Function to record and process audio
def record_and_process_audio():
    global frames, full_audio_frames, recording, start_time

    transcription_text = ""  # Store the full transcription
    filename = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    print("Audio recording ready... Press 'q' to stop manually.")

    while True:
        try:
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)
            full_audio_frames.append(data)

            # Detect voice activity (dummy detection logic)
            detection = True

            if detection:
                if not recording:
                    recording = True
                    start_time = datetime.datetime.now()
                    print(f"Started Recording: {filename}.wav")

            if len(frames) >= 44100 * 5 // 1024:  # Process every 5 seconds
                audio_chunk = b"".join(frames)
                frames = []

                # Save the audio chunk to a temporary file for transcription
                temp_audio_path = os.path.join(audio_folder, f"temp_{filename}.wav")
                with wave.open(temp_audio_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(44100)
                    wf.writeframes(audio_chunk)

                # Transcribe the audio chunk
                transcription = transcribe_audio_live(temp_audio_path)
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                transcription_text += f"[{timestamp}] {transcription}\n"
                print(f"[{timestamp}] {transcription}")

                # Clean up temporary file
                os.remove(temp_audio_path)

            if keyboard.is_pressed('q'):
                print("Manual stop detected. Finalizing...")
                break
        except KeyboardInterrupt:
            print("Recording stopped manually via Ctrl+C.")
            break
        except Exception as e:
            print(f"Error in recording loop: {e}")
            break

    # Save the full audio and transcription
    full_audio_data = b"".join(full_audio_frames)
    audio_path = save_full_audio(full_audio_data, filename)
    transcription_path = save_full_transcription(transcription_text, filename)

    # Process audio with speakers
    process_audio_with_speakers(audio_path, transcription_path)

if __name__ == "__main__":
    try:
        record_and_process_audio()
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
