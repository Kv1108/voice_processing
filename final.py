import os
import datetime
import threading
import pyaudio
import wave
import keyboard
import speech_recognition as sr
from utils import ensure_folder_exists
import torch
from speechbrain.pretrained import SpeakerRecognition

# Constants
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 2
CHUNK_SIZE = 1024
RATE = 44100

audio_folder = "recorded_audio"
transcription_folder = "transcriptions"
ensure_folder_exists(audio_folder)
ensure_folder_exists(transcription_folder)

audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)

frames = []
full_audio_frames = []
silent_chunks = 0
recognizer = sr.Recognizer()

speaker_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_model")
speaker_map = {}  # Maps audio chunks to speaker IDs
speaker_id_counter = 1  # Incremental counter for assigning speaker IDs

def save_audio(audio_data, filename):
    audio_path = os.path.join(audio_folder, f"{filename}.wav")
    with wave.open(audio_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(RATE)
        wf.writeframes(audio_data)
    return audio_path

def save_transcription(transcription_text, filename):
    transcription_path = os.path.join(transcription_folder, f"{filename}_transcription.txt")
    with open(transcription_path, "w") as f:
        f.write(transcription_text)
    print(f"Transcription saved: {transcription_path}")

def transcribe_audio(audio_path):
    """
    Transcribes the given audio and determines if it's intelligible.
    """
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            transcription = recognizer.recognize_google(audio_data)
            return transcription, True  # Intelligible audio
    except sr.UnknownValueError:
        return "[Unintelligible audio]", False  # Unintelligible
    except sr.RequestError as e:
        return f"[Error: {e}]", False
    except Exception as e:
        return f"[Error during transcription: {e}]", False


def is_silence(audio_chunk):
    amplitude = [int.from_bytes(audio_chunk[i:i+2], byteorder='little', signed=True) for i in range(0, len(audio_chunk), 2)]
    return all(abs(a) < SILENCE_THRESHOLD for a in amplitude)

def compare_speakers_advanced(audio_path1, audio_path2):
    """
    Compare two audio files to check if they belong to the same speaker using SpeechBrain.

    :param audio_path1: Path to the first audio file
    :param audio_path2: Path to the second audio file
    :return: Similarity score (higher means more likely the same person) and boolean indicating if they are the same speaker
    """
    # Load pre-trained speaker recognition model
    model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_model")

    # Compute similarity score between the two audio files
    score, prediction = model.verify_files(audio_path1, audio_path2)

    return score.item(), prediction  # Convert to scalar and return

def assign_speaker(audio_path, previous_chunks):
    """
    Assign a speaker label to the given audio chunk by comparing it with previous chunks.

    :param audio_path: Path to the current audio file
    :param previous_chunks: Dictionary containing paths to previous chunks and their speaker labels
    :return: Speaker label
    """
    for prev_path, speaker_id in previous_chunks.items():
        similarity_score, is_same_speaker = compare_speakers_advanced(prev_path, audio_path)
        if is_same_speaker:
            return speaker_id  # Return the existing speaker label if a match is found

    # If no match, assign a new speaker label
    new_speaker_id = len(previous_chunks) + 1
    previous_chunks[audio_path] = new_speaker_id
    return new_speaker_id


def process_audio_chunk(audio_chunk, filename, transcription_text, previous_chunks):
    audio_path = save_audio(audio_chunk, filename)
    transcription, is_intelligible = transcribe_audio(audio_path)

    if is_intelligible:
        speaker = assign_speaker(audio_path, previous_chunks)
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        transcription_text.append(f"[{timestamp}] [Speaker {speaker}]: {transcription}")
        print(f"[{timestamp}] [Speaker {speaker}]: {transcription}")
    else:
        print(f"Skipping unintelligible audio chunk: {filename}")

def record_and_transcribe():
    global frames, full_audio_frames, silent_chunks

    transcription_text = []
    filename_base = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    chunk_count = 1
    transcription_threads = []
    previous_chunks = {}

    try:
        while True:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            frames.append(data)
            full_audio_frames.append(data)

            if is_silence(data):
                silent_chunks += 1
            else:
                silent_chunks = 0

            if silent_chunks >= (RATE * SILENCE_DURATION // CHUNK_SIZE):
                if frames:
                    audio_chunk = b"".join(frames)
                    frames = []
                    chunk_filename = f"{filename_base}_chunk{chunk_count}"
                    thread = threading.Thread(target=process_audio_chunk, args=(audio_chunk, chunk_filename, transcription_text, previous_chunks))
                    thread.start()
                    transcription_threads.append(thread)
                    chunk_count += 1
                    silent_chunks = 0

            if keyboard.is_pressed('q'):
                break
    finally:
        for thread in transcription_threads:
            thread.join()

        full_audio_data = b"".join(full_audio_frames)
        save_audio(full_audio_data, filename_base)
        transcription_path = os.path.join(transcription_folder, f"{filename_base}_transcription.txt")
        save_transcription("\n".join(transcription_text), filename_base)

        # Display the content of the transcription file
        try:
            with open(transcription_path, "r") as f:
                print("\nFinal Transcription Content:")
                print(f.read())
        except FileNotFoundError:
            print(f"Error: Transcription file '{transcription_path}' not found.")


if __name__ == "__main__":
    try:
        record_and_transcribe()
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
