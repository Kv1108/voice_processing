import pyaudio
import wave
import datetime
import os
import threading
import speech_recognition as sr
import keyboard
from utils import ensure_folder_exists

# Constants
SECONDS_TO_RECORD_AFTER_DETECTION = 5
audio_folder = "recorded_audio"
transcription_folder = "transcriptions"
ensure_folder_exists(audio_folder)
ensure_folder_exists(transcription_folder)

# Audio setup
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

frames = []
full_audio_frames = []
print("Audio recording ready... Press 'q' to stop manually.")

recognizer = sr.Recognizer()
recording = False
start_time = None

def save_full_audio(audio_data, filename):
    """Save full audio data to a WAV file."""
    audio_path = os.path.join(audio_folder, f"{filename}.wav")
    with wave.open(audio_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(audio_data)
    print(f"Full audio saved as: {audio_path}")


def save_full_transcription(transcription_text, filename):
    """Save full transcription to a text file."""
    transcription_file_path = os.path.join(transcription_folder, f"{filename}_transcription.txt")
    with open(transcription_file_path, "w") as file:
        file.write(transcription_text)
    print(f"Full transcription saved as: {transcription_file_path}")


def transcribe_audio_live(audio_chunk):
    """Transcribe audio chunk and return the transcription with timestamps."""
    try:
        with sr.AudioFile(audio_chunk) as source:
            audio_data = recognizer.record(source)
            transcription = recognizer.recognize_google(audio_data)
            return transcription
    except sr.UnknownValueError:
        return "[Unintelligible audio]"
    except sr.RequestError as e:
        return f"[Error: {e}]"
    except Exception as e:
        return f"[Error during transcription: {e}]"


def record_and_transcribe():
    global frames, full_audio_frames, recording, start_time

    transcription_text = ""  # Store the full transcription
    filename = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

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

            if len(frames) >= 44100 * SECONDS_TO_RECORD_AFTER_DETECTION // 1024:
                # Save intermediate transcription
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
    save_full_audio(full_audio_data, filename)
    save_full_transcription(transcription_text, filename)


if __name__ == "__main__":
    try:
        record_and_transcribe()
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
