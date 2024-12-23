import pyaudio
import wave
import datetime
import os
import time
import threading
import speech_recognition as sr
import keyboard
from utils import ensure_folder_exists, preprocess_audio
from pyannote.audio import Pipeline  # Diarization pipeline

# Initialize folders for audio and transcriptions
audio_folder = "recorded_audio"
transcription_folder = "transcriptions"
ensure_folder_exists(audio_folder)
ensure_folder_exists(transcription_folder)

# PyAudio for audio streaming
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

frames = []
recording = False
recognizer = sr.Recognizer()

# PyAnnote Diarization pipeline
pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization')  # Pre-trained diarization model


def diarize_audio(audio_path):
    """Runs diarization on the recorded audio and returns a mapping of speaker labels."""
    try:
        diarization = pipeline(audio_path)
        speaker_mapping = {}

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start_time = round(turn.start, 2)
            end_time = round(turn.end, 2)
            if speaker not in speaker_mapping:
                speaker_mapping[speaker] = f"speaker_{len(speaker_mapping) + 1}"
            print(f"[{start_time}s - {end_time}s] {speaker_mapping[speaker]}")
        
        return diarization, speaker_mapping

    except Exception as e:
        print(f"Error during diarization: {e}")
        return None, {}


def transcribe_audio_live(audio_chunk, timestamp):
    """Saves and transcribes the audio with speaker diarization."""
    try:
        audio_path = os.path.join(audio_folder, f"{timestamp}.wav")
        sound_file = wave.open(audio_path, "wb")
        sound_file.setnchannels(1)
        sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        sound_file.setframerate(44100)
        sound_file.writeframes(audio_chunk)
        sound_file.close()

        # Run speaker diarization
        diarization, speaker_mapping = diarize_audio(audio_path)

        if not diarization:
            print("Diarization failed, skipping transcription.")
            return

        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            transcription = recognizer.recognize_google(audio_data)

            # Extract speaker timestamps from diarization
            formatted_output = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start_time = datetime.timedelta(seconds=round(turn.start))
                end_time = datetime.timedelta(seconds=round(turn.end))
                speaker_label = speaker_mapping.get(speaker, "unknown_speaker")
                text_segment = transcription  # For simplicity, using full transcription
                formatted_output.append(f"({start_time} - {end_time}): {{{speaker_label}}}: {text_segment}")

            transcription_file_path = os.path.join(transcription_folder, f"{timestamp}_transcription.txt")
            with open(transcription_file_path, "w") as file:
                file.write("\n".join(formatted_output))
            
            print(f"Transcription saved to '{transcription_file_path}'")

    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
    except Exception as e:
        print(f"Error during transcription: {e}")


def record_and_transcribe():
    """Continuously record audio and process it in 5-second chunks."""
    global frames
    global recording

    while True:
        try:
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)
            detection = True

            if detection:
                if not recording:
                    recording = True
                    timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
                    print(f"Started Recording: {timestamp}.wav")

            if len(frames) >= 44100 * 5 // 1024:  # 5-second chunks
                audio_chunk = b"".join(frames)
                frames = []  # Clear frames for the next 5-second chunk
                threading.Thread(target=transcribe_audio_live, args=(audio_chunk, timestamp)).start()

            if keyboard.is_pressed('q'):
                print("Manual stop detected. Finalizing...")
                break

        except KeyboardInterrupt:
            print("Recording stopped manually via Ctrl+C.")
            break
        except Exception as e:
            print(f"Error in recording loop: {e}")
            break


if __name__ == "__main__":
    try:
        record_and_transcribe()
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
