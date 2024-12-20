import pyaudio
import wave
import datetime
import os
import time
import threading
import speech_recognition as sr
import keyboard  
from utils import ensure_folder_exists, preprocess_audio
from pyannote.audio import Pipeline

recording = False  
SECONDS_TO_RECORD_AFTER_DETECTION = 5
audio_folder = "recorded_audio"
transcription_folder = "transcriptions"
ensure_folder_exists(audio_folder)
ensure_folder_exists(transcription_folder)

audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

frames = []
print("Audio recording ready... Press 'q' to stop manually.")

recognizer = sr.Recognizer()
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

def transcribe_audio_live(audio_chunk, timestamp):
    try:
        audio_path = os.path.join(audio_folder, f"{timestamp}.wav")
        sound_file = wave.open(audio_path, "wb")
        sound_file.setnchannels(1)
        sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        sound_file.setframerate(44100)
        sound_file.writeframes(audio_chunk)
        sound_file.close()

        diarization = pipeline(audio_path)
        speaker_segments = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append((turn.start, turn.end))

        for speaker, segments in speaker_segments.items():
            speaker_audio = b""
            for start, end in segments:
                with wave.open(audio_path, "rb") as wf:
                    wf.setpos(int(start * 44100))
                    speaker_audio += wf.readframes(int((end - start) * 44100))

            speaker_audio_path = os.path.join(audio_folder, f"{timestamp}_{speaker}.wav")
            with wave.open(speaker_audio_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(44100)
                wf.writeframes(speaker_audio)

            with sr.AudioFile(speaker_audio_path) as source:
                audio_data = recognizer.record(source)
                transcription = recognizer.recognize_google(audio_data)
                print(f"({timestamp} - {speaker}): {transcription}")

                transcription_file_path = os.path.join(transcription_folder, f"{timestamp}_{speaker}_transcription.txt")
                with open(transcription_file_path, "w") as file:
                    file.write(transcription)
    except sr.UnknownValueError:
        print("---")
    except sr.RequestError as e:
        print(f"Error: Could not request results from the Google Speech Recognition service; {e}")
    except Exception as e:
        print(f"Error during transcription: {e}")

def record_and_transcribe():
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
            
            if len(frames) >= 44100 * 5 // 1024:  
                audio_chunk = b"".join(frames)
                frames = []  
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