import pyaudio
import wave
import datetime
import os
import time
import threading
import speech_recognition as sr
import keyboard  
from utils import ensure_folder_exists, preprocess_audio

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


def transcribe_audio_live(audio_chunk, timestamp):

    try:
        audio_path = os.path.join(audio_folder, f"{timestamp}.wav")
        sound_file = wave.open(audio_path, "wb")
        sound_file.setnchannels(1)
        sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        sound_file.setframerate(44100)
        sound_file.writeframes(audio_chunk)
        sound_file.close()

        with sr.AudioFile(audio_path) as source:
            #print("Processing audio chunk for transcription...")
            audio_data = recognizer.record(source)
            transcription = recognizer.recognize_google(audio_data)
            print(f"({timestamp}): {transcription}")

            transcription_file_path = os.path.join(transcription_folder, f"{timestamp}_transcription.txt")
            with open(transcription_file_path, "w") as file:
                file.write(transcription)
            #print(f"Transcription saved to '{transcription_file_path}'")
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
