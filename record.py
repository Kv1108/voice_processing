import pyaudio
import wave
import datetime
import os
import time
import keyboard  # For listening to key presses
from utils import ensure_folder_exists, preprocess_audio

# Detection variables
recording = False
recording_stopped_time = None
SECONDS_TO_RECORD_AFTER_DETECTION = 5
timer_started = False

# Folder setup for audio files
audio_folder = "recorded_audio"
ensure_folder_exists(audio_folder)

# Initialize PyAudio
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

frames = []
print("Audio recording ready... Press 'q' to stop manually.")

try:
    while True:
        # Read audio input
        data = stream.read(1024, exception_on_overflow=False)
        frames.append(data)
        
        # Simulate detection (you can replace this logic with actual detection logic if required)
        detection = True  # Assume we're always detecting audio input for now

        if detection:
            if recording:
                timer_started = False
            else:
                # Start recording
                recording = True
                timer_started = False
                timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
                audio_path = os.path.join(audio_folder, f"{timestamp}.wav")
                print(f"Started Recording: {audio_path}")
        elif recording:  # No detection
            if timer_started:
                if time.time() - recording_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                    # Stop recording and save the audio
                    recording = False
                    timer_started = False
                    sound_file = wave.open(audio_path, "wb")
                    sound_file.setnchannels(1)
                    sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
                    sound_file.setframerate(44100)
                    sound_file.writeframes(b"".join(frames))
                    sound_file.close()
                    print(f"Stopped Recording! Saved to: {audio_path}")

                    # Preprocess the audio
                    preprocessed_path = audio_path.replace(".wav", "_preprocessed.wav")
                    try:
                        preprocess_audio(audio_path, preprocessed_path)
                        print(f"Preprocessed Audio Saved to: {preprocessed_path}")
                    except Exception as e:
                        print(f"Error during preprocessing: {e}")
                    frames = []  # Clear frames for the next recording
            else:
                timer_started = True
                recording_stopped_time = time.time()

        # Stop recording manually by pressing 'q'
        if keyboard.is_pressed('q'):
            print("Manual stop detected. Finalizing...")
            if recording:
                # Save the current recording if active
                sound_file = wave.open(audio_path, "wb")
                sound_file.setnchannels(1)
                sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
                sound_file.setframerate(44100)
                sound_file.writeframes(b"".join(frames))
                sound_file.close()
                print(f"Stopped Recording! Saved to: {audio_path}")

                # Preprocess the audio
                preprocessed_path = audio_path.replace(".wav", "_preprocessed.wav")
                try:
                    preprocess_audio(audio_path, preprocessed_path)
                    print(f"Preprocessed Audio Saved to: {preprocessed_path}")
                except Exception as e:
                    print(f"Error during preprocessing: {e}")
            break

except KeyboardInterrupt:
    print("Recording stopped manually via Ctrl+C.")

# Cleanup
stream.stop_stream()
stream.close()
audio.terminate()
